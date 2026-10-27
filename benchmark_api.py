"""Measure final API overhead; preserve raw samples and correctness checks."""
import json
import time
from pathlib import Path
import torch
import triton
import validation_support as shared
from benchmark_utils import measure
from benchmark_utils import host_times
from layernorm_api import backward,PreparedBackward,POLICY_VERSION
from split_layernorm import backward_tiled

def run(m,n):
    torch.manual_seed(404)
    x=torch.randn((m,n),device='cuda',dtype=torch.bfloat16)
    dy=torch.randn_like(x)
    w=torch.randn(n,device='cuda',dtype=torch.bfloat16)
    bias=torch.zeros_like(w)
    _,mu,rs=torch.ops.aten.native_layer_norm.default(x,[n],w,bias,1e-5)
    mu,rs=mu.contiguous(),rs.contiguous()
    def native():
        return torch.ops.aten.native_layer_norm_backward.default(dy,x,[n],mu,rs,w,bias,[True,True,True])
    prepared=PreparedBackward(x,dy,w,mu,rs)
    calls={'native':native,
           'api_auto':lambda:backward(x,dy,w,mu,rs),
           'api_native':lambda:backward(x,dy,w,mu,rs,'native'),
           'prepared_auto':prepared.run}
    if prepared.backend=='tiled':
        calls['tiled_direct']=lambda:backward_tiled(x,dy,w,mu,rs,4096)
    ref=native()
    print(f'CASE {m}x{n} selected={prepared.backend}',flush=True)
    for name,call in calls.items():
        for grad,a,b in zip(('dx','dw','db'),call(),ref):
            shared.check(a,b,f'{m}x{n},{name},{grad}')
    eager=measure(calls)
    host=host_times(calls)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls.values():
            for _ in range(3): call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs,outputs={},{}
    for name,call in calls.items():
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): outputs[name]=call()
        graphs[name]=graph
        graph.replay()
    torch.cuda.synchronize()
    for name,out in outputs.items():
        for grad,a,b in zip(('dx','dw','db'),out,ref):
            shared.check(a,b,f'graph,{m}x{n},{name},{grad}')
    graph_result=measure({name:g.replay for name,g in graphs.items()})
    record=dict(m=m,n=n,selected=prepared.backend,eager=eager,host=host,cuda_graph=graph_result)
    shared.report['cases'].append(record)
    shared.save()
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':
    shared.OUT=Path(__file__).resolve().parent/f'api_benchmark_{time.time_ns()}.json'
    shared.report['environment']=dict(torch=torch.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name())
    shared.report['policy']=POLICY_VERSION
    shared.report['scope']='backward only; constructor/setup excluded; eager includes per-call wrapper and allocation overhead; graph replay reuses storage; direct native receives a preexisting bias tensor while API native creates a zero bias per call; CPU enqueue diagnostic is separate'
    print('REPORT '+str(shared.OUT),flush=True)
    try:
        assert triton.__version__=='3.8.0'
        for m,n in ((33,4096),(64,12288),(1152,36864)):
            run(m,n)
        shared.report['status']='PASS'
    except Exception as exc:
        shared.report.update(status='FAIL',error=repr(exc))
        raise
    finally: shared.save()
    print('ALL PASS '+str(shared.OUT),flush=True)
