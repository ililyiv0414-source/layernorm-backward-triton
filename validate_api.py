"""Held-out shape/seed and API contract checks. No benchmark-based policy changes."""
import json
import time
from pathlib import Path
import torch
import triton
import validation_support as shared
from layernorm_api import backward,PreparedBackward,select_backend,POLICY_VERSION

def run(m,n,seed):
    torch.manual_seed(seed)
    x=torch.randn((m,n),device='cuda',dtype=torch.bfloat16)
    dy=torch.randn_like(x)
    w=torch.randn(n,device='cuda',dtype=torch.bfloat16)
    bias=torch.zeros_like(w)
    _,mu,rs=torch.ops.aten.native_layer_norm.default(x,[n],w,bias,1e-5)
    mu,rs=mu.contiguous(),rs.contiguous()
    prepared=PreparedBackward(x,dy,w,mu,rs)
    chosen=select_backend(x)
    if (m,n)!=(1152,36864): assert chosen=='native'
    print(f'CASE {m}x{n} seed={seed} selected={chosen}',flush=True)
    for repeat in range(2):
        if repeat: dy.normal_()
        ref=torch.ops.aten.native_layer_norm_backward.default(dy,x,[n],mu,rs,w,bias,[True,True,True])
        for name,call in (
            ('auto',lambda:backward(x,dy,w,mu,rs)),
            ('native',lambda:backward(x,dy,w,mu,rs,'native')),
            ('forced_tiled',lambda:backward(x,dy,w,mu,rs,'tiled')),
            ('prepared_auto',prepared.run)):
            for grad,a,b in zip(('dx','dw','db'),call(),ref):
                shared.check(a,b,f'{m}x{n},seed={seed},repeat={repeat},{name},{grad}')
    # Fail explicitly rather than silently accepting an invalid contract.
    try:
        backward(x,dy,w,mu.double(),rs)
    except ValueError: pass
    else: raise AssertionError('FP64 saved stats were not rejected')
    try:
        backward(x,dy,w,mu,rs,'unknown')
    except ValueError: pass
    else: raise AssertionError('invalid backend was not rejected')
    shared.report['cases'].append(dict(m=m,n=n,seed=seed,backend=chosen,status='PASS'))
    shared.save()

if __name__=='__main__':
    shared.OUT=Path(__file__).resolve().parent/f'api_validation_{time.time_ns()}.json'
    shared.report['policy']=POLICY_VERSION
    shared.report['environment']=dict(torch=torch.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name())
    print('REPORT '+str(shared.OUT),flush=True)
    try:
        for case in ((64,12288,101),(257,24576,202),(65,4103,303),(1152,36864,101),(1152,36864,202)):
            run(*case)
        shared.report['status']='PASS'
    except Exception as exc:
        shared.report.update(status='FAIL',error=repr(exc))
        raise
    finally: shared.save()
    print('ALL PASS '+str(shared.OUT),flush=True)
