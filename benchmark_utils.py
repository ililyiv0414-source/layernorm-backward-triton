import statistics
import time
import torch

def measure(calls):
    for call in calls.values():
        for _ in range(20):
            call()
    torch.cuda.synchronize()
    samples = {k:[] for k in calls}
    for trial in range(12):
        names = list(calls)
        shift = trial % len(names)
        for name in names[shift:]+names[:shift]:
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(30):
                calls[name]()
            b.record()
            b.synchronize()
            samples[name].append(a.elapsed_time(b)*1000/30)
    med = {k:statistics.median(v) for k,v in samples.items()}
    return dict(samples_us=samples,median_us=med,
                ratio_to_native={k:v/med['native'] for k,v in med.items()})

def host_times(calls):
    # Separate host enqueue time from synchronized batch wall time.
    result={}
    for name,call in calls.items():
        submit,complete=[],[]
        for _ in range(5):
            torch.cuda.synchronize()
            begin=time.perf_counter()
            for _ in range(30): call()
            enqueued=time.perf_counter()
            torch.cuda.synchronize()
            end=time.perf_counter()
            submit.append((enqueued-begin)*1e6/30)
            complete.append((end-begin)*1e6/30)
        result[name]=dict(host_enqueue_us=statistics.median(submit),
                          synchronized_batch_us=statistics.median(complete))
    return result
