import json
import time
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent/f'validation_{time.time_ns()}.json'
report={'checks': [], 'cases': [], 'status': 'running'}

def save():
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')

def check(a, b, label):
    a, b = a.float(), b.float()
    diff = (a-b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    bad = int((diff > 0.03125 + 0.01*b.abs()).sum())
    item = dict(label=label, max_abs=diff.max().item(),
                relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                mismatches=bad, finite=finite, atol=0.03125, rtol=0.01)
    report['checks'].append(item)
    print(json.dumps(item), flush=True)
    save()
    if bad or not finite:
        raise AssertionError(label)
