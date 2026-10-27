# Triton LayerNorm Backward for Wide Features

An experimental BF16 LayerNorm backward implementation on NVIDIA A800, focused on compilation cost, register pressure, and reproducible performance measurements.

**Scope:** backward-only research prototype. It is not a Triton compiler fix, a full training layer, or a universally faster replacement for PyTorch. No license grant has been selected for this research prototype.

## Results

On A800-SXM4-80GB, PyTorch 2.8.0+cu128, Triton 3.8.0:

| Experiment | Baseline | Tiled implementation |
|---|---:|---:|
| Median kernel compilation, original rows=4 vs rewritten kernels | 102.79 s | 1.51 s |
| Packaged API eager backward, 1152×36864 BF16 | 408.71 μs | 378.74 μs |
| Packaged API CUDA Graph backward, same shape | 405.95 μs | 371.69 μs |

Compilation is the sum of Triton warmup calls with fresh Triton caches over three independent processes; it excludes setup, GPU execution, and the original wrapper's final eager PyTorch reductions. It is **not** end-to-end first-use latency. Runtime includes all backward stages but excludes forward and compilation, using repeatedly resident inputs. Graph replay reuses captured storage.

All four tiled kernels reported zero spills. A fixed-configuration sweep found wins in 5/18 eager cases and 15/18 Graph cases; small eager shapes can be substantially slower. See [technical report](docs/technical-report.md) and [independent package acceptance](docs/acceptance-v1.md).

## Design

1. Compute partial row sums of `dy * weight` and `dy * weight * normalized_x` in column tiles.
2. Combine these small partial sums and reread each tile to compute `dx`.
3. Compute grouped partial `dweight` and `dbias` reductions.
4. Reduce the groups and write BF16 parameter gradients.

This exchanges extra launches and reads for a smaller live working set and more parallel tasks. It does not imply that every performance improvement is caused solely by eliminating spills.

## Environment and reproduction

Tested on Linux x86_64, Python 3.10.19, A800, torch 2.8.0+cu128 and triton 3.8.0 in an isolated experimental environment. This is a tested combination, not a claim of official dependency compatibility. No wheels are included. Reuse the validated environment instead of overwriting shared installations; torch.compile is not tested.

```bash
# Set PY to your existing validated Python executable and select an available GPU.
PY=/path/to/validated/environment/bin/python
CUDA_VISIBLE_DEVICES=0 "$PY" -u validate_api.py
# Continue only if validation passes.
CUDA_VISIBLE_DEVICES=0 "$PY" -u benchmark_api.py
```

Both scripts save timestamped JSON with errors, raw timing samples, environment and status. They require only this repository plus installed PyTorch/Triton. These are executable GPU regression checks rather than a pytest suite. The independent package passed these checks; the rearranged public-repository tree has only been statically checked so far.

The cold-compile and broad shape experiments are historical evidence described in the report; this minimal repository does not yet contain their entire runner set or original reproducer. Do not imply that the two commands above reproduce the 68× compilation result.

## API

```python
from layernorm_api import backward, PreparedBackward

# CUDA, contiguous BF16 x/dy: [M,N], weight: [N].
# Saved mean/rstd: one FP32 value per row, same device; detach autograd tensors.
dx, dw, db = backward(x, dy, weight, mean, rstd, backend="auto")

prepared = PreparedBackward(x, dy, weight, mean, rstd)
dx, dw, db = prepared.run()
```

`auto` uses tiled kernels only for the tested A800 sm80 / torch 2.8.0 / triton 3.8.0 / 1152×36864 case. Other supported inputs use native PyTorch. Explicit `backend="tiled"` supports N≤65536 experimentally; neither correctness nor speed is guaranteed for every possible input.

The native fallback adds validation, dispatch and zero-bias initialization overhead. It is not performance-neutral. Prefer direct PyTorch for latency-sensitive small eager shapes.

Prepared tiled outputs alias reusable buffers and are overwritten on the next call. Keep input storage/shapes fixed, clone outputs when persistence is required, and do not share an instance across concurrent streams or in-flight calls. Native prepared execution still allocates outputs. Higher-order autograd, noncontiguous layouts, other dtypes, other GPU families and end-to-end training are not validated.

## Provenance and license

Research started from [Triton issue #8908](https://github.com/triton-lang/triton/issues/8908). See [SOURCES.md](SOURCES.md) for attribution and unresolved licensing questions. No original gist source, credentials, binary dependencies or raw user transcripts are included. No license grant is made until the owner chooses a license and provenance review is completed.
