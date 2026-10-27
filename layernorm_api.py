"""Experimental backward-only API. Forward must supply FP32 saved statistics.

Auto policy is deliberately an exact-match allowlist, not a general speed claim.
"""
import torch
import triton
from split_layernorm import backward_tiled
from tiled_workspace import TiledWorkspace

POLICY_VERSION = 'a800-bf16-1152x36864-v1'

def _validate(x,dy,w,mean,rstd):
    if not x.is_cuda or x.ndim!=2 or not x.is_contiguous():
        raise ValueError('x must be a contiguous 2D CUDA tensor')
    m,n=x.shape
    if m<1 or n<1 or dy.shape!=x.shape or w.shape!=(n,):
        raise ValueError('invalid input, gradient or weight shape')
    if any(v.device!=x.device or not v.is_contiguous() for v in (dy,w,mean,rstd)):
        raise ValueError('all tensors must be contiguous and on the same CUDA device')
    if x.dtype!=torch.bfloat16 or dy.dtype!=x.dtype or w.dtype!=x.dtype:
        raise ValueError('this API currently supports BF16 inputs/weight only')
    if mean.dtype!=torch.float32 or rstd.dtype!=torch.float32 or mean.numel()!=m or rstd.numel()!=m:
        raise ValueError('mean/rstd must contain one FP32 value per row')
    if torch.is_grad_enabled() and any(v.requires_grad for v in (x,dy,w,mean,rstd)):
        raise ValueError('backward-only API does not support higher-order autograd; detach saved tensors')

def select_backend(x,backend='auto'):
    if backend not in ('auto','native','tiled'):
        raise ValueError('backend must be auto, native or tiled')
    if backend!='auto':
        if backend=='tiled' and not (1<=x.shape[1]<=65536):
            raise ValueError('tiled backend requires N <= 65536')
        return backend
    props=torch.cuda.get_device_properties(x.device)
    approved=(tuple(x.shape)==(1152,36864) and 'A800' in props.name
              and (props.major,props.minor)==(8,0)
              and triton.__version__=='3.8.0'
              and torch.__version__.split('+')[0]=='2.8.0')
    return 'tiled' if approved else 'native'

def _native(x,dy,w,mean,rstd,bias=None):
    # Native forward uses (M,1) saved statistics for this 2D input.
    # PyTorch 2.8 CUDA backward needs bias metadata when dbias is requested.
    # Bias values do not enter the LayerNorm backward formulas.
    if bias is None:
        bias=torch.zeros_like(w)
    return torch.ops.aten.native_layer_norm_backward.default(
        dy,x,[x.shape[1]],mean.reshape(-1,1),rstd.reshape(-1,1),w,bias,[True,True,True])

def backward(x,dy,w,mean,rstd,backend='auto'):
    """Return new dx,dweight,dbias tensors. No forward, no autograd integration."""
    _validate(x,dy,w,mean,rstd)
    chosen=select_backend(x,backend)
    with torch.cuda.device(x.device):
        if chosen=='tiled':
            return backward_tiled(x,dy,w,mean,rstd,4096)
        return _native(x,dy,w,mean,rstd)

class PreparedBackward:
    """Bind input storage. Tiled outputs are overwritten by each run().

    Input values may change in place; shapes/storage must stay fixed. Do not
    share one instance across concurrent streams/in-flight invocations.
    The native fallback allocates outputs on each eager call.
    """
    def __init__(self,x,dy,w,mean,rstd,backend='auto'):
        _validate(x,dy,w,mean,rstd)
        self.backend=select_backend(x,backend)
        self.inputs=(x,dy,w,mean,rstd)
        self.device=x.device
        with torch.cuda.device(x.device):
            self.workspace=TiledWorkspace(x,dy,w,mean,rstd,4096) if self.backend=='tiled' else None
            self.bias=torch.zeros_like(w) if self.backend=='native' else None

    def run(self):
        with torch.cuda.device(self.device):
            return self.workspace.run() if self.workspace else _native(*self.inputs,bias=self.bias)
