"""Prepared serial executor. Outputs alias reusable storage and are overwritten.

Keep input storage alive. Use one workspace per stream/in-flight invocation.
Shape/device/dtype validation and launch geometry are prepared once.
"""
import torch
import triton
from split_layernorm import dx_stats_kernel,dx_apply_kernel,partial_kernel,finish_kernel

class TiledWorkspace:
    def __init__(self,x,dy,w,mu,rs,cols=4096):
        assert x.is_cuda and x.ndim==2 and x.is_contiguous()
        m,n=x.shape
        assert m>0 and 1<=n<=65536 and cols in (2048,4096)
        assert dy.shape==x.shape and w.shape==(n,)
        assert x.dtype==dy.dtype==w.dtype==torch.bfloat16
        assert mu.numel()==rs.numel()==m and mu.dtype==rs.dtype==torch.float32
        assert all(v.device==x.device and v.is_contiguous() for v in (dy,w,mu,rs))
        t,g=triton.cdiv(n,cols),triton.cdiv(m,32)
        dx=torch.empty_like(x)
        stats=torch.empty((m,2*t),device=x.device,dtype=torch.float32)
        pw=torch.empty((g,n),device=x.device,dtype=torch.float32)
        pb=torch.empty_like(pw)
        dw,db=torch.empty_like(w),torch.empty_like(w)
        self.outputs=(dx,dw,db)
        self.buffers=(stats,pw,pb)
        self.launches=(
            (dx_stats_kernel,(m,t),(x,dy,w,mu,rs,stats,n,t,cols)),
            (dx_apply_kernel,(m,t),(x,dy,w,mu,rs,stats,dx,n,t,triton.next_power_of_2(t),cols)),
            (partial_kernel,(g,triton.cdiv(n,128)),(x,dy,mu,rs,pw,pb,m,n,32,128)),
            (finish_kernel,(triton.cdiv(n,128),),(pw,pb,dw,db,n,g,triton.next_power_of_2(g),128)))

    def run(self):
        for kernel,grid,params in self.launches:
            kernel[grid](*params,num_warps=4,num_stages=1)
        return self.outputs
