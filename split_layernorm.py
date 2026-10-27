import torch
import triton
import triton.language as tl

@triton.jit
def partial_kernel(X, DY, Mean, Rstd, PW, PB,
                   M: tl.constexpr, N: tl.constexpr,
                   R: tl.constexpr, C: tl.constexpr):
    group = tl.program_id(0)
    rows = group*R + tl.arange(0, R)
    cols = tl.program_id(1)*C + tl.arange(0, C)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(X + rows[:, None]*N + cols[None, :], valid, other=0).to(tl.float32)
    dy = tl.load(DY + rows[:, None]*N + cols[None, :], valid, other=0).to(tl.float32)
    mu = tl.load(Mean + rows, rows < M, other=0)
    rs = tl.load(Rstd + rows, rows < M, other=0)
    norm = (x-mu[:, None])*rs[:, None]
    dw = tl.sum(dy*norm, 0)
    db = tl.sum(dy, 0)
    tl.store(PW + group*N + cols, dw, cols < N)
    tl.store(PB + group*N + cols, db, cols < N)


@triton.jit
def finish_kernel(PW, PB, DW, DB, N: tl.constexpr, G: tl.constexpr,
                  BG: tl.constexpr, C: tl.constexpr):
    groups = tl.arange(0, BG)
    cols = tl.program_id(0)*C + tl.arange(0, C)
    mask = (groups[:, None] < G) & (cols[None, :] < N)
    dw = tl.load(PW + groups[:, None]*N + cols[None, :], mask, other=0)
    db = tl.load(PB + groups[:, None]*N + cols[None, :], mask, other=0)
    tl.store(DW + cols, tl.sum(dw, 0), cols < N)
    tl.store(DB + cols, tl.sum(db, 0), cols < N)


@triton.jit
def dx_stats_kernel(X, DY, W, Mean, Rstd, Stats,
                    N: tl.constexpr, T: tl.constexpr, C: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile*C + tl.arange(0,C)
    valid = col < N
    x = tl.load(X+row*N+col,valid,other=0).to(tl.float32)
    dy = tl.load(DY+row*N+col,valid,other=0).to(tl.float32)
    w = tl.load(W+col,valid,other=0).to(tl.float32)
    mu,rs = tl.load(Mean+row),tl.load(Rstd+row)
    norm = tl.where(valid,(x-mu)*rs,0.)
    weighted = dy*w
    tl.store(Stats+row*(2*T)+tile,tl.sum(weighted,0))
    tl.store(Stats+row*(2*T)+T+tile,tl.sum(weighted*norm,0))


@triton.jit
def dx_apply_kernel(X, DY, W, Mean, Rstd, Stats, DX,
                    N: tl.constexpr, T: tl.constexpr, BT: tl.constexpr,
                    C: tl.constexpr):
    row,tile = tl.program_id(0),tl.program_id(1)
    part = tl.arange(0,BT)
    a = tl.sum(tl.load(Stats+row*(2*T)+part,part<T,other=0),0)/N
    b = tl.sum(tl.load(Stats+row*(2*T)+T+part,part<T,other=0),0)/N
    col = tile*C+tl.arange(0,C)
    valid = col<N
    x = tl.load(X+row*N+col,valid,other=0).to(tl.float32)
    dy = tl.load(DY+row*N+col,valid,other=0).to(tl.float32)
    w = tl.load(W+col,valid,other=0).to(tl.float32)
    mu,rs = tl.load(Mean+row),tl.load(Rstd+row)
    norm = (x-mu)*rs
    tl.store(DX+row*N+col,(dy*w-a-norm*b)*rs,valid)


def backward_tiled(x, dy, weight, mean, rstd, tile_cols=2048):
    assert x.ndim == 2 and x.is_cuda and x.is_contiguous()
    assert dy.shape == x.shape and dy.is_contiguous()
    assert x.dtype == dy.dtype == weight.dtype == torch.bfloat16
    m,n = x.shape
    assert m>0 and 1<=n<=65536 and weight.numel()==n and weight.is_contiguous()
    assert all(v.device==x.device for v in (dy,weight,mean,rstd))
    assert mean.numel()==rstd.numel()==m
    assert mean.dtype==rstd.dtype==torch.float32
    assert mean.is_contiguous() and rstd.is_contiguous()
    assert tile_cols in (1024,2048,4096)
    tiles,groups = triton.cdiv(n,tile_cols),triton.cdiv(m,32)
    dx = torch.empty_like(x)
    stats = torch.empty((m,2*tiles),device=x.device,dtype=torch.float32)
    pw = torch.empty((groups,n),device=x.device,dtype=torch.float32)
    pb = torch.empty_like(pw)
    dw,db = torch.empty_like(weight),torch.empty_like(weight)
    dx_stats_kernel[(m,tiles)](x,dy,weight,mean,rstd,stats,n,tiles,tile_cols,
                              num_warps=4,num_stages=1)
    dx_apply_kernel[(m,tiles)](x,dy,weight,mean,rstd,stats,dx,n,tiles,
                              triton.next_power_of_2(tiles),tile_cols,num_warps=4,num_stages=1)
    partial_kernel[(groups,triton.cdiv(n,128))](x,dy,mean,rstd,pw,pb,m,n,32,128,
                                              num_warps=4,num_stages=1)
    finish_kernel[(triton.cdiv(n,128),)](pw,pb,dw,db,n,groups,
                    triton.next_power_of_2(groups),128,num_warps=4,num_stages=1)
    return dx,dw,db
