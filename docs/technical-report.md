# 面向宽特征维度的 Triton LayerNorm backward 优化

## 问题与目标

以 triton-lang/triton #8908 的慢编译内核为研究起点，在 A800 上复现、定位并改写 LayerNorm backward。目标同时覆盖编译时间、执行时间和正确性；成果是独立内核原型，不是编译器补丁。

环境：A800-SXM4-80GB，Python 3.10.19，PyTorch 2.8.0+cu128，Triton 3.4.0/3.8.0。运行性能比较采用 3.8.0；目标输入为连续 BF16 1152×36864，保存的 mean/rstd 为 FP32。

## 定位过程

1. 原始 rows4 在 Triton 3.4 的冷编译约 218 秒，其中 LLVM O3 约 185 秒。rows1 编译显著缩短，提示展开规模与优化复杂度有关。
2. 升级到 3.8 后 LLIR 阶段降至约 3.43 秒，但 cubin 汇编约 89 秒。升级收益属于版本变化，不能算作本项目优化。
3. 同一旧 PTX 在 ptxas 12.8/12.9 分别约 26.72/28.14 秒；新 PTX 在 12.9 约 92.27 秒。不指定与指定 regAllocOptLevel=2 得到相同 cubin 哈希，未见移除该选项的收益。定位范围收窄至代码生成形态及其与汇编器的交互，未定位编译器内部具体 pass。
4. rows1 运行约 4.39 ms，而 PyTorch 约 0.41 ms，说明参数调整远不足以形成有竞争力的实现。
5. 首轮分离 dx 和参数梯度后约 2.43 ms；dx 独占约 2.30 ms，16 warp spill 指标仍为 486。不能只增加 warp，必须控制单块活跃数据量。

## 算法改写

定义 z=(x-mean)*rstd，v=dy*weight。输入梯度为 rstd*(v-mean(v)-z*mean(v*z))；参数梯度沿行分别归约 dy*z 与 dy。

最终四个阶段：

1. dx_stats：每行按 2048/4096 列分块，输出 sum(v) 和 sum(v*z)。
2. dx_apply：读取少量块间部分和得到整行均值，重新读取本列块，计算 dx。
3. parameter_partials：32 行×128 列分块，计算 dweight/dbias 部分和。
4. parameter_finish：合并行组部分和，输出 BF16 参数梯度。

设计取舍是更多 kernel 启动和重复读取，换取更小的寄存器工作集、无 spill 与更多并行任务。四内核本身没有原子累加；不能由此声称跨所有平台逐位确定。

## 结果

### 同口径冷编译

三个独立进程/缓存、轮换执行顺序，中位数：原始 rows4 102.7906 s，rows1 6.8759 s，tiled4096 四内核合计 1.5080 s。相对 rows4 约 68.17×。包含首次编译器初始化，不包含环境准备、GPU 执行和原始实现的最终 PyTorch eager 归约，OS 缓存未清除。这是内核编译时间，不是首次训练步延迟。

### 性能与资源

4096 列版本四内核 spill 均为 0，寄存器分别为 53/53/64/47。先前 dx spill=486 与新版 0 的对照支持减少寄存器压力的解释，但不构成“所有收益只来自 spill”的单变量证明。

目标形状底层实现三个独立进程复测：eager 耗时降低约 7.8%，Graph 降低约 8.6%–8.8%。独立包最终 API 验收为：

| 模式 | native | auto | prepared |
|---|---:|---:|---:|
| eager，μs | 408.71 | 378.74 | 377.14 |
| Graph，μs | 405.95 | 371.69 | 372.05 |

auto 耗时减少约 7.33%/8.44%。此处全含 backward 所需阶段，不包含前向或编译；Graph 复用捕获存储，输入在测试中重复驻留。CUDA events 的 eager 测量可能包含 CPU 提交间隙。

### 适用范围与负结果

18 组形状单轮扫描，固定 4096 列配置：eager 5 组快于 native，Graph 15 组快于 native，不能推广为所有 LayerNorm 场景更快。33×4096 最终 auto 为 49.31 μs，而直接 native 为 16.90 μs；保守回退仍有封装开销。

工作区将提交时间约 109 μs 降至 58 μs，移出了分配、检查及配置准备。Graph 下工作区与分配型版本接近，符合相同 GPU 工作、不同 host 开销的解释。没有把全部差异归为纯内存分配成本。

## 验证与工程边界

已做数学参考、native/autograd 对照、多随机种子、常量/小方差、行尾块、非对齐列、变更输入后的缓冲区复用、Graph 输出与错误参数测试。BF16 最终梯度多数使用 atol=0.03125、rtol=0.01；有限值、最大绝对误差与相对 L2 记录在日志。通过容差不代表数学精确或逐位一致。

auto 只放行 A800 sm80、torch2.8.0、triton3.8.0、1152×36864；其他合法输入回退 native。公开接口为 backward-only，不支持高阶 autograd。工作区绑定存储且会覆盖输出，不支持同一实例并发使用。

独立包通过哈希、正确性与基准验收，见 ACCEPTANCE_v1.md。未验证 RTX3090/4090、其他 dtype/布局或端到端训练，尚未发布上游 PR。

## 上游状态

本轮 GitHub 查询返回 #8908 为 Open，标题为 Extremely slow compile time for triton kernel (151 seconds)，评论计数 3。已读取三条评论，讨论了大 block size 与后续调查；未据此断言无人解决，也未完成关联 PR 的全面搜索。后续反馈宜定位为诊断数据和 kernel-level workaround，是否适合作为编译器回归测试需维护者判断。

参考：https://github.com/triton-lang/triton/issues/8908
