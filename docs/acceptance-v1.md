# v1 独立包验收记录

状态：用户远程执行的文件完整性、正确性及性能验收全部通过。此记录位于交付包外，原 ZIP 和 MANIFEST 保持不变。

- ZIP SHA256：7330ff4279c5678cb83db1a6c63ca0d43a49c466fd0dbfd8f9af20e557ea57f7
- 远程目录：<independent acceptance directory>/layernorm_backward_v1
- 文件检查：MANIFEST PASS，10 个文件。
- 正确性：api_validation_1793090746570548277.json，ALL PASS，退出码 0。
- 性能：api_benchmark_1793091282136194912.json，78 项检查通过，ALL PASS，退出码 0。

环境沿用 A800、PyTorch 2.8.0+cu128、Triton 3.8.0。数据来自用户提交的远程输出，本地未运行 GPU。

## 目标形状 1152×36864，BF16，单位 μs

| 模式 | PyTorch native | API auto | Prepared auto |
|---|---:|---:|---:|
| eager | 408.7125 | 378.7435 | 377.1392 |
| CUDA Graph | 405.9477 | 371.6949 | 372.0533 |

auto 相对 native eager 耗时减少约 7.33%，Graph 约 8.44%。结果与打包前一致。仅限 backward，同一驻留输入重复测量，不包含前向或冷编译。

## 保留限制

33×4096 eager native/auto/prepared 为 16.8960/49.3056/27.0677 μs；64×12288 为 22.3232/48.4864/26.6581 μs。小形状回退开销仍存在，不宣称无退化。Graph prepared 回退接近原生。

本次完成的是实验原型独立交付验收；未完成公开发布、上游贡献、其他显卡或完整训练验证。包内 README/RESULTS 中“独立包验收待完成”为打包时状态，以本记录更新，避免改动已验证包的哈希。
