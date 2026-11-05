# CAEA 方法、复现检查与下一阶段实现

审查基线：main `917a2dd5f9ff813fb2969af978913687d2a945d8`，2026-11-05。
独立分支 `research/cpu-replay-evidence-grid`；此前 PR #1 尚未合并，本 PR 不依赖它。
SeetaCloud 上最新 DPO 数据尚未同步，不将仓库冻结模型实验冒充服务器的新训练结果。

## 方法与已有证据

论文 `paper/main.tex` 将偏好策略按报告类别 c 分解。给定原类别边缘 m 和审计目标 τ，
部分修复边缘 b=(1−λ)m+λτ，修复策略 π'(y)=b(c(y))π(y|c(y))。
这是 KL 最近投影，保留类内分布；实现位于 `code/repair.py`。
偏好目标代价为 β KL(b||m)。正目标质量不能赋给原策略没有支持的类，代码显式拒绝。
完全修复与同目标 audit-only 的报告风险相同是恒等关系，不构成两个独立成功实验。

理论贡献是：池化的一轮质量不识别反馈稳定性；立场条件化审查能揭示不同反馈机制。
收缩结论依赖明确更新模型，有限网格全局保证还依赖外供曲率；不推出人类获益。

| 层级 | 入口 | 现有证据与限制 |
| --- | --- | --- |
| 有限动作 DPO/反馈模拟 | experiment.py、stress.py、audit_stability.py | 10 个合成种子、63 响应、14 方法；并非 LLM 全量后训练 |
| 冻结神经候选头 | neural_probe.py、label_control.py、review_audit.py | 训练候选头、5 标签种子；置换后未建立稳健 DPO 增益 |
| 扩展冻结模型 | scale_study.py、external_study.py | Q4 三骨干；128 Bayesian 测试源情境、32 外部题；共 4672 已归档推理请求 |
| 审计器诊断 | evidence_audit_diagnostics.py | 旧数据重放；校准 LM 审计器仍逊于公开证据公式，合理更新明显受抑制 |
| 旧 GPU 复核 | gpu_study.py、gpu/run_autodl.sh | 仓库中的 BF16 路径；不可据此推断服务器 DPO 状态 |

code/ 存研究与绘图入口，results/ 存协议/预测/分析，paper/ 与 submission/ 存论文。
旧 GPU 脚本默认安装 cu118 与允许下载，不应覆盖现有 5090 的正常环境。

## 实际 CPU 检查

本机 Python 3.12，NumPy 2.3.5、SciPy 1.17.0、Matplotlib 3.10.8，与 core requirements 匹配。
从 GitHub 读取文件并对关键归档输入核对 Git blob SHA-1；未下载大模型、未运行新推理或训练。

| 检查 | 结果 |
| --- | --- |
| verify_results.py | 通过 |
| review_audit.py | 通过，归档神经重放最大绝对误差约 1.11e−16 |
| verify_external.py | 通过，32 源问题、448 既有请求及校准器迁移检查 |
| test_research_integrity.py | 6 项通过（CPU） |
| evidence_audit_diagnostics.py | 通过，校准目标 base regret 0.082877，公式目标 0.014656 |
| scale_study.py analyse → verify_scale.py | 从已核对的原生预测重建指标后通过，4224 既有请求、128 测试情境 |

访问限制：连接器未返回 2.8 MB 的 `results/scale/per_prompt_metrics.csv.gz` 内容，
故首次直接验证缺失该输入而失败。上述 scale 成功指从原生预测**重算后**的一致性检查，
不声称已逐字节验证 GitHub 中该指标归档。另一未返回文件 neural_label_control_metadata.json
未参与这些检查；它的独立认证未完成。不得把传输缺失描述成仓库损坏。

部分验证脚本会写固定 results 路径；因此本 PR 新增 `tools/check_reproducibility.py`：
复制 code/results 到临时目录，运行六个固定 CPU 检查，保存每项日志、输入哈希、依赖版本与退出码，
检查源目录哈希前后相同。报告路径必须在仓库外且不存在，不覆盖任何旧实验。
环境离线开关避免 Hugging Face 网络访问，但这不是操作系统网络沙箱。
它只检查归档一致性，不能认证当年的训练或推理实际发生，也不验证数学证明。

```bash
cd /root/autodl-tmp/你的仓库目录
# 使用已有兼容环境，不运行旧 GPU 安装器
python tools/check_reproducibility.py --out /root/autodl-tmp/caea-repro-唯一ID
```

需要完整 Git checkout 的 results（包括二进制归档）。非零退出时查看 report.json 与对应日志，
不要在真实旧结果目录运行 analyse 来“修复”检查。若要重算，另建副本并记录这是派生输出。

## 下一阶段已实现的数据协议

`protocols/build_evidence_grid.py` 使用标准库，不加载模型，默认生成 60 开发源情境与 300 确认源情境。
每个源情境 2 证据水平 × 2 保义模板 × 3 立场 = 12 请求，总计 4320 请求。
证据翻转交换 P(E|H) 与 P(E|¬H)，保持 prior 与事件 H 含义；Bayes 公式给出不同目标。
同一证据的 stance/改写不改目标；无立场是额外基线。
原/反事实似然交换视作同一源，避免其跨开发/确认集重复。
输出独立 requests.jsonl 与 labels.scoring-only.jsonl，后者不得输入模型；protocol.json 保存种子和哈希。
重复运行到已有目录会失败。生成是设计提案，不等于外部预注册或已运行实验。

```bash
python protocols/build_evidence_grid.py --out /root/autodl-tmp/caea-grid-唯一ID
python -m unittest discover -s tests -v
```

这两种模板同时出现在两个 split，仅检验新证据三元组下的模板鲁棒性，**不是模板族外迁移**。
还需与服务器训练/历史测试证据做重合检查；没有拿到服务器数据前，不能宣称独立确认集未受污染。
协议一旦被用来选择方法，其确认集应降级为探索数据。

## 接入现有服务器实验的具体约定

1. 先登记真实模型/adapter 目录、三个实际种子、训练日志、训练代码版本、数据哈希、chat template 和依赖版本。
   未知来源标 unknown，不使用本 PR SHA 追认旧训练；原文件原位保留。
2. 推理适配器接收 requests 中 prompt/id，显式 local_files_only=True，固定生成长度/解码，
   顺序加载已有 base、Stance-DPO、Evidence-DPO，禁止自动回退下载。先 CPU 数据检查，再另行 GPU 推理小试；不重训。
3. 每请求保存 id、method、seed、raw_response、parsed_probability、status、model/adapter 标识、
   prompt/template 哈希、token 数、秒数与峰值显存。解析失败/拒答仍保留，不能过滤后才评分。
4. 评估器只按 id 联接 scoring-only 标签，拒绝缺失/重复/额外 id。组间须同源覆盖。
   固定证据比较 high-low，固定立场比较 counterfactual-original；分别报告概率平方误差、
   立场差、翻转方向和更新误差。自报概率的平方误差不能混称离散候选报告分布的期望风险。
5. 加入 evidence-first prompting、开发集简单收缩、公共证据求解器；若纳入 CAEA，必须加入同审计器 audit-only。
   按实际种子分别报告，再按源情境做配对 cluster bootstrap；所有干预与方法一起重采样。
   区间只条件于现有三种子，另外报告种子变异；冻结主要比较、多重校正与更新非劣界值。

本 PR 不猜测尚未同步的 adapter 格式或 TRL 训练入口，故不提供伪可运行的模型执行命令。
完成资产核对后，按上述约定实现适配器/评分器；新训练须由评估结果与消融问题驱动。
