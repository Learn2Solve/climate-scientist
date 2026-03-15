# Climate Scientist Agent 规范（SPEC）

本规范将“Fully workable climate scientist agent”的工作流与论文写作落地为可执行工程方案，面向当前仓库实现。

## 1. 目标与贡献

### 1.1 目标
构建一个可运行、可复现、可审计的 CLI Climate Scientist Agent：
- 能拆解任务、跑代码、拉数据、做实验、出图、写结论。
- 对每一步有验证标准与可追溯证据链。
- 对 hurricane track 预测任务形成“复现 + 自动评测 + 小幅可复现提升”的闭环。

### 1.2 论文贡献（3 点）
1. **可运行的 CLI Climate Scientist Agent**：任务规划、工具执行、自检迭代、步数/预算限制、全量日志落盘。
2. **可复现实验工作台**：数据/环境/评测标准化，确保结果可跑可对齐。
3. **任务导向评测套件**：硬指标（track error、wind error）自动打分 + LLM 输出有效率。

## 2. 设计原则
- **可复现优先**：所有运行产物与配置落盘。
- **可审计优先**：任何结论可追溯到数据版本、环境、代码与输出。
- **最小可用**：先保证 hurricane track 预测闭环可跑，再扩展任务面。
- **失败可诊断**：对失败原因分类并写入报告。

## 3. Agent 架构（最小 5 模块）

### 3.1 Planner
- 输入：自然语言目标、paper/repo、数据集指针。
- 输出：研究计划 DAG（步骤、依赖、产物、验收标准）。

### 3.2 Tools
必须覆盖科研闭环：
- 代码执行（Python / shell / git）
- 数据访问（HTTP/OPeNDAP/本地缓存）
- 绘图/报告（matplotlib / markdown）
- 算力接口（本地 GPU/CPU；可扩展到 SLURM）
- 文献检索（可选：外部搜索与 PDF 拉取）

### 3.3 Memory（科研资产库）
- 运行元数据、数据版本、环境哈希、指标、结论、失败原因、复现脚本。
- 最重要：**审计链**完整。

### 3.4 Verifier
- 每步有验收条件（schema 校验、指标检查、数据泄漏检查等）。
- 不满足则回滚/修正计划。

### 3.5 Safety
- step limit、timeout、预算上限、写权限隔离、循环检测。

## 4. CLI Agent 最小实现

### 4.1 入口
- 新增：`src/agent_cli.py`
- 目标：以 workflow 为单位执行研究计划

示例：
```bash
uv run --no-project python src/agent_cli.py \
  --goal "Reproduce HURDAT2 24/48/72h track skill baseline" \
  --workflow reproduce \
  --data hurdat2_llm_toy \
  --out runs
```

### 4.2 模块建议
```
src/agent/
  planner.py
  runner.py
  tools.py
  memory.py
  verifier.py
  safety.py
src/workflows/
  reproduce.yaml
  improve.yaml
  new_research.yaml
```

## 5. 运行产物规范（审计链核心）

每次运行生成：
```
runs/<run_id>/
  plan.json
  tool_calls.jsonl
  env.lock
  data_manifest.json
  metrics.json
  report.md
  artifacts/
```

字段说明：
- `plan.json`: 研究计划 DAG + 验收标准
- `tool_calls.jsonl`: 工具调用日志（时间、命令、输出摘要）
- `env.lock`: python 依赖与环境信息（pip freeze / uv lock）
- `data_manifest.json`: 数据来源、版本、hash
- `metrics.json`: 指标结果
- `report.md`: 结论 + 失败分析 + 复现说明

## 6. 三条工作流（科研闭环）

### 6.1 Workflow 1: Reproduce
**输入**：paper/repo + 数据集
**输出**：可跑复现脚本 + 指标对齐 + 失败分析

步骤模板：
1. 解析 paper → 抽取数据/特征/模型/指标
2. 拉数据并校验版本 → `data_manifest.json`
3. 建环境 → `env.lock`
4. 跑 baseline → 生成指标
5. 误差来源定位 → `report.md`

### 6.2 Workflow 2: Improve
**输入**：复现 baseline
**输出**：可复现提升 + ablation

步骤模板：
1. 生成 3–5 个改进候选
2. 自动 ablation matrix
3. 输出：提升幅度 + 方差 + 代价

### 6.3 Workflow 3: New Research
**输入**：任务目标
**输出**：新方法 + 实验 + 结论

步骤模板：
1. 定义研究空间（变量/模型/损失族）
2. 受控探索
3. 统计显著性 + 结论

## 7. Hurricane Track 任务闭环

### 7.1 数据
- HURDAT2 / IBTrACS best track 作为真值
- 已有数据准备脚本：`src/data_prep.py`

### 7.2 指标
- Track error: great-circle distance
- Wind MAE
- valid_json_rate
- Rapid Intensification (RI) @ 24h: 定义 ΔV ≥ 30 kt/24h，报告 precision/recall/F1 + RI 子集 wind MAE（`src/ri_metrics.py`）

（目前 `src/evaluate.py` 和 `src/evaluate_jsonl.py` 已覆盖核心指标）

### 7.3 Baselines
- Persistence / kinematic baseline（`src/baselines.py`）
- LLM baseline（`src/run_forecaster_jsonl.py`）
- TTM baseline（`src/ttm_baseline*.py`）

## 8. 论文实验设计（最少 4 组）

1. **Reproduction Success Rate**：N 篇论文/项目复现成功率 + 失败原因分类
2. **Metric Fidelity**：与官方指标对齐误差
3. **Improvement Study**：自动 ablation 的可复现提升
4. **Auditability**：展示证据链 case study

## 9. MVP 落地顺序

1. 产物落盘与审计链（runs/ 结构）
2. 仅支持 hurricane track eval
3. 加入 reproduce workflow
4. 扩展到 improve / new research

## 10. 非目标（当前阶段）
- 不追求多任务泛化
- 不做复杂 UI
- 不做自动论文写作全流程

## 11. 风险与对策
- **指标实现偏差**：用官方定义交叉校验
- **数据泄漏**：数据切分与时间窗口检查
- **LLM 输出不稳定**：严格 schema + valid_json_rate
- **算力成本**：限制样本数与默认预算

---

本 SPEC 与当前仓库的脚手架直接对齐，旨在在最小工程代价下完成可运行、可审计的 climate scientist agent MVP。
