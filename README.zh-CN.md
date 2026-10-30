<div align="center">

![WorldLedger — 看清决策背后的候选世界](docs/assets/banner.png)

**评估候选世界，把决策与证据放在一起。**

[![CI](https://github.com/LeriusLei/worldledger-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/LeriusLei/worldledger-toolkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%E2%80%93%203.12-607D8B?logo=python&logoColor=white)](pyproject.toml)
[![MIT](https://img.shields.io/badge/License-MIT-6E8B5B)](LICENSE)

[English](README.md) · [文档入口](docs/README.md) · [快速上手](#快速上手) · [案例图解](docs/EXAMPLES.md) · [工程模块](#三个可复用的包)

</div>

WorldLedger 是研究**可执行世界模型如何影响决策**的 Python 工具箱。它把有限环境、精确遗憾评估、完整候选模型委员会和实验审计账本，整理成三个可以独立复用的包。

- **看清候选集。** 同时检查行为覆盖、决策覆盖，以及遗漏候选世界引入的风险。
- **隔离在线行动与事后评估。** 在线轨迹只接收实际执行分支的反馈，评估真值在 episode 封存后解锁。
- **保留可追溯证据。** 对输入文件计算哈希，先保存原始记录再计算指标，重放事件，并保留未核实的费用。

仓库中的任务是公开透明的合成控制，库仍处于实验阶段；[证据与适用边界](docs/EVIDENCE.md) 与实现一起提供。

## 快速上手

Python 3.10–3.12，CPU 即可；示例运行不需要模型下载或 API 密钥。

```bash
git clone https://github.com/LeriusLei/worldledger-toolkit.git
cd worldledger-toolkit
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev,plots]"
python examples/finite_worlds.py
```

[安装细节、审计账本用法与 wheel 说明 →](docs/QUICKSTART.md)

## 覆盖率相同，遗漏风险可以不同

下面从同一个注册 Grid 实例中选出两组候选世界，直接比较覆盖率和遗漏遗憾。代码完整可运行：

```python
from worldledger.benchmark.generators.grid import generate_family_instance
from worldledger.benchmark.history_generator import generate_histories
from worldledger.evaluator.regret import evaluate_pool

world = generate_family_instance(4)
history = generate_histories(world, world.models[0])[0]
pools = {"A": world.models[:2], "B": world.models[::2]}

for name, models in pools.items():
    report = evaluate_pool(world, history, [m.model_id for m in models], horizon=1)
    print(name, "coverage:", report["hcr"], "omission regret:", report["delta_omit"])
```

```text
A coverage: 0.5 omission regret: 0.0
B coverage: 0.5 omission regret: 3.0
```

两组都与公开历史一致。`delta_omit` 表示：相对于完整一致支持集，保留候选集的策略增加了多少最坏情形遗憾。这是人为选取候选集的一步有限环境示例，解释限于给定模型族、奖励与 horizon。[更多案例与指标说明 →](docs/EXAMPLES.md)

## 三个可复用的包

| 包 | 可以用来做什么 | 入口 |
|---|---|---|
| **`worldledger`** | 构造有限任务，计算精确支持集与遗憾，评估共享观测条件策略，解释受限源语言 | [任务](worldledger/benchmark/) · [评估器](worldledger/evaluator/) · [有限核](worldledger/world_model/) |
| **`sbcc`** | 使用完整候选委员会、动作集、episode 级统计、反事实轨迹控制与模型内想象 | [决策组件](sbcc/) · [信息边界](sbcc/information_firewall.py) |
| **`research_audit`** | 固定输入身份，原始数据先落盘，连接事件哈希，核对费用并重放运行 | [账本 API](research_audit/harness.py) · [示例](examples/audit_ledger.py) |

### 组件如何连接

![公开历史、候选决策、封存评估和审计账本的系统关系](docs/assets/architecture.png)

在线策略接收公开观测；可信评估器记录反事实指标，评估结果在 episode 封存后读取。账本记录整个过程的执行事实。[接口与运行限制 →](docs/ARCHITECTURE.md)

## 四个可以立即运行的控制

| 想检查的问题 | 在仓库根目录运行 | 观察什么 |
|---|---|---|
| 事实历史如何收缩模型支持集？ | `python examples/finite_worlds.py` | Grid、Inventory、Queue 的支持集 4 → 2 → 1 |
| 哪些信息会影响决策？ | `python experiments/run_sprinkler.py --seeds 400 --output-dir artifacts/sprinkler` | 效用、模型熵和决策等价类 |
| 跨候选混配组件会发生什么？ | `python experiments/run_hybrid_counterexample.py --output-dir artifacts/hybrid` | 四组转移/奖励配对 |
| 记录能保留不确定性并发现修改吗？ | `python examples/audit_ledger.py` | 实际文件重哈希、原始记录与未知费用 |

![Sprinkler 实际合成结果：效用与剩余决策歧义](docs/assets/sprinkler_results.png)

图片使用仓库已保存的 [400-seed 数据](docs/demo_results/sprinkler_summary.json)：均衡分配、无噪声探测、单次探测成本 0.05。它展示构造环境中的机制。[查看完整案例图解与适用范围 →](docs/EXAMPLES.md)

## 复现与验证

```bash
python -m pytest -q
python scripts/privacy_check.py
python scripts/check_docs.py
python scripts/create_figures.py
```

[CI](https://github.com/LeriusLei/worldledger-toolkit/actions/workflows/ci.yml) 覆盖 Linux/Windows 与 Python 3.10/3.11/3.12，检查测试、示例、隐私、文档和构建。另附 [源文件哈希](docs/SOURCE_MANIFEST.json)、[数据](docs/demo_results/)、[验证记录](docs/VALIDATION.md) 和 [制图源码](scripts/create_visuals.py)。

## 继续阅读

- [文档地图](docs/README.md)：安装、案例和模块入口。
- [架构说明](docs/ARCHITECTURE.md)：策略范围、评估信息隔离、AST 语言、账本契约。
- [证据边界](docs/EVIDENCE.md)：合成控制支持的结论，以及仍未建立的研究主张。
- [项目地图](docs/PROJECT_MAP.md)：各研究分支的去留与来源说明。

欢迎通过 [Issues](https://github.com/LeriusLei/worldledger-toolkit/issues) 和 PR 参与，参见 [贡献指南](CONTRIBUTING.md)、[社区规范](CODE_OF_CONDUCT.md) 与 [安全说明](SECURITY.md)。引用信息见 [CITATION.cff](CITATION.cff)。

许可证为 [MIT](LICENSE)，派生代码和字体保留 [上游声明](THIRD_PARTY_NOTICES.md)。公开目录不包含凭据、私人对话、原始供应商响应、模型缓存和原私有 Git 历史。
