# WorldLedger Toolkit

![WorldLedger Toolkit](docs/assets/banner.svg)

**可执行世界模型的离线评测与研究审计工具箱。**

[English](README.md) · [快速上手](docs/QUICKSTART.md) · [架构与限制](docs/ARCHITECTURE.md) · [证据边界](docs/EVIDENCE.md)

这个仓库整理了原项目中可以独立复用的工程：有限环境、精确遗憾评估、候选模型委员会、反事实评估信息隔离，以及保留原始数据和费用不确定性的实验账本。核心问题是：**模型之间的分歧，是否真的会改变决策？**

开源定位是「模块化研究工具箱 + 合成实验套件」。这里提供 Python 组件和有限世界模型规格，没有训练好的小模型权重，也没有可以支撑通用能力排名的公开排行榜。原研究中未通过或尚无定论的分支，保持原来的证据结论。

## 安装和运行

CI 目标为 Python 3.10–3.12。示例只需 CPU，运行不需要密钥、付费 API、模型下载或 GPU；首次安装依赖通常需要访问 Python 包索引。

```bash
git clone https://github.com/LeriusLei/worldledger-toolkit.git
cd worldledger-toolkit
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev,plots]"
python -m pytest -q
python examples/finite_worlds.py
python examples/audit_ledger.py
python experiments/run_sprinkler.py --seeds 400 --output-dir artifacts/sprinkler
python experiments/run_hybrid_counterexample.py --output-dir artifacts/hybrid_counterexample
python scripts/privacy_check.py
```

实验输出 JSON、CSV 和说明文档；安装 Matplotlib 后，Sprinkler 实验还会输出 PNG 图。

## 可复用的部分

| 组件 | 用途 |
|---|---|
| `worldledger/benchmark` | Grid、Inventory、Queue 有限任务与公开历史生成 |
| `worldledger/evaluator` | 一致性、行为/决策等价类、覆盖率、候选遗漏与部署遗憾 |
| `worldledger/world_model` | 有界 DSL、确定/随机核穷举、Machine Maintenance、精确共享策略 |
| `sbcc` | 候选委员会、决策证书、episode 级统计、候选混配诊断 |
| `sbcc/information_firewall.py` | 在线轨迹与评估真值隔离、随机状态恢复、episode 结束后解封 |
| `worldledger/sandbox` | 受限 AST 源语言验证与解释执行 |
| `research_audit` | 协议身份、原始数据先落盘、哈希链、未知费用保留与重放 |
| `sbcc/imagined_rollouts.py` | NumPy lambda return、模型内想象轨迹、离散 CEM |

![架构与信息边界](docs/assets/architecture.svg)

## 示例能说明什么

Sprinkler 是人为构造的有限环境，用于展示「了解模型」和「知道该采取什么行动」可以不同。下面是仓库脚本的 400-seed、无噪声示例结果，不能外推为真实任务或学习模型的胜率。数据及配置见 [summary](docs/demo_results/sprinkler_summary.json)。

![合成实验结果](docs/assets/sprinkler_results.png)

候选组件混配实验给出一个精确反例：两个完整模型都支持安全动作，但跨模型混配转移与奖励后，会出现原模型均不支持的动作排序。它说明这种现象可能发生，不代表外部系统的普遍失败率。

审计账本负责记录执行事实，不负责判断科学假设是否成立。账本是单写者设计；本地哈希链不是数字签名。评估信息隔离和受限解释器属于研究控制，不是操作系统级安全沙箱。Conformal 组件需要适当的 episode 划分和交换性等条件，不能凭合成测试宣称真实部署覆盖率保证。

## 为什么这样开源

原项目包含多轮尝试、不同接口的第三方集成和多个停止结论。最有复用价值的是已经实现的评测与审计组件。把它们独立打包，可以方便他人搭实验、复核候选身份、保留失败分母和排查信息泄漏。

详细的分支去留、证据边界与文件来源见 [项目地图](docs/PROJECT_MAP.md)、[证据说明](docs/EVIDENCE.md) 和 [来源清单](docs/SOURCE_MANIFEST.json)。私人对话、API 凭据、原始供应商响应、论文草稿、运行缓存和旧 Git 历史不进入这个公开仓库。

欢迎提交问题和 PR，参见 [贡献指南](CONTRIBUTING.md)。许可证为 [MIT](LICENSE)，DreamerV3 派生部分保留原许可证。引用信息见 [CITATION.cff](CITATION.cff)。
