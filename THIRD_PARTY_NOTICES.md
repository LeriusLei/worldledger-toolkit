# Third-party notices and related projects

`sbcc/imagined_rollouts.py::lambda_return` is a NumPy adaptation of the return recurrence in [DreamerV3](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py) at commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe`, copyright 2024 Danijar Hafner, MIT. The complete required notice is retained in [licenses/DreamerV3-MIT.txt](licenses/DreamerV3-MIT.txt). This module provides no RSSM or actor-critic training implementation.

Some generic adapter class names refer to [Pinductor](https://arxiv.org/abs/2605.13740), which motivated parts of the research interfaces. Its code, candidates, data and runtime are not bundled. NumPy and SciPy are installed as dependencies under their own licenses; optional Matplotlib, pytest and build dependencies retain their respective licenses. No third-party model weights or vendored benchmark repositories are redistributed.

The original research also explored battery simulation, NeuronBench, PatchWorld and other world-model systems. Those integrations are described only as excluded research branches in [PROJECT_MAP.md](docs/PROJECT_MAP.md); no results or licenses from those systems are implicitly transferred to this release.
