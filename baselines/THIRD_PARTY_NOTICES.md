# Baseline algorithm sources

GCBC and goal-conditioned IQL follow the published objectives and the reference
loss definitions in [OGBench](https://github.com/seohongpark/ogbench), inspected
at commit `1d4140997f60c52c6fb0702ec100dc988b18c548` (`impls/agents/gcbc.py` and
`impls/agents/gciql.py`). The standalone data contract, categorical encodings,
checkpoint/run interfaces and generic transition model are implemented here.
There is no runtime dependency on OGBench or the unpublished CoPuzzleLab workspace.
The reference notice is retained below. This notice does not choose a license for
the project's own code; that remains a release decision.

Algorithm references:

- Kostrikov, Nair and Levine, [Offline Reinforcement Learning with Implicit
  Q-Learning](https://arxiv.org/abs/2110.06169), ICLR 2022.
- Park, Frans, Eysenbach and Levine, [OGBench: Benchmarking Offline
  Goal-Conditioned RL](https://arxiv.org/abs/2410.20092), ICLR 2025.

## OGBench MIT notice

The MIT License (MIT)

Copyright (c) 2024 OGBench Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
