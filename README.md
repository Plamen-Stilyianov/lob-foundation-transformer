# lob-foundation-transformer

A PyTorch **Limit Order Book (LOB) Time-Series Foundation Model** pre-trained from scratch using mixed precision (`bfloat16`), causal self-attention (`SDPA`), and patch-level microstructure tokenization—complete with **scaling-law ablations**, **cross-regime stress testing**, and **transaction-cost-aware economic evaluation**.

Designed around quantitative research workflows for electronic market making, execution, and cross-asset alpha modeling.

## Architecture & Training Stack
- **From-Scratch Causal Transformer (`src/model.py`):** Patch-tokenized L2 order-book encoder (`RMSNorm`, `SwiGLU`, causal `F.scaled_dot_product_attention` with zero future leakage) trained jointly on **next-patch microstructure reconstruction** and **multi-horizon forward mid-price return prediction (1, 5, 15 ticks)**.
- **Mixed-Precision Pre-Training (`src/pretrain.py`):** Native PyTorch `torch.amp.autocast(dtype=torch.bfloat16)` on CUDA (`NVIDIA RTX 3080 Ti`), `AdamW` with linear warmup + cosine decay, gradient clipping (`max_norm=1.0`), and checkpoint persistence.
- **Multi-Regime L2 Microstructure Simulator (`data/lob_simulator.py`):** Simulates 10-level bid/ask order-book depth, queue skew, order-flow imbalance (OFI), and jump dynamics across **Calm**, **Volatile**, and **Liquidity-Shock** regimes.
- **Economic & Latency Evaluation (`src/evaluate_economics.py`):** Bridges pre-training loss to out-of-sample **Information Coefficient (IC)**, **net PnL after half-spread crossing costs (bps)**, **annualized net Sharpe**, and **CUDA p99 inference latency (ms)**.

## Scaling Laws & Cross-Regime Results (`NVIDIA RTX 3080 Ti`, `bfloat16`)

Both models were pre-trained from scratch exclusively on **Calm** regime L2 order-book data (`30,000` ticks) and evaluated out-of-sample across unseen **Calm**, **Volatile**, and **Liquidity-Shock** regimes after deducting half-spread crossing costs:

| Model Scale | Parameters | Pre-Train Loss | p99 Latency (CUDA) | Eval Regime | Out-of-Sample IC | Net PnL (after spread) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **Tiny** (`d=96, L=2`) | 252,095 | 4.3940 | **2.229 ms** | Calm | +0.3375 | +7,122.64 bps |
| **Tiny** (`d=96, L=2`) | 252,095 | 4.3940 | 2.229 ms | Volatile | +0.1127 | +5,945.94 bps |
| **Tiny** (`d=96, L=2`) | 252,095 | 4.3940 | 2.229 ms | Liquidity Shock | +0.0623 | +8,183.17 bps |
| **Base** (`d=192, L=4`) | **1,831,967** | **4.2525** | 3.877 ms | **Calm** | **+0.3522** | **+8,699.72 bps** |
| **Base** (`d=192, L=4`) | **1,831,967** | **4.2525** | 3.877 ms | **Volatile** | **+0.1221** | **+7,362.15 bps** |
| **Base** (`d=192, L=4`) | **1,831,967** | **4.2525** | 3.877 ms | **Liquidity Shock** | **+0.0698** | **+11,574.98 bps** |

### Key Takeaways
1. **Monotonic Scaling Across Regimes:** Scaling parameters 7.3x (`252K` -> `1.83M`) lowers pre-training loss (`4.3940` -> `4.2525`) and improves out-of-sample IC and post-cost PnL across all three market regimes—including severe liquidity shocks never seen during pre-training.
2. **Latency-Quality Frontier:** The 7.3x larger model adds only **+1.65 ms** at p99 (`2.229 ms` -> `3.877 ms`) while delivering **+22.1%** higher net PnL in Calm regimes and **+41.4%** higher net PnL during Liquidity Shocks.

## Project Structure

```text
lob-foundation-transformer/
├── data/
│   ├── __init__.py
│   └── lob_simulator.py        # Multi-regime 10-level L2 Limit Order Book & OFI generator
├── src/
│   ├── __init__.py
│   ├── model.py                # From-scratch Causal LOB Transformer (RMSNorm, SwiGLU, SDPA)
│   ├── pretrain.py             # bfloat16 mixed-precision pre-training loop & cosine warmup
│   └── evaluate_economics.py   # Out-of-sample IC, spread-adjusted PnL, Sharpe & p99 latency
├── tests/
│   ├── __init__.py
│   └── test_lob_foundation.py  # Pytest suite: causal invariance, tensor shapes & cost deduction
├── run_experiment.py           # End-to-end scaling-law & cross-regime benchmark runner
└── README.md
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas pytest

# Verify causal mask invariance, shapes, and spread-cost accounting
pytest -v

# Pre-train Tiny & Base LOB Transformers from scratch and run regime stress tests
python run_experiment.py
```
