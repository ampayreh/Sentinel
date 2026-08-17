# Sentinel Super-Repo: NVIDIA Grace Blackwell GB10 Production System

The **Sentinel Super-Repo** combines the mathematical & statistical rigor of **Halima**, the real-world dataset integrations and Glassmorphic Web Dashboard of **Ashish**, and the generalized fuzz-testing suite of **Graeme**.

---

## 🏛️ Super-Repo Architecture & Stack

```
                          ┌─────────────────────────────────────────────────────────┐
                          │              NVIDIA GB10 UNIFIED MEMORY                │
                          │                 128 GB LPDDR5x (900 GB/s)              │
                          └────────────────────────────┬────────────────────────────┘
                                                       │
                       ┌───────────────────────────────┴───────────────────────────────┐
                       ▼                                                               ▼
          ┌─────────────────────────────┐                             ┌────────────────────────────────┐
          │ DETERMINISTIC CORE SAFETY   │                             │ COGNITIVE ADVISORY & BENCHMARK │
          │ (Conformal GRU Autoencoder) │                             │ (Nemotron 3.5 Lightning / Ollama)│
          │ • M1 Conformal Autoencoder  │                             │ • Async Non-Blocking Task      │
          │ • M2 CUDA Sensor Agreement  │  ◄── Asymmetric Monotonic ──┤ • Ollama API / vLLM Binding    │
          │ • M3/M4 Timing & ODD Gate   │      Authority Restriction  │ • GB10 Hardware Telemetry      │
          └────────────┬────────────────┘                             └────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────────────────────────────────────────────────────────┐
          │                     EVIDENCE LOGGER & REPLAY SYSTEM                        │
          │  • SHA-256 Hash Chain • Bit-Identical Replay • JSON Lines Audit Log        │
          └────────────┬───────────────────────────────────────────────────────────────┘
                       │
                       ├───────────────────────────────────────┐
                       ▼                                       ▼
          ┌───────────────────────────┐           ┌────────────────────────────┐
          │ RICH TERMINAL UI (TUI)    │           │ FASTAPI + WEBSOCKET WEB UI │
          │ `sentinel run --dashboard`│           │ `sentinel run --web`       │
          └───────────────────────────┘           └────────────────────────────┘
```

---

## 🚀 Quickstart & Operations

### 1. Setup & Verification
```bash
cd /home/acer01/Downloads/Halima/sentinel_super_repo
make setup             # Sets up .venv and installs dependencies
make test              # Runs 48 unit, property, and monotonicity fuzz tests
make agent             # Proves advisory monotonicity fuzzing (3,000 cases)
```

### 2. Live Web Dashboard (FastAPI + WebSockets)
```bash
make web
# Starts live Glassmorphic Web Dashboard on http://localhost:8765
```

### 3. Evaluation & Hardware Benchmarking
```bash
make eval
# Generates full evaluation results and latency percentiles in eval_results/results.md
```
