# Semantic Consensus Framework (SCF)

**Process-Aware Conflict Detection and Resolution for Enterprise Multi-Agent LLM Systems**

[![Paper](https://img.shields.io/badge/Paper-Applied%20Sciences-blue)](https://www.mdpi.com/journal/applsci)

## Overview

SCF is a middleware layer that detects and resolves semantic conflicts between cooperating LLM-based agents in enterprise environments. It addresses **Semantic Intent Divergence** — the phenomenon where agents develop inconsistent interpretations of shared objectives.

### Key Results
- **74.3%** reduction in semantic conflict incidence
- **91.7%** conflict detection precision
- **88.4%** conflict detection recall
- **93.4%** workflow completion rate
- **145ms** median governance overhead (<3% of interaction time)

## Architecture

SCF comprises six components:

1. **Process Context Layer (PCL)** — Shared operational semantics from enterprise workflow models
2. **Semantic Intent Graph (SIG)** — Captures agent intents before execution
3. **Conflict Detection Engine (CDE)** — Detects Type 1/2/3 semantic conflicts
4. **Consensus Resolution Protocol (CRP)** — Policy → Authority → Temporal resolution hierarchy
5. **Drift Monitor (DM)** — Continuous semantic alignment monitoring
6. **Governance Integration (PAGI)** — Policy mapping, audit trails, dashboard

## Quick Start

```bash
pip install -r requirements.txt

# Run all experiments (600 runs across 4 scenarios)
python -m experiments.runner --scenario all --runs 50 --seed 42

# Run a single scenario
python -m experiments.runner --scenario financial --runs 50 --seed 42
```

## Project Structure

```
semantic-consensus/
├── scf/                          # Core framework
│   ├── core/
│   │   ├── sig.py               # Semantic Intent Graph
│   │   └── pcl.py               # Process Context Layer
│   ├── detect/
│   │   └── cde.py               # Conflict Detection Engine
│   ├── resolve/
│   │   └── crp.py               # Consensus Resolution Protocol
│   ├── drift/
│   │   └── monitor.py           # Drift Monitor
│   ├── governance/
│   │   └── pagi.py              # Governance Integration
│   └── middleware.py             # Main SCF API
├── process_models/               # Enterprise workflow models (YAML)
│   ├── financial_processing.yaml
│   ├── customer_support.yaml
│   ├── supply_chain.yaml
│   └── software_development.yaml
├── experiments/
│   ├── runner.py                # Experiment orchestrator
│   └── results/                 # Output data
├── requirements.txt
└── README.md
```

## Citation

If you use this work, please cite:

```bibtex
@article{acharya2026semantic,
  title={Semantic Consensus: Process-Aware Conflict Detection and Resolution for Enterprise Multi-Agent LLM Systems},
  author={Acharya, Vivek},
  journal={Applied Sciences},
  year={2026},
  publisher={MDPI}
}
```

## Author

**Vivek Acharya** — Independent Researcher, Boston University
- Email: vacharya@bu.edu
- ORCID: [0009-0002-0860-9462](https://orcid.org/0009-0002-0860-9462)
- GitHub: [curiosityexplorer](https://github.com/curiosityexplorer)
