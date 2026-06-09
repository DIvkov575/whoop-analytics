# Whoop Analytics — Brainstorm Handoff

## Goal

Build a personal application that mines **causal relationships** between sleep/HRV data and brain fog, using data from Whoop (both physiological metrics and Journal subjective reports).

## Decisions Made

| Question | Answer |
|----------|--------|
| Scope | Personal-first, designed so it could grow to multi-user |
| Brain fog input | Direct question in Whoop Journal (subjective, daily) |
| Insight type | Causal inference — not just correlation |
| Rigor level | Formal causal inference: DAGs, do-calculus, structural causal models |
| Structure learning | Hybrid: system proposes candidate causal graphs, user validates/edits |
| Data history | 3–6 months (~90–180 daily observations) |
| Interaction model | Report — periodic generated analysis pushed to user |
| Tech stack | Python (best causal inference ecosystem) |
| Repo | Separate project (this one) |

## Whoop API

- Official OAuth 2.0 via developer.whoop.com
- Rate limits: 100 req/min, 10,000/day
- Key data: HRV (rMSSD), resting HR, SpO2, skin temp, sleep stages (light/SWS/REM/awake), efficiency, respiratory rate, disturbances, sleep debt decomposition, strain, Journal entries
- NOT exposed: raw continuous HR, frequency-domain HRV (LF/HF), sleep latency, SDNN/pNN50

## Market Gap

No existing tool combines Whoop's HRV + sleep-stage data with cognitive outcome tracking for causal inference. Rise Science is closest (energy/focus predictions) but lacks Whoop integration and uses no HRV data.

## Relevant Libraries

- **DoWhy** — Microsoft's causal inference framework (effect estimation, refutation tests)
- **CausalNex** — Bayesian network structure learning + causal reasoning (McKinsey/QuantumBlack)
- **pgmpy** — Probabilistic graphical models (structure learning, parameter estimation)
- **Tigramite** — Time-series causal discovery (PCMCI algorithm, designed for exactly this type of data)
- **DAGitty** — DAG editor (web-based, could embed for graph validation)

## Key Design Considerations

- ~90–180 daily observations is tight for structure learning → keep variable set focused, use time-series methods (Tigramite/PCMCI suited for small-N)
- Temporal structure matters: sleep variables at night → fog next day. Exploit time ordering as a causal constraint.
- Report format: periodic (daily or weekly) generated causal analysis
- System proposes causal structure, user validates before estimating effects

## Relevant Open Source

- `maxnau89/openclaw-biohub` — self-hosted health dashboard (Python/Next.js)
- `ald0405/whoop-data` — LangGraph multi-agent AI coaching
- `shashankswe2020-ux/whoop-mcp` — MCP server for LLM Whoop queries
- `hedgertronic/whoop` — simple Python API client

## Next Steps

1. Propose 2–3 architectural approaches
2. Present design for approval
3. Write spec → implementation plan
