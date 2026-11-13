# Profit-Aware Campaign Optimizer

A prototype that decides **whom a bank should contact** in a term-deposit campaign, and **through which channel**, so that a fixed budget earns the most expected profit within responsible-targeting rules.

It is built on the public UCI Bank Marketing dataset (45,211 customers, a 2008–2010 Portuguese campaign). **All economic figures are illustrative assumptions, not bank performance.**

## Results at a glance (default scenario)

| | Random | Rule-based | Propensity | Greedy | **Genetic Algorithm** | Exact optimum |
|---|---|---|---|---|---|---|
| Expected contribution | £60.7K | £83.8K | £91.2K | £103.75K | **£103.79K** | £103.79K |
| Realised on actual outcomes | £59.7K | £86.6K | £90.9K | £102.9K | **£102.8K** | £102.8K |
| Contacts / spend | 5,517 / £50.0K | 5,374 / £50.0K | 4,275 / £50.0K | 6,000 / £46.4K | 6,000 / £47.4K | 6,000 / £47.4K |

With a £50K budget, the optimised list earns **about £43K (+71%) more than an untargeted campaign** and **£12.5K (+14%) more than targeting by propensity alone**. It reaches 100% of the proven optimum. Most of the gain comes from ranking customers by **profit per £**, not by likelihood alone. The genetic algorithm matters most when the campaign's constraints interact (see [Q5](#q5-how-much-extra-profit)).

## Setup and run

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 itself if needed.

```bash
uv sync                                          # create .venv from uv.lock (exact versions)
uv run python -m pacopt.pipeline                 # raw CSV -> features -> model -> economics -> optimisation (~1-3 min)
uv run streamlit run app/Home.py                 # dashboard at http://localhost:8501
uv run pytest                                    # full test suite (~5 min)
```

| Command | Purpose |
|---|---|
| `uv run python -m pacopt.pipeline --no-retrain` | Reuse the saved propensity scores and recompute economics + optimisation |
| `uv run python -m pacopt.pipeline --no-optimise` | Stop after the campaign table |
| `uv run python scripts/check_reproducibility.py` | Rebuild everything in a clean copy of the repo and compare it with this checkout |
| `uv run jupyter lab` | Notebooks 01–05 (kernel = the project `.venv`) |
| `uv run python scripts/build_notebook_0N.py` | Regenerate notebook N from its builder script |
| `uv run --group docs python scripts/build_plan_pdf.py` | Rebuild the plan PDF |

The raw data is committed (`data/raw/`, CC BY 4.0, with a SHA-256 check). `data/processed/` and `artifacts/` are git-ignored and rebuilt by the pipeline. The dashboard needs the pipeline to have run once.

## How it works

```
bank-full.csv ──► features.py ──► ENGINE 1  Customer Intelligence   model.py, segmentation.py
                  RFM-inspired      calibrated propensity P_i, four segments, RFM / engagement overlays
                  eligibility       │
                  leakage guard     ▼
                                  ENGINE 2  Profit Intelligence     economics.py
                                    V_i = deposit × funding spread − incentive − onboarding
                                    EP_i = P_i × V_i − C_i      (the principal is never revenue)
                                    │
                                    ▼
                                  ENGINE 3  Optimization Intelligence   optimize.py
                                    genetic algorithm + Random / Propensity / Greedy / Rule-based
                                    exact MILP optimum, one shared constraint checker
                                    │
                                    ▼
                  pipeline.py ──► artifacts/ ──► Streamlit dashboard (app/) and notebooks 01–05
```

| Module | Role |
|---|---|
| `config.py` | Every assumption: economics, channels, eligibility, constraints, model and GA settings |
| `data.py`, `features.py` | Download / validate / clean; RFM-inspired features, eligibility flags, leakage allow-list, split |
| `model.py` | Logistic Regression (primary, isotonic-calibrated) and Random Forest (challenger); evaluation |
| `economics.py`, `segmentation.py` | Deposit size, value, cost, break-even, expected profit; segments and RFM scores |
| `optimize.py` | Constraint engine, strategies, GA, MILP, `check_constraints()` |
| `scenario.py`, `evaluation.py` | Editable scenarios for the dashboard; validation (ablation, backtest, robustness, fairness) |
| `pipeline.py`, `app/` | One-command pipeline; the five-page dashboard |

Each design choice, the alternatives considered and the reasons are in [docs/DECISIONS.md](docs/DECISIONS.md) (D-001 to D-040). The full plan is in [plans/implementation-plan.md](plans/implementation-plan.md).

## Answers to the five business questions

### Q1. Which customers are most likely to subscribe?
A calibrated Logistic Regression scores every customer, using pre-campaign information only.

- **Accuracy:** test-set ROC-AUC 0.753 and PR-AUC 0.380, against an 11.7% base rate.
- **Precision:** the 1,000 highest-scored customers convert at **42.8% (3.7× lift)**.
- **Calibration:** predicted and actual rates agree within 1.2 percentage points in every decile.
- **Strongest drivers:** contact channel (customers without a phone channel respond less), age band (60+ respond more, 30-59 less), recency of the last campaign contact, and success in a previous campaign.

*Dashboard: Propensity & Opportunities.*

### Q2. Which customers have the greatest financial value?
The deposit is valued as a funding product: the bank earns the 1.75% funding spread on the deposit, never the principal itself.

- **Concentration:** the top 10% of eligible customers by expected profit hold **58% of all positive expected profit**.
- **Segments** (from a propensity-by-value grid):

  | Segment | Customers |
  |---|---|
  | High value | 2,527 |
  | Growing | 1,482 |
  | Dormant | 2,475 |
  | Low engagement | 3,516 |

- **Dormant customers:** high balances but low propensity, and 88% have never been contacted.

*Dashboard: Customer Intelligence.*

### Q3. How much should the bank spend to acquire each customer?
At most the **break-even cost P_i × V_i**. Contact is worthwhile when this exceeds the channel cost (£12 mobile, £10 landline, £2 digital).

- **Typical customer:** the median eligible customer is worth about £11 of contact.
- **Worth contacting:** 74% of eligible customers are worth more than their channel cost.
- **Best customer:** worth up to £525.
- **Diminishing returns:** with the default limits, extra budget stops adding profit at **£46.4K**. Beyond that the 6,000-contact cap and digital capacity bind, not money.

*Dashboard: What-if Simulation.*

### Q4. Which combination of customers should be targeted?
The GA list contains **6,000 customers (3,600 calls, 2,400 digital)** and spends £47.4K.

- **Segment mix:** 39% High value, 18% Growing, 26% Dormant, 17% Low engagement.
- **Compliance:** it meets every constraint — budget, contacts, channel capacity, segment maximum and minimum shares, and at most 2% aged 75+.
- **Protected customers:** the 56 customers aged 75+ (0.9%) are flagged for human review before contact.

*Dashboard: Campaign Optimization (downloadable list).*

### Q5. How much extra profit?
The GA list's gain over each baseline, in expected contribution:

| Compared with | Extra profit |
|---|---|
| Random (untargeted) | **+£43.1K (+71%)** |
| Rule-based (highest balance first) | +£20.0K (+24%) |
| Propensity-only | +£12.5K (+14%) |

The gain holds on actual outcomes: the GA list was expected to convert 891 customers and 889 did. Its realised contribution has a 95% bootstrap interval of £93K–£112K, against about £60K for Random.

In the default scenario, Greedy (profit per £) already gets within £41 of the optimum, because only the contact cap binds. The GA's advantage grows when constraints interact or when a fixed ranking rule stops being the right one:

| Scenario | GA gain over Greedy |
|---|---|
| Few call agents | +£3.5K |
| Balanced segment mix | +£1.6K |
| Contact costs halved | +£2.7K |

When contact costs halve, the contact cap binds before the budget does, so ranking by profit per £ favours cheap contacts and even falls behind Propensity.

## Validation (notebook 05)

| Check | Result |
|---|---|
| Clean-clone reproduction | Identical metrics, expected profit, segments and strategy results |
| Leakage ablation | Chosen model ROC-AUC 0.753. Adding `campaign`/`day`/`month` gives 0.776, and adding `duration` gives **0.908**: the leak the model avoids |
| Transforms | Scaling and encoding sit inside the sklearn Pipeline for both models, and the campaign universe is never used in training |
| Constraints | Every strategy is feasible in the default case and in 35 random property-test scenarios |
| Robustness | GA reaches the optimum on 10 of 10 runs (5 seeds, seeded and unseeded). Under ±50% changes to spread, incentive and contact cost, the GA is first or tied in 13 of 13 cases. When contact costs halve, Propensity overtakes Greedy and the GA leads Greedy by £2.7K. At half the spread, Random and Rule-based lose money while the GA earns £18.6K |
| Fairness monitor | No group flagged. Skews follow propensity (e.g. retired customers and students at about 1.3–1.5×, housemaids at 0.68×). With 66% of eligible customers selected, no group can exceed about 1.51×, so the 2× flag cannot fire in this scenario; the share-of-list vs share-of-eligible columns are the better guide |

## Assumptions and limitations

1. **Not true RFM.** The data has no transactions, so the features are RFM-*inspired*: recency of the last campaign contact, number of previous contacts, and balance.
2. **Balance is not revenue.** It only sizes the expected deposit: £10,000 minimum, plus the balance, capped at £100,000.
3. **The economics are illustrative.** The 1.75% funding spread, the £40 incentive, the £5 onboarding cost and the channel costs are assumptions, all editable in the dashboard. **The funding spread dominates:** a 25% lower spread roughly halves the positive expected profit. Confirm it with Treasury before using any figure.
4. **"Incremental" means relative to the baseline strategies, not causal uplift.** P_i is P(subscribe | contacted). There is no control group, so the organic rate P0 defaults to 0 and is configurable.
5. **Historical data.** The source is a 2008–2010 Portuguese campaign, and balances are euros shown as £. The results demonstrate the method, not current bank performance.
6. **The channel is fixed** by the `contact` field. The model does not choose channels.
7. **The backtest uses the historical outcomes** of customers who were contacted then. It checks the accuracy of P_i on the selected lists, not the effect of a new campaign.
8. **Fairness is monitored, not enforced.** Age 75+ is capped and flagged for review. Other skews are reported for a business owner to judge.

## Demo script (about 10 minutes)

1. **Command Centre (1 min).** Introduce the four headline numbers: 10,000 customers analysed, 14.8% average propensity on the list, a £50K budget and £103.8K of expected contribution. Point out the "+£43.1K vs Random" figure. Then run through the five question cards, which are the storyline.
2. **Customer Intelligence (2 min).**
   - Show the segment chart, which matches the brief's reference.
   - Use the quadrant scatter to explain the grid: High value is high on both axes; Dormant is valuable but unlikely.
   - Show the eligibility table: default, negative balance and fatigue are excluded, and 75+ customers are flagged rather than excluded.
3. **Propensity & Opportunities (2 min).**
   - Search for a customer and switch on **Economics** to show V_i, the break-even cost and EP_i (Q3).
   - Show the top-N opportunities.
   - In model performance, show the calibration tab: the predictions are trustworthy as probabilities, which matters because they are multiplied by money.
4. **Campaign Optimization (3 min).**
   - Present the headline chart: Random → Propensity → Greedy → GA against the dashed optimum line.
   - Be candid: Greedy is already near-optimal here. Tighten a constraint and the GA pulls ahead.
   - Walk through the constraint checklist, the channel and segment mix, and the fairness monitor.
   - Download the target list.
5. **What-if Simulation (2 min).**
   - Move the funding spread from 1.75% to 1.30% and watch the contribution fall. This is the key assumption to confirm with Treasury.
   - Show the profit-vs-budget curve: beyond £46K, more budget buys nothing until the contact cap is raised.
   - Click **Run GA on this what-if**.
6. **Close.** Every number comes from the pipeline. The method is reproducible from a clean clone, and every decision is logged.

## Project layout

```
app/              Streamlit dashboard: Home.py, common.py, views/ (5 pages)
src/pacopt/       the package (see "How it works")
notebooks/        01 data · 02 model · 03 profit engine · 04 GA · 05 validation
scripts/          notebook builders, reproducibility check, plan PDF builder
tests/            unit, integration, property and dashboard tests
data/raw/         bank-full.csv (committed); data/processed/ is generated
artifacts/        model, metrics, strategy results (generated)
docs/DECISIONS.md decision log
plans/            implementation plan (.md + .pdf) and the reference images
```

## Data and licence

The data is the UCI Bank Marketing dataset by Moro, Rita & Cortez (2014), [doi:10.24432/C5K306](https://doi.org/10.24432/C5K306), used under CC BY 4.0 (see [data/raw/README.md](data/raw/README.md)). The code is licensed under this repository's [LICENSE](LICENSE).
