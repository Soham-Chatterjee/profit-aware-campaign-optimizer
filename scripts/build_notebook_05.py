"""Generate notebooks/05_validation.ipynb (then execute with nbconvert)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("# 05 · Validation\n\n"
       "Milestone 6. Can the results be trusted? This notebook checks the whole system end to end:\n\n"
       "| Check | Section |\n|---|---|\n"
       "| 6.1 Reproducible from a clean clone | 1 |\n"
       "| 6.2 No leakage: ablation and pipeline placement | 2 |\n"
       "| 6.3 Every strategy respects every constraint | 3 |\n"
       "| 6.4 Backtest on the actual outcomes | 4 |\n"
       "| 6.5 Robustness: GA seeds and economic assumptions | 5 |\n"
       "| 6.6 Fairness monitor | 6 |\n\n"
       "*All economic figures are illustrative assumptions, not bank performance.*"),
    code("from dataclasses import replace\n"
         "import json, subprocess, sys\n"
         "import numpy as np, pandas as pd\n"
         "import plotly.graph_objects as go\n"
         "import plotly.io as pio\n"
         "from pacopt import config as cfg, evaluation as ev, features as ft, model as m, optimize as op, pipeline as pl, charts as ch\n\n"
         "pio.renderers.default = 'notebook_connected'\n"
         "pd.set_option('display.width', 170); pd.set_option('display.max_columns', 25)\n"
         "t = pl.load_campaign_table()\n"
         "comp = op.with_optimisers(t, op.compare_strategies(t, run_genetic=False, run_milp=False), *pl.load_optimisation(t))\n"
         "train, test, universe = m.load_processed()\n"
         "print(t.shape, '| GA and MILP results loaded from artifacts (fingerprint match)')"),

    md("## 1. Reproducible from a clean clone (6.1)\n"
       "`scripts/check_reproducibility.py` copies only the files git would commit (no processed data, "
       "no artifacts) into a fresh folder, runs `python -m pacopt.pipeline` there from the raw CSV, and "
       "compares every headline output with this checkout. It reuses the project environment instead of "
       "a second `uv sync`, and `uv lock --check` confirms that the lockfile matches `pyproject.toml`."),
    code("r = subprocess.run([sys.executable, str(cfg.PROJECT_ROOT / 'scripts' / 'check_reproducibility.py')],\n"
         "                   capture_output=True, text=True, cwd=cfg.PROJECT_ROOT)\n"
         "out = r.stdout\n"
         "print(out[out.find('Pipeline from raw CSV'):])"),

    md("## 2. Leakage checks (6.2)\n"
       "### 2.1 Ablation\n"
       "The same tuned LR, fitted three times. Only the input columns change. `duration` is the length of "
       "the call, known only after the call has happened. `campaign`, `day` and `month` describe *this* "
       "campaign's calls (D-004)."),
    code("ab = ev.leakage_ablation(train, test)\n"
         "ab[['leaky_columns_used', 'n_features', 'roc_auc', 'pr_auc', 'precision_at_1000', 'roc_auc_vs_chosen', 'pr_auc_vs_chosen']].round(4)"),
    code("fig = go.Figure()\n"
         "labels = ['Chosen model', '+ campaign, day, month', '+ duration too']\n"
         "for metric, color, name in (('roc_auc', ch.BLUE, 'ROC-AUC'), ('pr_auc', ch.ORANGE, 'PR-AUC')):\n"
         "    fig.add_bar(x=labels, y=ab[metric], name=name, marker=dict(color=color, cornerradius=4),\n"
         "                text=[f'{v:.3f}' for v in ab[metric]], textposition='outside', textfont=dict(color=ch.INK_2),\n"
         "                hovertemplate='%{x}<br>' + name + ' %{y:.3f}<extra></extra>')\n"
         "ch.style(fig, 'Test-set performance with and without post-campaign columns', height=400)\n"
         "fig.update_layout(barmode='group', bargap=0.3, bargroupgap=0.08)\n"
         "fig.update_yaxes(range=[0, 1.05], title='Score')\n"
         "fig.show()"),
    md("Adding `duration` lifts ROC-AUC from about 0.75 to about 0.91. That is the textbook leak in this "
       "dataset: a model built that way would look excellent and fail on the morning of a real campaign, "
       "because nobody knows the call length before calling. The excluded campaign columns add a "
       "smaller but real +0.02. Our model gives up both on purpose."),
    md("### 2.2 Every fitted transform sits inside the sklearn Pipeline, after the split"),
    code("b = m.load_model()\n"
         "pd.DataFrame({k: ev.transforms_inside_pipeline(b[k]) for k in ('lr', 'rf')}).T"),
    code("print('Model features = allow-list:', b['features'] == ft.MODEL_FEATURES)\n"
         "print('Forbidden columns in features:', sorted(set(b['features']) & cfg.FORBIDDEN_FEATURES) or 'none')\n"
         "print('Universe customers in training data:', len(set(universe.customer_id) & set(train.customer_id)))"),
    md("The scaler and one-hot encoder are the `prep` step of each calibrated fold's Pipeline, so "
       "cross-validation and calibration re-fit them on training folds only. The universe is a subset of "
       "the held-out test set, so every score is out-of-sample (D-014)."),

    md("## 3. Optimisation constraint checks (6.3)\n"
       "`tests/test_validation.py` runs property tests on 35 random scenarios: random customers, costs, "
       "eligibility, budgets, capacities, segment maxima and minima, and protected-share caps. Every "
       "strategy's list passes the independent `check_constraints()`, and repair returns a feasible list "
       "from any starting chromosome. The default scenario's audit is below."),
    code("audit = pd.DataFrame({name: op.check_constraints(t, comp.selected_ids(name)).set_index('constraint')['ok']\n"
         "                      for name in comp.selections})\n"
         "print('All constraints met for every strategy:', bool(audit.all().all()))\n"
         "audit.replace({True: '✓', False: '✗'})"),

    md("## 4. Backtest on actual outcomes (6.4)\n"
       "The campaign universe holds real outcomes (`y`). For each strategy's list, compare the **expected** "
       "conversions and contribution (from P_i) with what these customers **actually** did.\n\n"
       "Realised contribution = V_i summed over the customers who subscribed, minus the contact cost of "
       "everyone on the list.\n\n"
       "*Caveat: `y` was recorded when these customers were contacted in the historical campaign. This "
       "tests whether P_i is accurate on the chosen lists. It is not a causal estimate of a new "
       "campaign's effect (D-015).*"),
    code("bt = ev.backtest(t, comp)\n"
         "bt[['contacts', 'expected_conversions', 'actual_conversions', 'expected_rate', 'actual_rate',\n"
         "    'expected_contribution', 'realised_contribution', 'realised_vs_expected', 'realised_incremental_vs_random']].round(3)"),
    code("names = list(op.CHART_STRATEGIES)\n"
         "fig = go.Figure()\n"
         "fig.add_bar(x=names, y=bt.loc[names, 'expected_contribution'], name='Expected (from P_i)',\n"
         "            marker=dict(color=ch.BLUE, cornerradius=4), hovertemplate='%{x}<br>expected £%{y:,.0f}<extra></extra>')\n"
         "fig.add_bar(x=names, y=bt.loc[names, 'realised_contribution'], name='Realised (actual y)',\n"
         "            marker=dict(color=ch.ORANGE, cornerradius=4), hovertemplate='%{x}<br>realised £%{y:,.0f}<extra></extra>')\n"
         "ch.style(fig, 'Expected vs realised contribution by strategy', height=420)\n"
         "fig.update_layout(barmode='group', bargap=0.3, bargroupgap=0.08)\n"
         "ch.set_value_axis(fig, bt.loc[names, ['expected_contribution', 'realised_contribution']].max().max() * 1.12, money=True)\n"
         "fig.show()"),
    code("ci = ev.bootstrap_backtest(t, comp, ('Genetic Algorithm', 'Greedy', 'Propensity', 'Rule-based'))\n"
         "rnd = bt.loc['Random']\n"
         "ci['random_realised_mean'] = rnd['realised_contribution']\n"
         "ci['random_realised_std'] = rnd['realised_contribution_std']\n"
         "ci.round(0)"),
    md("The backtest confirms the model's accuracy where it matters, on the selected customers: the GA list "
       "was expected to convert about 891 customers and 889 did. Its realised contribution is within about "
       "1% of the expectation, and its 95% bootstrap interval lies well above Random's realised mean "
       "(about £60K, std about £2K across 50 seeds). The ranking also holds on real outcomes: GA ≈ Greedy > "
       "Propensity > Rule-based > Random. Greedy edges the GA on *realised* contribution by about £50, which "
       "is noise: the 95% intervals are about ±£10K."),

    md("## 5. Robustness (6.5)\n### 5.1 GA across 5 seeds"),
    code("runs = ev.ga_seed_runs(t, seeds=range(5), optimum=comp.milp.expected_profit)\n"
         "summary = runs.groupby('seeded').agg(mean=('expected_contribution', 'mean'), std=('expected_contribution', 'std'),\n"
         "                                     min_pct_of_optimum=('pct_of_optimum', 'min'),\n"
         "                                     mean_generations=('generations', 'mean'), mean_runtime_s=('runtime_s', 'mean'),\n"
         "                                     all_feasible=('feasible', 'all'))\n"
         "summary.index = summary.index.map({True: 'Greedy-seeded', False: 'Unseeded'})\n"
         "display(summary.round(4))\n"
         "runs.round(4)"),
    md("### 5.2 Does the strategy ranking survive different economics?\n"
       "One driver changes at a time (±25%, ±50%), and everything else stays at the default. Segments are "
       "re-derived each time, because the value cut-off moves with the economics. Random is averaged over 20 "
       "seeds here, and the GA is capped at 30 s per case."),
    code("base = pl.load_universe().merge(pl.load_scored(), on='customer_id')\n"
         "rs = ev.ranking_sensitivity(base)\n"
         "rs[['change', 'parameter', 'ep_Random', 'ep_Propensity', 'ep_Greedy', 'ep_Genetic Algorithm', 'ep_Rule-based',\n"
         "    'ga_rank', 'ga_vs_greedy', 'ga_vs_random_pct', 'all_feasible', 'ranking']].round(1)"),
    code("print('GA first or tied first in', int(((rs.ga_rank == 1) | (rs.ga_vs_greedy >= -1e-6)).sum()), 'of', len(rs), 'cases')\n"
         "print('GA never below Greedy:', bool((rs.ga_vs_greedy >= -1e-6).all()), '| every list feasible:', bool(rs.all_feasible.all()))\n"
         "print('Propensity beats Greedy in:', rs.loc[rs['ep_Propensity'] > rs['ep_Greedy'], ['parameter', 'change']].to_dict('records') or 'none')\n"
         "gain = rs['ep_Genetic Algorithm'] - rs['ep_Random']\n"
         "print(f'GA gain over Random: £{gain.min():,.0f} to £{gain.max():,.0f}; largest GA gain over Greedy: £{rs.ga_vs_greedy.max():,.0f}')\n"
         "print('Cases where Random or Rule-based lose money:', rs.loc[(rs['ep_Random'] < 0) | (rs['ep_Rule-based'] < 0), ['parameter', 'change']].to_dict('records') or 'none')"),
    md("**The GA comes first, or tied first, in all 13 cases and is never below Greedy.** The rest of the "
       "ordering mostly holds (GA ≥ Greedy > Propensity > Rule-based > Random), with one exception: when "
       "contact costs halve, **Propensity beats Greedy**. With cheap contacts the binding limit is the "
       "6,000-contact cap, not money, so ranking by profit per £ is the wrong rule and favours cheap "
       "contacts. The GA is not tied to one ranking rule, so it handles that case (+£2.7K over Greedy). "
       "This is the clearest example of why an optimiser is worth having over a fixed rule.\n\n"
       "The size of the gain moves a lot. At half the funding spread, an untargeted (Random) or "
       "balance-first (Rule-based) campaign **loses money**, while the GA still earns about £18.6K. The "
       "funding spread is the assumption to confirm first."),

    md("## 6. Fairness monitor review (6.6)\n"
       "Targeting rate by group, compared with the overall rate (about 66% of eligible customers are "
       "selected). A group is flagged if it is targeted at under half or over twice the overall rate, and "
       "only if it has at least 30 eligible customers. This is **reporting only**: no group label enters the "
       "optimiser, which only sees EP."),
    code("fm = ev.fairness_monitor(t, comp.selected_ids('Genetic Algorithm'))\n"
         "print('Flagged groups:', fm[fm.flag != ''][['dimension', 'group', 'rate_ratio', 'flag']].to_dict('records') or 'none')\n"
         "fm.round(3)"),
    code("drv = pd.concat({'age_band': ev.fairness_drivers(t, 'age_band'), 'job': ev.fairness_drivers(t, 'job')})\n"
         "rate = fm.set_index(['dimension', 'group'])['rate_ratio']\n"
         "drv['rate_ratio'] = rate.reindex(drv.index).values\n"
         "drv.round(3)"),
    code("fig = go.Figure()\n"
         "for dim, color in (('age_band', ch.BLUE), ('job', ch.ORANGE)):\n"
         "    f = fm[fm.dimension == dim]\n"
         "    fig.add_bar(x=f['group'], y=f['rate_ratio'], name='Age band' if dim == 'age_band' else 'Job',\n"
         "                marker=dict(color=color, cornerradius=3),\n"
         "                customdata=f[['eligible', 'targeting_rate']],\n"
         "                hovertemplate='%{x}<br>rate ratio %{y:.2f}<br>%{customdata[0]:,} eligible, %{customdata[1]:.0%} targeted<extra></extra>')\n"
         "ch.reference_line(fig, -0.5, 1, 16.5, 1, 'overall rate', label_at='start')\n"
         "for yv, lab in ((0.5, 'flag below 0.5×'), (2.0, 'flag above 2×')):\n"
         "    ch.reference_line(fig, -0.5, yv, 16.5, yv, lab)\n"
         "ch.style(fig, 'Targeting rate relative to the overall rate (GA list)', height=430)\n"
         "fig.update_yaxes(title='Rate ratio', range=[0, 2.2])\n"
         "fig.show()"),
    md("**Review.** No group is flagged, and the skews follow the economics, not the group label:\n\n"
       "- **Customers aged 60+, students and retired customers** are targeted at about 1.3-1.5×. Their "
       "propensity is 2-4× the average of about 12% (prior-campaign success is concentrated there). Every "
       "eligible 75+ customer has positive EP, so all 56 are selected. That is 0.9% of the list, under the "
       "2% cap, and each one is flagged for human review before contact (D-007).\n"
       "- **Housemaids, entrepreneurs and blue-collar workers** are targeted at about 0.7-0.85×. Their "
       "average propensity is about 8%, against 14-30% for management, retired customers and students.\n\n"
       "**A limit of this monitor:** the list covers 66% of eligible customers, so no group can be "
       "targeted above 1/0.66 ≈ 1.51× the overall rate. The 2× over-targeting flag therefore cannot fire "
       "in the default scenario, and several groups are at that ceiling (every eligible member selected). "
       "The *share of list vs share of eligible* columns are the more informative view here. The 0.5× "
       "under-targeting flag still works. Housemaids, the lowest group at 0.68×, are not flagged.\n\n"
       "A business owner should confirm these patterns are acceptable. If not, they can be capped with the "
       "same segment-share constraints the optimiser already supports."),

    md("## Summary\n"
       "| Check | Result |\n|---|---|\n"
       "| Clean-clone reproduction | All headline outputs identical (metrics, EP, segments, every strategy, fingerprint) |\n"
       "| Leakage | Chosen model ROC-AUC ≈ 0.75. With `duration` it would be ≈ 0.91: the leak we excluded |\n"
       "| Transforms | Inside the Pipeline for both models; universe never seen in training |\n"
       "| Constraints | Every strategy feasible in the default case and 35 random scenarios |\n"
       "| Backtest | GA list: 891 expected vs 889 actual conversions; ranking holds on real outcomes |\n"
       "| Robustness | GA reaches the optimum on all 10 seeded and unseeded runs. Under ±50% economics the GA is first or tied in 13/13 cases; Propensity overtakes Greedy when contacts are cheap |\n"
       "| Fairness | No group flagged; skews explained by propensity; 75+ capped and flagged for review. The 2× flag can't fire when 66% of the eligible base is selected |"),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "05_validation.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
