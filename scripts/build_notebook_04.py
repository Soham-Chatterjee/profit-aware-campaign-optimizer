"""Generate notebooks/04_genetic_algorithm.ipynb (then execute with nbconvert)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("# 04 · Genetic algorithm and strategy comparison\n\n"
       "Milestone 4. Engine 3 answers **Q4** (which combination of customers to target) and **Q5** "
       "(how much more profit than the existing strategies).\n\n"
       "```\n"
       "maximise   sum_i x_i * EP_i                       x_i in {0, 1}, eligible customers only\n"
       "subject to sum_i x_i * C_i            <= budget                (£50,000)\n"
       "           sum_i x_i                  <= max contacts          (6,000)\n"
       "           sum_{i in g} x_i           <= channel capacity      (calls 4,500, digital 2,500)\n"
       "           share of each segment      <= 60%\n"
       "           share Growing >= 15%, share Dormant >= 10%\n"
       "           share of protected 75+     <= 2%\n"
       "```\n\n"
       "*All economic figures are illustrative assumptions, not bank performance.*"),
    code("from dataclasses import asdict, replace\n"
         "import time\n"
         "import numpy as np, pandas as pd\n"
         "import plotly.graph_objects as go\n"
         "import plotly.io as pio\n"
         "from pacopt import config as cfg, optimize as op, pipeline as pl, charts as ch\n\n"
         "pio.renderers.default = 'notebook_connected'\n"
         "pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)\n"
         "t = pl.load_campaign_table()\n"
         "pr = op.build_problem(t)\n"
         "print(f'{len(t):,} customers | {pr.n:,} eligible enter the optimisation | {(pr.ep > 0).sum():,} with positive EP')\n"
         "pd.DataFrame({'constraints': pd.Series(asdict(cfg.CONSTRAINTS)), }).astype(str)"),

    md("## 1. Design (sub-tasks 4.1-4.4)\n"
       "| Component | Choice |\n|---|---|\n"
       "| Chromosome | Binary NumPy vector, one gene per eligible customer with positive EP, plus the members of "
       "segments with a minimum share (a negative-EP customer can only help by meeting a minimum). "
       "The population is a (100 x n) boolean matrix, so every operator is vectorised. |\n"
       "| Fitness | Total EP of the list, minus a large penalty per unit of constraint violation "
       "(the fallback if repair ever leaves a violation). |\n"
       "| Repair | Shared constraint engine: drop the weakest selections until channel, contact and budget "
       "limits hold; trim over-represented segments; add the best members of under-represented ones; "
       "then **fill** spare capacity in EP-per-£ order (D-036). |\n"
       "| Selection / crossover / mutation | Tournament (k = 3) / uniform (p = 0.9) / asymmetric bit-flip: "
       "about 30 genes switched on and 0.5 off per child, so with repair a mutation works as a set of swaps. |\n"
       "| Elitism / stopping | Top 2% kept / 300 generations or 50 without improvement. |\n"
       "| Seeding | 4 of 100 chromosomes start from the Greedy list (and perturbed copies); the rest are random "
       "sparse lists at the density the budget allows. An unseeded run is reported too. |\n\n"
       "Every strategy, not just the GA, goes through the same constraint engine. `check_constraints()` "
       "then audits every list independently, from the campaign table and the customer IDs alone."),

    md("## 2. Run all strategies (sub-task 4.5)"),
    code("t0 = time.perf_counter()\n"
         "comp = op.compare_strategies(t)\n"
         "print(f'All strategies in {time.perf_counter() - t0:.0f}s | GA: {comp.ga.generations} generations, '\n"
         "      f'{comp.ga.runtime_s:.1f}s | MILP: {comp.milp.message.split(\"(\")[0].strip()} in {comp.milp.runtime_s:.0f}s, '\n"
         "      f'gap {comp.milp.mip_gap:.1e}')\n"
         "cols = ['contacts', 'contacts_calls', 'contacts_digital', 'spend', 'expected_conversions', 'expected_contribution',\n"
         "        'roi', 'incremental_vs_random', 'uplift_vs_random', 'pct_of_optimum', 'feasible']\n"
         "summary = comp.table[cols]\n"
         "summary.style.format({'spend': '£{:,.0f}', 'expected_conversions': '{:,.0f}', 'expected_contribution': '£{:,.0f}',\n"
         "                      'roi': '{:.2f}', 'incremental_vs_random': '£{:+,.0f}', 'uplift_vs_random': '{:+.0%}',\n"
         "                      'pct_of_optimum': '{:.2%}', 'contacts': '{:,}', 'contacts_calls': '{:,.0f}',\n"
         "                      'contacts_digital': '{:,.0f}'})"),
    md("*ROI = expected contribution / spend. Random is the mean of 50 seeds "
       f"(standard deviation shown below). Rule-based = highest balance first.*"),
    code("r = comp.random_runs['expected_contribution']\n"
         "print(f'Random baseline over {len(r)} seeds: mean £{r.mean():,.0f}, std £{r.std():,.0f}, '\n"
         "      f'range £{r.min():,.0f} to £{r.max():,.0f}; all feasible: {comp.random_runs.feasible.all()}')"),

    md("## 3. Expected contribution by strategy (the Screen 3 view, sub-task 4.7)"),
    code("names = list(op.CHART_STRATEGIES)\n"
         "vals = comp.table.loc[names, 'expected_contribution']\n"
         "fig = go.Figure(go.Bar(x=names, y=vals, marker_color=ch.BAR_BLUE, marker_cornerradius=8,\n"
         "                       text=[f'£{v / 1e3:,.1f}K' for v in vals], textposition='outside',\n"
         "                       textfont=dict(color=ch.INK_2),\n"
         "                       hovertemplate='%{x}<br>£%{y:,.0f}<extra></extra>'))\n"
         "ch.style(fig, 'Expected contribution by strategy (£50K budget)', height=430, showlegend=False)\n"
         "ch.reference_line(fig, -0.5, comp.milp.expected_profit, len(names) - 0.5, comp.milp.expected_profit,\n"
         "                  label=f'Theoretical max (MILP) £{comp.milp.expected_profit / 1e3:,.1f}K', label_at='start')\n"
         "fig.update_yaxes(tickprefix='£', tickformat='~s', rangemode='tozero', range=[0, comp.milp.expected_profit * 1.15])\n"
         "fig.update_layout(bargap=0.35)\n"
         "fig.show()"),
    code("best_baseline = comp.table.loc[['Random', 'Propensity', 'Rule-based'], 'expected_contribution']\n"
         "ga = comp.table.loc['Genetic Algorithm']\n"
         "print('Q5: incremental expected contribution of the GA list')\n"
         "for k, v in best_baseline.items():\n"
         "    print(f'  vs {k:<11s} £{ga.expected_contribution - v:>+10,.0f}  ({ga.expected_contribution / v - 1:+.0%})')\n"
         "print(f\"  vs Greedy      £{ga.expected_contribution - comp.table.loc['Greedy', 'expected_contribution']:>+10,.0f}\")\n"
         "print(f'GA reaches {ga.pct_of_optimum:.2%} of the MILP optimum.')"),
    md("**Reading the result honestly.** The GA beats Random by about 70% and Propensity by about 14%. "
       "Most of that gain comes from **ranking by profit per £, not from the GA itself**: Greedy "
       "(EP / C order) already reaches 99.96% of the optimum. In the default scenario the only limit "
       "binding at the optimum is the 6,000-contact cap. The optimum leaves about £2.6K of budget and "
       "100 digital slots unused, and no segment share binds (see section 6). So the problem is close "
       "to a knapsack, which a greedy ratio rule solves almost exactly. Greedy fills all 2,500 digital "
       "slots first because they are cheap. Once the head-count cap binds, though, a contact's EP "
       "matters more than its EP per £. The GA finds the last £41 by trading 100 digital contacts for "
       "100 calls with higher EP, a swap a single ranking cannot see. Section 7 shows scenarios where "
       "the constraints interact and the GA's gain over Greedy is material."),

    md("## 4. Convergence (sub-task 4.6)\n"
       "The seeded run starts from the Greedy list and is already near the optimum at generation 0, so "
       "the unseeded run is the clearer view of the algorithm's search."),
    code("ga0 = op.run_ga(pr, replace(cfg.GA, greedy_seeds=0))\n"
         "print(f'Unseeded GA: £{ga0.expected_profit:,.0f} ({ga0.expected_profit / comp.milp.expected_profit:.2%} of optimum) '\n"
         "      f'in {ga0.generations} generations, {ga0.runtime_s:.1f}s | seeded: {comp.ga.generations} generations, {comp.ga.runtime_s:.1f}s')\n"
         "fig = go.Figure()\n"
         "for h, name, color in ((ga0.history, 'Unseeded', ch.BLUE), (comp.ga.history, 'Greedy-seeded', ch.ORANGE)):\n"
         "    fig.add_scatter(x=h['generation'], y=h['best'], name=f'{name}: best', mode='lines', line=dict(color=color, width=2),\n"
         "                    hovertemplate=name + ' best<br>generation %{x}<br>£%{y:,.0f}<extra></extra>')\n"
         "    fig.add_scatter(x=h['generation'], y=h['mean'], name=f'{name}: population mean', mode='lines',\n"
         "                    line=dict(color=color, width=1.5, dash='dot'),\n"
         "                    hovertemplate=name + ' mean<br>generation %{x}<br>£%{y:,.0f}<extra></extra>')\n"
         "g_end = max(ga0.generations, comp.ga.generations)\n"
         "ch.reference_line(fig, 0, comp.milp.expected_profit, g_end, comp.milp.expected_profit, 'MILP optimum')\n"
         "greedy_ep = comp.table.loc['Greedy', 'expected_contribution']\n"
         "ch.style(fig, 'GA convergence: best and mean fitness per generation', height=450)\n"
         "fig.update_xaxes(title='Generation')\n"
         "fig.update_yaxes(title='Expected contribution (£)', tickprefix='£', tickformat=',.0f',\n"
         "                 range=[ga0.history['mean'].min() * 0.995, comp.milp.expected_profit * 1.003])\n"
         "fig.show()\n"
         "print(f'Greedy for reference: £{greedy_ep:,.0f}')"),

    md("## 5. The GA target list: channel and segment mix (Q4)"),
    code("seg_mix = op.selection_mix(t, comp, 'segment')\n"
         "ch_mix = op.selection_mix(t, comp, 'channel_group')\n"
         "display(seg_mix.style.format('{:.1%}').set_caption('Segment share of each list (Random = seed 0)'))\n"
         "ch_mix.style.format('{:.1%}').set_caption('Channel share of each list')"),
    code("tl = op.target_list(t, comp)\n"
         "print(f'GA target list: {len(tl):,} customers, spend £{tl.contact_cost.sum():,.0f}, '\n"
         "      f'expected conversions {tl.propensity.sum():,.0f}, expected contribution £{tl.expected_profit.sum():,.0f}')\n"
         "tl.head(10).round(3)"),

    md("## 6. Constraint compliance (sub-task 4.3)\n"
       "`check_constraints()` audits each list from the campaign table and customer IDs alone. Below: "
       "the value of every constraint for each strategy, with ✓ when it holds."),
    code("audit = {}\n"
         "for name in comp.selections:\n"
         "    ck = op.check_constraints(t, comp.selected_ids(name)).set_index('constraint')\n"
         "    audit[name] = [(f'{v:.1%}' if 'share' in c.lower() else f'{v:,.0f}') + (' ✓' if ok else ' ✗')\n"
         "                   for c, v, ok in zip(ck.index, ck['value'], ck['ok'])]\n"
         "limits = op.check_constraints(t, comp.selected_ids('Greedy')).set_index('constraint')['limit']\n"
         "audit = pd.DataFrame(audit, index=limits.index)\n"
         "audit.insert(0, 'limit', [f'{v:.0%}' if 'share' in c.lower() else f'{v:,.0f}' for c, v in limits.items()])\n"
         "audit"),
    code("bind = op.check_constraints(t, comp.selected_ids('MILP optimum')).set_index('constraint')\n"
         "slack = (bind['limit'] - bind['value']).where(~bind.index.str.startswith('Min'), bind['value'] - bind['limit'])\n"
         "print('Binding constraints at the optimum (slack 0):', ', '.join(slack.index[(slack.abs() < 1e-9) & ~slack.index.str.contains('customers')]))"),

    md("## 7. Where the GA matters: stressed constraint scenarios\n"
       "Same customers and economics; only the campaign limits change. These are what-ifs, not new "
       "defaults (D-037)."),
    code("scenarios = {\n"
         "    'Default': cfg.CONSTRAINTS,\n"
         "    'Few call agents (calls 1,500; Growing >= 20%, Dormant >= 15%)':\n"
         "        replace(cfg.CONSTRAINTS, channel_capacity={'calls': 1_500, 'digital': 2_500},\n"
         "                segment_min_share={'Growing': 0.20, 'Dormant': 0.15}),\n"
         "    'Balanced mix (High value <= 30%; Growing, Dormant >= 25%)':\n"
         "        replace(cfg.CONSTRAINTS, segment_max_share={**{s: 0.60 for s in cfg.SEGMENTS}, 'High value': 0.30},\n"
         "                segment_min_share={'Growing': 0.25, 'Dormant': 0.25}),\n"
         "}\n"
         "rows = {}\n"
         "for name, cc in scenarios.items():\n"
         "    c = comp if name == 'Default' else op.compare_strategies(t, cc)\n"
         "    tb = c.table['expected_contribution']\n"
         "    rows[name] = {**tb.to_dict(), 'GA gain vs Greedy': tb['Genetic Algorithm'] - tb['Greedy'],\n"
         "                  'Greedy % of optimum': tb['Greedy'] / tb['MILP optimum'],\n"
         "                  'GA % of optimum': tb['Genetic Algorithm'] / tb['MILP optimum'],\n"
         "                  'GA runtime (s)': c.ga.runtime_s, 'all feasible': bool(c.table['feasible'].all())}\n"
         "stress = pd.DataFrame(rows).T\n"
         "money = list(op.STRATEGY_ORDER) + ['GA gain vs Greedy']\n"
         "stress.style.format({**{k: '£{:,.0f}' for k in money}, 'Greedy % of optimum': '{:.2%}',\n"
         "                     'GA % of optimum': '{:.2%}', 'GA runtime (s)': '{:.0f}'})"),
    md("When a segment minimum or a tight channel cap binds, the ratio order makes locally good choices "
       "that clash with the other limits, and repairing them afterwards costs profit. The GA searches "
       "across whole lists, so it recovers most of that loss. This is the case the plan anticipated: "
       "the GA's value rises with how much the constraints interact.\n\n"
       "The stressed scenarios take longer, because repair works harder when several share limits bind "
       "at once. The 60-second target applies to the default run. For dashboard re-runs, "
       "`GAConfig.time_limit_s` caps the wall-clock time, and the best feasible list found so far is "
       "returned."),

    md("## 8. Save the optimisation artifacts"),
    code("pl.save_optimisation(t, comp)\n"
         "for p in (pl.STRATEGY_PATH, pl.TARGET_LIST_PATH, pl.GA_HISTORY_PATH, pl.SELECTIONS_PATH):\n"
         "    print(p.relative_to(cfg.PROJECT_ROOT))"),
    md("**Milestone 4 output:** the strategy comparison, the GA target list, the convergence history and "
       "every strategy's selection, all under `artifacts/`. The dashboard (M5) reads these and re-runs "
       "the optimiser when the sliders change."),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "04_genetic_algorithm.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
