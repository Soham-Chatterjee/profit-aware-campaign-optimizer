"""Generate notebooks/03_profit_engine.ipynb (then execute with nbconvert)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("# 03 · Profit engine and segmentation\n\n"
       "Milestone 3. Engine 2 turns each customer's propensity into money, and Engine 1 places "
       "each customer in one of four business segments.\n\n"
       "The term deposit is a **funding product** (D-008). The bank doesn't earn the principal. It "
       "earns the funding spread (FTP rate minus customer rate) on the deposit over the term.\n\n"
       "```\n"
       "D_i  = clip(D_min + alpha x max(balance_i, 0), D_min, D_max)      deposit size\n"
       "V_i  = D_i x spread x term - incentive - onboarding                 contribution if converted\n"
       "C_i  = channel cost (fully loaded: mobile £12, landline £10, digital £2)\n"
       "BE_i = (P_i - P0) x V_i                                             break-even contact cost (Q3)\n"
       "EP_i = BE_i - C_i                                                   expected incremental profit\n"
       "```\n\n"
       "*All economic figures are illustrative assumptions, not bank performance. Balances are euros "
       "in the source data and are shown as £ for illustration.*"),
    code("from dataclasses import asdict, replace\n"
         "import numpy as np, pandas as pd\n"
         "import plotly.graph_objects as go\n"
         "import plotly.io as pio\n"
         "from pacopt import config as cfg, economics as econ, segmentation as seg, pipeline as pl, charts as ch\n\n"
         "pio.renderers.default = 'notebook_connected'\n"
         "pd.set_option('display.width', 140)\n"
         "base = pl.load_universe().merge(pl.load_scored(), on='customer_id')   # features + propensity\n"
         "t = pl.campaign_table()                                                # + economics + segments\n"
         "print(t.shape)\n"
         "a = asdict(cfg.ECONOMICS); a['spread'] = cfg.ECONOMICS.spread\n"
         "pd.Series(a, name='default').to_frame()"),

    md("## 1. Calibrating the default assumptions (sub-task 3.3)\n"
       "The plan's starting defaults (D_min £5,000, incentive £15) had two targets: (a) roughly 25-50% "
       "of eligible customers with positive EP, and (b) a **£50K budget that binds**, meaning the "
       "positive-EP customers, after channel capacity caps (calls 4,500, digital 2,500), cost more "
       "than £50K to contact."),
    code("scenarios = {\n"
         "    'Plan start: D_min £5K, incentive £15': replace(cfg.ECONOMICS, deposit_min=5_000, incentive=15),\n"
         "    'D_min £10K, incentive £15': replace(cfg.ECONOMICS, incentive=15),\n"
         "    'D_min £10K, incentive £25': replace(cfg.ECONOMICS, incentive=25),\n"
         "    'Chosen default: D_min £10K, incentive £40': cfg.ECONOMICS,\n"
         "}\n"
         "rows = {name: econ.budget_check(econ.add_economics(base, ec)) for name, ec in scenarios.items()}\n"
         "calib = pd.DataFrame(rows).T[['positive_ep', 'positive_ep_share_of_eligible', 'cost_after_channel_caps',\n"
         "                              'sum_positive_ep', 'budget_binds']]\n"
         "calib"),
    code("el = base[base['eligible'] == 1].assign(contact_cost=lambda d: econ.contact_cost(d['contact']),\n"
         "                                        channel_group=lambda d: econ.channel_of(d['contact']).map(cfg.CHANNEL_GROUP).values)\n"
         "ceiling = sum(g.nlargest(cfg.CHANNEL_CAPACITY[k], 'contact_cost')['contact_cost'].sum()\n"
         "              for k, g in el.groupby('channel_group'))\n"
         "print(f'Eligible customers: {len(el):,}  |  most the capped campaign could ever cost: £{ceiling:,.0f}')"),
    md("**The two targets conflict.** Even if *every* eligible customer had positive EP, the "
       "channel caps limit the spend to about £59K. For the budget to bind, roughly 70% or more of "
       "eligible customers must have positive EP, which is above the 25-50% target. The plan sets "
       "the binding budget as the condition that makes the optimisation meaningful, so target (b) "
       "takes priority. The chosen defaults (D_min £10,000, incentive £40) give **74% positive EP "
       "and £54.3K of capped demand against £50K**. That is binding, but only by about 9%, and the "
       "incentive stays realistic (D-031). Every value is editable in the dashboard."),

    md("## 2. Per-customer economics (sub-task 3.4) and the break-even cost (Q3)"),
    code("cols = ['deposit', 'value', 'contact_cost', 'break_even_cost', 'expected_profit', 'ep_per_cost']\n"
         "t[cols].describe(percentiles=[.1, .5, .9, .99]).round(2)"),
    code("c = t.loc[t['expected_profit'].idxmax()]\n"
         "e = cfg.ECONOMICS\n"
         "print(f\"Worked example: {c.customer_id}  balance £{c.balance:,.0f}  channel {c.channel}  P = {c.propensity:.3f}\")\n"
         "print(f\"  D  = clip({e.deposit_min:,.0f} + {e.deposit_alpha} x {max(c.balance, 0):,.0f}) = £{c.deposit:,.0f}\")\n"
         "print(f\"  V  = {c.deposit:,.0f} x {e.spread:.2%} x {e.term_years:g} - {e.incentive:g} - {e.onboarding_cost:g} = £{c.value:,.2f}\")\n"
         "print(f\"  BE = {c.propensity:.3f} x {c.value:,.2f} = £{c.break_even_cost:,.2f}   (most worth spending to contact)\")\n"
         "print(f\"  EP = {c.break_even_cost:,.2f} - {c.contact_cost:g} = £{c.expected_profit:,.2f}\")"),
    md("The break-even cost answers **Q3** directly: it is the most the bank should spend to contact "
       "a customer. Contact is worthwhile when BE_i is greater than the channel cost."),
    code("q3 = t.groupby('channel').agg(customers=('customer_id', 'size'), cost=('contact_cost', 'first'),\n"
         "                              median_break_even=('break_even_cost', 'median'),\n"
         "                              p90_break_even=('break_even_cost', lambda s: s.quantile(.9)),\n"
         "                              share_worth_contacting=('positive_ep', 'mean'))\n"
         "q3.round(2)"),

    md("## 3. Rankings: by EP and by return per £ (sub-task 3.5)\n"
       "EP ranks customers by absolute profit. EP / C ranks them by profit per £ of budget, which "
       "is the right order when the budget is the binding limit. The greedy baseline (M4) uses it."),
    code("show = ['customer_id', 'segment', 'channel', 'propensity', 'value', 'contact_cost', 'expected_profit', 'ep_per_cost']\n"
         "print('Top 8 by expected profit')\n"
         "display(t.nsmallest(8, 'ep_rank')[show].round(3))\n"
         "print('Top 8 by EP per £ of contact cost')\n"
         "display(t.nsmallest(8, 'ep_per_cost_rank')[show].round(3))\n"
         "k = 1_000\n"
         "a_, b_ = set(t.nsmallest(k, 'ep_rank').customer_id), set(t.nsmallest(k, 'ep_per_cost_rank').customer_id)\n"
         "print(f'Overlap of the two top-{k:,} lists: {len(a_ & b_) / k:.0%}  |  digital share: '\n"
         "      f\"by EP {t[t.customer_id.isin(a_)].channel.eq('digital').mean():.0%}, by EP/C {t[t.customer_id.isin(b_)].channel.eq('digital').mean():.0%}\")"),

    md("## 4. Segmentation (sub-task 3.6): the Screen 1 view\n"
       "A transparent propensity-by-value grid. Propensity is \"high\" in the top 40% of P_i, and "
       "value is \"high\" at or above the median of V_i. Every customer gets exactly one segment."),
    code("cut = seg.thresholds(t)\n"
         "counts = t['segment'].value_counts().reindex(list(cfg.SEGMENTS))\n"
         "fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=ch.BAR_BLUE, marker_cornerradius=8,\n"
         "                       hovertemplate='%{x}<br>%{y:,} customers<extra></extra>'))\n"
         "ch.style(fig, 'Customers by segment', height=400, showlegend=False)\n"
         "fig.update_yaxes(tickformat=',', rangemode='tozero')\n"
         "fig.update_layout(bargap=0.35)\n"
         "fig.show()\n"
         "print(f\"Cut-offs: propensity >= {cut['propensity']:.3f} | value >= £{cut['value']:.2f}   counts sum = {counts.sum():,}\")"),
    code("seg.segment_profile(t).round(3)"),
    code("colors = dict(zip(cfg.SEGMENTS, [ch.BLUE, ch.ORANGE, '#1baf7a', '#eda100']))\n"
         "fig = go.Figure()\n"
         "for s_ in cfg.SEGMENTS:\n"
         "    d = t[t['segment'] == s_]\n"
         "    fig.add_scattergl(x=d['propensity'], y=d['value'], mode='markers', name=f'{s_} ({len(d):,})',\n"
         "                      marker=dict(color=colors[s_], size=5, opacity=0.55),\n"
         "                      customdata=d[['customer_id', 'expected_profit']],\n"
         "                      hovertemplate='%{customdata[0]}<br>P %{x:.1%}<br>V £%{y:,.0f}<br>EP £%{customdata[1]:,.2f}<extra>' + s_ + '</extra>')\n"
         "fig.add_vline(x=cut['propensity'], line=dict(color=ch.MUTED, dash='dash', width=1.5))\n"
         "fig.add_hline(y=cut['value'], line=dict(color=ch.MUTED, dash='dash', width=1.5))\n"
         "ch.style(fig, 'Value by propensity: the four segment quadrants', height=480)\n"
         "fig.update_xaxes(title='Propensity P_i', tickformat='.0%')\n"
         "fig.update_yaxes(title='Contribution value V_i (£, log scale)', type='log')\n"
         "fig.show()"),
    md("**Overlays, not segments (D-033):** RFM scores from 1 to 5 (R from the recency bucket, F "
       "from `previous`, M from the balance quintile) and an engagement score from 0 to 100 "
       "(weights: recency 40%, frequency 30%, prior success 30%). The table shows each segment's "
       "averages."),
    code("ov = t.groupby('segment', observed=False)[['r_score', 'f_score', 'm_score', 'engagement_score']].mean()\n"
         "ov['customers_75plus'] = t.groupby('segment', observed=False)['flag_age_75plus'].sum()\n"
         "ov['ineligible'] = t.groupby('segment', observed=False)['eligible'].apply(lambda s: int((s == 0).sum()))\n"
         "ov.round(2)"),

    md("## 5. High-value opportunities (sub-task 3.7)"),
    code("top = econ.top_decile_share(t)\n"
         "print(f\"Top EP decile: {top['customers']:,} eligible customers hold £{top['top_decile_ep']:,.0f} of \"\n"
         "      f\"£{top['total_positive_ep']:,.0f} positive EP ({top['share']:.0%}); entry bar EP >= £{top['min_ep_in_top_decile']:.2f}\")\n"
         "dec = econ.ep_decile_table(t)\n"
         "fig = go.Figure(go.Bar(x=dec.index.astype(str), y=dec['share_of_positive_ep'], marker_color=ch.BAR_BLUE,\n"
         "                       marker_cornerradius=4, text=[f'{v:.0%}' if v >= 0.005 else '' for v in dec['share_of_positive_ep']],\n"
         "                       textposition='outside', textfont=dict(color=ch.INK_2),\n"
         "                       hovertemplate='EP decile %{x}<br>%{y:.1%} of positive EP<extra></extra>'))\n"
         "ch.style(fig, 'Share of total positive expected profit by EP decile (1 = highest; eligible customers)',\n"
         "         height=400, showlegend=False)\n"
         "fig.update_xaxes(title='EP decile')\n"
         "fig.update_yaxes(title='Share of positive EP', tickformat='.0%')\n"
         "fig.show()\n"
         "dec.round(3)"),
    code("t[t['eligible'] == 1].nsmallest(int(np.ceil((t['eligible'] == 1).sum() / 10)), 'ep_rank')['segment']"
         ".value_counts().reindex(list(cfg.SEGMENTS)).rename('customers in top EP decile').to_frame()"),

    md("## 6. Sensitivity (sub-task 3.8)\n"
       "Total positive EP across eligible customers when one driver changes by ±50% and everything "
       "else stays at the default. The spread changes through the FTP rate, with the customer rate "
       "held fixed."),
    code("sens = econ.sensitivity(base)\n"
         "names = {'spread': 'Funding spread', 'incentive': 'Incentive', 'contact_cost': 'Contact cost'}\n"
         "fig = go.Figure()\n"
         "for (p, d), color in zip(sens.groupby('parameter', sort=False), [ch.BLUE, ch.ORANGE, '#1baf7a']):\n"
         "    fig.add_scatter(x=d['factor'] - 1, y=d['total_positive_ep'], name=names[p], mode='lines+markers',\n"
         "                    line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=ch.SURFACE, width=2)),\n"
         "                    hovertemplate=names[p] + ' %{x:+.0%}<br>positive EP £%{y:,.0f}<extra></extra>')\n"
         "ch.style(fig, 'Total positive expected profit vs change in one driver', height=430)\n"
         "fig.update_xaxes(title='Change from default', tickformat='+.0%')\n"
         "fig.update_yaxes(title='Total positive EP (£)', tickprefix='£', tickformat=',.0f', rangemode='tozero')\n"
         "fig.show()\n"
         "pv = sens.pivot(index='change', columns='parameter', values='total_positive_ep')\n"
         "pv = pv.loc[sorted(pv.index, key=lambda s: float(s.strip('%')))]\n"
         "bind = sens.pivot(index='change', columns='parameter', values='budget_binds').loc[pv.index]\n"
         "display(pv.round(0))\n"
         "bind.rename(columns=lambda c: c + ' binds')"),
    md("The funding spread dominates. A 25% lower spread nearly halves the positive EP and the "
       "budget stops binding. Contact cost and incentive matter less. The campaign's value rests "
       "mainly on the funding-spread assumption, so it should be confirmed with Treasury before "
       "the results are used."),

    md("## 7. Save the campaign table"),
    code("pl.save_campaign_table(t)\n"
         "print(pl.CAMPAIGN_TABLE_PATH.name, t.shape)\n"
         "t[['customer_id', 'segment', 'channel', 'eligible', 'propensity', 'deposit', 'value', 'contact_cost',\n"
         "   'break_even_cost', 'expected_profit', 'ep_per_cost', 'rfm_code', 'engagement_score']].head()"),
    md("**Milestone 3 output:** `data/processed/campaign_table.parquet`. It has one row for each of "
       "the 10,000 customers, with eligibility, propensity, D, V, C, BE, EP, EP/C, both ranks, the "
       "segment, the RFM scores and the engagement score. This is the input to the genetic "
       "algorithm (M4)."),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "03_profit_engine.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
