"""Generate notebooks/02_propensity_model.ipynb (then execute with nbconvert)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("# 02 · Propensity model\n\n"
       "Milestone 2. Engine 1 predicts **P_i = probability that customer *i* subscribes if contacted**, "
       "using only pre-campaign information (leakage policy D-004).\n\n"
       "- **Primary: Logistic Regression.** One-hot + scaling, `C` tuned on PR-AUC with 5-fold CV, "
       "no class weighting, then **cross-validated isotonic calibration** (D-027).\n"
       "- **Challenger: Random Forest**, isotonic-calibrated, used as a benchmark.\n\n"
       "Discrimination and calibration are measured on the full held-out test set (13,564). "
       "Precision@K is measured on the 10,000-customer campaign universe (a subset of the test set)."),
    code("import json\n"
         "import numpy as np, pandas as pd\n"
         "import plotly.graph_objects as go\n"
         "import plotly.io as pio\n"
         "from pacopt import config as cfg, features as ft, model as m, charts as ch\n\n"
         "pio.renderers.default = 'notebook_connected'\n"
         "pd.set_option('display.width', 140)\n"
         "train, test, universe = m.load_processed()\n"
         "print(f'train {train.shape}  test {test.shape}  universe {universe.shape}')\n"
         "ft.assert_no_leakage(ft.MODEL_FEATURES)\n"
         "print('Features:', ft.MODEL_FEATURES)"),
    md("## 1. Train both models"),
    code("%%time\n"
         "res = m.train_and_evaluate(train, test, universe)\n"
         "print('LR best C =', res.lr_search.best_params_['clf__C'], '| CV PR-AUC =', round(res.lr_search.best_score_, 4))"),
    code("cv = pd.DataFrame(res.lr_search.cv_results_)[['param_clf__C', 'mean_test_score', 'std_test_score']]\n"
         "cv.columns = ['C', 'cv_pr_auc', 'std']\n"
         "cv.round(4)"),
    md("## 2. Evaluation: ROC-AUC, PR-AUC, Brier, calibration, precision@K"),
    code("rows = []\n"
         "for ev in (res.eval_lr, res.eval_rf):\n"
         "    r = {'model': ev.name, 'ROC-AUC': ev.roc_auc, 'PR-AUC': ev.pr_auc, 'Brier': ev.brier,\n"
         "         'max calibration gap (pp)': ev.max_calibration_gap_pp}\n"
         "    r.update({f'precision@{k}': v for k, v in ev.precision_at_k.items()})\n"
         "    rows.append(r)\n"
         "summary = pd.DataFrame(rows).set_index('model').T\n"
         "summary.loc['budget-implied K'] = [res.eval_lr.budget_k, res.eval_rf.budget_k]\n"
         "summary.loc['base rate'] = res.eval_lr.base_rate\n"
         "summary.round(4)"),
    md("The budget-implied K is the number of contacts the £50K budget buys when customers are "
       "contacted in descending propensity at fully-loaded channel costs (£12 mobile, £10 landline, "
       "£2 digital)."),
    code("p_lr, p_rf = m.predict(res.lr, test), m.predict(res.rf, test)\n"
         "y = test['y'].to_numpy()\n"
         "c_lr, c_rf = m.curves(y, p_lr), m.curves(y, p_rf)\n\n"
         "fig = go.Figure()\n"
         "fig.add_scatter(x=c_lr['roc'].fpr, y=c_lr['roc'].tpr, name=f'Logistic Regression (AUC {res.eval_lr.roc_auc:.3f})',\n"
         "                line=dict(color=ch.BLUE, width=2))\n"
         "fig.add_scatter(x=c_rf['roc'].fpr, y=c_rf['roc'].tpr, name=f'Random Forest (AUC {res.eval_rf.roc_auc:.3f})',\n"
         "                line=dict(color=ch.ORANGE, width=2))\n"
         "ch.reference_line(fig, 0, 0, 1, 1, 'random')\n"
         "ch.style(fig, 'ROC curve: held-out test set', height=430)\n"
         "fig.update_xaxes(title='False positive rate', range=[0, 1])\n"
         "fig.update_yaxes(title='True positive rate', range=[0, 1])\n"
         "fig.show()"),
    code("fig = go.Figure()\n"
         "fig.add_scatter(x=c_lr['pr'].recall, y=c_lr['pr'].precision, name=f'Logistic Regression (AP {res.eval_lr.pr_auc:.3f})',\n"
         "                line=dict(color=ch.BLUE, width=2))\n"
         "fig.add_scatter(x=c_rf['pr'].recall, y=c_rf['pr'].precision, name=f'Random Forest (AP {res.eval_rf.pr_auc:.3f})',\n"
         "                line=dict(color=ch.ORANGE, width=2))\n"
         "base = res.eval_lr.base_rate\n"
         "ch.reference_line(fig, 0, base, 1, base, f'base rate {base:.1%}', label_at='start')\n"
         "ch.style(fig, 'Precision-recall curve: held-out test set', height=430)\n"
         "fig.update_xaxes(title='Recall', range=[0, 1])\n"
         "fig.update_yaxes(title='Precision', range=[0, 1], tickformat='.0%')\n"
         "fig.show()"),
    md("### Calibration\n"
       "P_i is multiplied by money in EP_i = P_i × V_i − C_i, so predicted probabilities must match "
       "observed rates. The acceptance criterion is a gap of at most 1.5 pp in every decile."),
    code("cal = pd.concat({'LR': m.calibration_table(y, p_lr), 'RF': m.calibration_table(y, p_rf)}, axis=1)\n"
         "fig = go.Figure()\n"
         "for name, color, t in (('Logistic Regression', ch.BLUE, cal['LR']), ('Random Forest', ch.ORANGE, cal['RF'])):\n"
         "    fig.add_scatter(x=t['predicted'], y=t['actual'], name=name, mode='lines+markers',\n"
         "                    line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=ch.SURFACE, width=2)),\n"
         "                    hovertemplate='predicted %{x:.1%}<br>actual %{y:.1%}<extra>' + name + '</extra>')\n"
         "ch.reference_line(fig, 0, 0, 0.5, 0.5, 'perfect calibration')\n"
         "ch.style(fig, 'Calibration by decile: held-out test set', height=430)\n"
         "fig.update_xaxes(title='Mean predicted probability', tickformat='.0%', range=[0, 0.5])\n"
         "fig.update_yaxes(title='Observed subscription rate', tickformat='.0%', range=[0, 0.5])\n"
         "fig.show()\n"
         "cal.xs('gap_pp', axis=1, level=1).round(2).T"),
    md("### Why the LR is calibrated (D-027)\n"
       "Uncalibrated, the LR under-predicts its top decile by about 4 pp (predicted 39.8% vs actual "
       "43.8%). That decile is dominated by customers whose previous campaign succeeded, where "
       "recency effects interact in a way a linear model can't fully express. Isotonic calibration "
       "is monotone, so it **doesn't change the ranking or the meaning of the coefficients**. It "
       "only re-maps the scores to the observed rates."),
    code("raw_lr = res.lr_search.best_estimator_\n"
         "p_raw = m.predict(raw_lr, test)\n"
         "pd.DataFrame({'uncalibrated gap (pp)': m.calibration_table(y, p_raw)['gap_pp'],\n"
         "              'calibrated gap (pp)': m.calibration_table(y, p_lr)['gap_pp']}).round(2).T"),
    md("## 3. Lift and gains: what targeting by propensity buys"),
    code("g = m.gains_table(y, p_lr)\n"
         "fig = go.Figure(go.Bar(x=g.index.astype(str), y=g['rate'], marker_color=ch.BAR_BLUE,\n"
         "                       marker_cornerradius=4, text=[f'{v:.0%}' for v in g['rate']],\n"
         "                       textposition='outside', textfont=dict(color=ch.INK_2),\n"
         "                       hovertemplate='decile %{x}<br>conversion %{y:.1%}<extra></extra>'))\n"
         "ch.reference_line(fig, -0.5, base, 9.5, base, f'base rate {base:.1%}')\n"
         "ch.style(fig, 'Observed conversion by LR propensity decile (1 = highest)', height=400, showlegend=False)\n"
         "fig.update_xaxes(title='Propensity decile')\n"
         "fig.update_yaxes(title='Conversion rate', tickformat='.0%')\n"
         "fig.show()\n"
         "g.round(3)"),
    md("## 4. Explainability: top drivers"),
    code("coef = m.lr_coefficients(res.lr_search)\n"
         "top = coef.head(16).iloc[::-1]\n"
         "fig = go.Figure(go.Bar(x=top['coef'], y=top['feature'], orientation='h',\n"
         "                       marker_color=[ch.BLUE if c > 0 else ch.RED for c in top['coef']],\n"
         "                       marker_cornerradius=3,\n"
         "                       hovertemplate='%{y}<br>coef %{x:.3f}<extra></extra>'))\n"
         "ch.style(fig, 'Logistic Regression: 16 largest coefficients (blue raises, red lowers propensity)',\n"
         "         height=520, showlegend=False)\n"
         "fig.update_xaxes(title='Coefficient (log-odds; numeric features per 1 SD)')\n"
         "fig.show()\n"
         "coef.head(16)[['feature', 'coef', 'odds_ratio']].round(3)"),
    md("Permutation importance measures the drop in PR-AUC when each **original** feature is "
       "shuffled. It works on both models, so they can be compared directly."),
    code("%%time\n"
         "pi = pd.concat({'LR': m.permutation_drivers(res.lr, test).set_index('feature')['importance'],\n"
         "                'RF': m.permutation_drivers(res.rf, test).set_index('feature')['importance']}, axis=1)\n"
         "pi = pi.sort_values('LR')\n"
         "fig = go.Figure()\n"
         "fig.add_bar(y=pi.index, x=pi['LR'], name='Logistic Regression', orientation='h', marker_color=ch.BLUE, marker_cornerradius=3)\n"
         "fig.add_bar(y=pi.index, x=pi['RF'], name='Random Forest', orientation='h', marker_color=ch.ORANGE, marker_cornerradius=3)\n"
         "ch.style(fig, 'Permutation importance (drop in PR-AUC when shuffled)', height=520)\n"
         "fig.update_layout(barmode='group', bargap=0.25, bargroupgap=0.1)\n"
         "fig.update_xaxes(title='Drop in PR-AUC')\n"
         "fig.show()\n"
         "pi.iloc[::-1].round(4)"),
    md("## 5. Model choice summary"),
    code("lr_e, rf_e = res.eval_lr, res.eval_rf\n"
         "choice = pd.DataFrame({\n"
         "    'Logistic Regression': [lr_e.roc_auc, lr_e.pr_auc, lr_e.brier, lr_e.max_calibration_gap_pp,\n"
         "                            lr_e.precision_at_k['K=1000'], 'coefficients + odds ratios', 'seconds'],\n"
         "    'Random Forest': [rf_e.roc_auc, rf_e.pr_auc, rf_e.brier, rf_e.max_calibration_gap_pp,\n"
         "                      rf_e.precision_at_k['K=1000'], 'permutation importance only', 'tens of seconds'],\n"
         "}, index=['ROC-AUC', 'PR-AUC', 'Brier', 'max calibration gap (pp)', 'precision@1000',\n"
         "          'explainability', 'training time'])\n"
         "print('Recommended:', res.recommended.upper(), '-', res.reason)\n"
         "print('Selected in config.MODEL.selected:', cfg.MODEL.selected.upper())\n"
         "choice"),
    md("**Decision:** keep **Logistic Regression** (calibrated) as the scoring model. The RF gains "
       "about 0.016 PR-AUC and 0.003 ROC-AUC, which is below the pre-agreed 0.02 PR-AUC switch "
       "threshold. The LR is now *better* calibrated (max gap 1.2 pp vs 2.0 pp), explainable "
       "through coefficients, and fast enough for interactive what-if re-runs. To switch, set "
       "`MODEL.selected = 'rf'` in `config.py`."),
    md("### Tail guard: propensity cap (D-028)\n"
       "Isotonic calibration is a step function. In the extreme tail it is fitted on very few "
       "customers and can give scores like 98%. The table checks this on the test set. The "
       "propensity passed to the profit and optimisation engines is capped at "
       "`MODEL.propensity_cap` (90%). The raw calibrated score is kept in `p_lr`."),
    code("d = pd.DataFrame({'p': p_lr, 'y': y})\n"
         "bands = pd.cut(d['p'], [0, 0.5, 0.7, 0.9, 1.0], include_lowest=True)\n"
         "d.groupby(bands, observed=True).agg(customers=('y', 'size'), mean_predicted=('p', 'mean'),\n"
         "                                    actual_rate=('y', 'mean')).round(3)"),
    md("## 6. Score the campaign universe and save artifacts"),
    code("m.save(res)\n"
         "s = res.scored\n"
         "print(s.shape, '| mean propensity', round(s['propensity'].mean(), 4), '| actual rate', round(universe['y'].mean(), 4))\n"
         "s.sort_values('propensity_rank').head(10)"),
    code("print(json.dumps(json.loads(m.METRICS_PATH.read_text()), indent=1)[:1500])"),
    md("**Milestone 2 output:** `artifacts/model.joblib` (calibrated LR, explainable LR, calibrated RF), "
       "`artifacts/metrics.json`, and `data/processed/universe_scored.parquet` with a propensity, rank "
       "and decile for each of the 10,000 customers."),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_propensity_model.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
