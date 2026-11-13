"""Generate notebooks/01_data_preparation.ipynb (run, then execute with nbconvert)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("# 01 · Data preparation\n\n"
       "Milestone 1 of the profit-aware campaign optimizer. This notebook only **calls** the "
       "`pacopt` package (`data.py`, `features.py`), so all of the logic is shared with the "
       "pipeline and the Streamlit app.\n\n"
       "**Steps:** load and validate → clean categoricals → RFM-inspired features → "
       "eligibility flags → leakage guard → stratified split → 10,000-customer campaign universe."),
    code("import pandas as pd\n"
         "import plotly.express as px\n"
         "import plotly.io as pio\n"
         "from pacopt import config as cfg, data, features\n\n"
         "pio.renderers.default = 'notebook_connected'\n"
         "pd.set_option('display.width', 140)"),
    md("## 1. Load and validate the raw data\n"
       "`load_raw()` checks the 17-column schema, the 45,211 row count, that there are no nulls "
       "and the ~11.7% subscription rate."),
    code("raw = data.load_raw()\n"
         "print(raw.shape, '| SHA-256 OK:', data.sha256(cfg.RAW_CSV) == cfg.RAW_CSV_SHA256)\n"
         "raw.head()"),
    code("raw['y'].value_counts(normalize=True).rename('share').to_frame()"),
    md("## 2. Clean categorical features\n"
       "Values are stripped and lower-cased, `admin.` becomes `admin`, binaries become 0/1, and "
       "**`unknown` stays as its own category** because it carries information "
       "(e.g. no recorded phone channel)."),
    code("clean = data.clean(raw)\n"
         "{c: list(clean[c].cat.categories) for c in ['job', 'contact', 'poutcome']}"),
    md("## 3. Leakage check: why `duration`, `campaign`, `day` and `month` are excluded\n"
       "These columns describe what happened **during** the campaign. The table shows how "
       "strongly `campaign` (contacts in this campaign) relates to the outcome. Customers who "
       "say yes stop being called, so a low count is partly a *result* of subscribing."),
    code("camp = clean.assign(campaign_band=pd.cut(clean['campaign'], [0, 1, 2, 3, 5, 100],\n"
         "                                         labels=['1', '2', '3', '4-5', '6+']))\n"
         "camp.groupby('campaign_band', observed=True)['y'].agg(['mean', 'size'])"
         ".rename(columns={'mean': 'conversion'})"),
    code("month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']\n"
         "m = clean.groupby('month', observed=True)['y'].agg(['mean', 'size']).reindex(month_order)\n"
         "m.rename(columns={'mean': 'conversion', 'size': 'calls'})  "
         "# small Mar/Sep/Oct/Dec waves convert ~45-50%: calling schedule, not a customer trait"),
    code("print('Model features:', features.MODEL_FEATURES)\n"
         "features.assert_no_leakage(features.MODEL_FEATURES)\n"
         "print('Leakage guard passed - forbidden:', sorted(cfg.FORBIDDEN_FEATURES))"),
    md("## 4. RFM-inspired features and eligibility flags\n"
       "| RFM | Feature | Note |\n|---|---|---|\n"
       "| Recency | `never_contacted`, `recency_days`, `recency_bucket` | from `pdays` (-1 = never) |\n"
       "| Frequency | `frequency` = `previous` (capped at 10), `prior_success` | prior campaigns only |\n"
       "| Monetary proxy | `balance`, `balance_slog`, `balance_quintile` | a value indicator, **not revenue** |"),
    code("feats, parts = features.prepare(raw)\n"
         "feats[['customer_id','age','age_band','balance','balance_slog','balance_quintile',\n"
         "       'pdays','never_contacted','recency_bucket','frequency','prior_success','eligible']].head()"),
    md("### Conversion by previous-campaign outcome (strongest pre-campaign signal)"),
    code("po = feats.groupby('poutcome', observed=True)['y'].agg(['mean', 'size']).sort_values('mean')\n"
         "fig = px.bar(po.reset_index(), x='poutcome', y='mean', text_auto='.1%',\n"
         "             labels={'mean': 'Conversion rate', 'poutcome': 'Previous outcome'},\n"
         "             title='Conversion rate by previous campaign outcome')\n"
         "fig.update_traces(marker_color='#4F97FB', marker_cornerradius=6)\n"
         "fig.update_layout(plot_bgcolor='white', yaxis_tickformat='.0%', height=380)\n"
         "fig.show()\n"
         "po"),
    md("### Conversion by recency bucket"),
    code("rec_order = ['never', '<=90d', '91-180d', '181-365d', '>365d']\n"
         "feats.groupby('recency_bucket', observed=True)['y'].agg(['mean', 'size']).reindex(rec_order)"),
    md("### Balance distribution (monetary proxy)\n"
       "Balance is heavily right-skewed with a negative tail, so the model uses a signed log "
       "transform. Negative balances are excluded from targeting (eligibility rule)."),
    code("print(feats['balance'].describe(percentiles=[.05, .25, .5, .75, .95, .99]).round(0))\n"
         "fig = px.histogram(feats, x='balance_slog', nbins=60, color='y',\n"
         "                   labels={'balance_slog': 'sign(balance) * log(1+|balance|)', 'y': 'Subscribed'},\n"
         "                   title='Signed-log balance by outcome', barmode='overlay', opacity=0.7)\n"
         "fig.update_layout(plot_bgcolor='white', height=380)\n"
         "fig.show()"),
    code("feats.groupby('balance_quintile')['y'].agg(['mean', 'size']).rename(columns={'mean': 'conversion'})"),
    md("## 5. Eligibility (responsible-customer rules, D-007)\n"
       "Hard exclusions: credit default, negative balance, contact fatigue (`previous > 10`). "
       "Customers aged 75+ are **flagged** (capped in optimisation), not excluded."),
    code("elig_cols = ['elig_default', 'elig_negative_balance', 'elig_fatigue', 'flag_age_75plus']\n"
         "summary = pd.DataFrame({'customers': feats[elig_cols].sum(),\n"
         "                        'share': feats[elig_cols].mean().round(4)})\n"
         "summary.loc['eligible (all rules)'] = [feats['eligible'].sum(), round(feats['eligible'].mean(), 4)]\n"
         "summary"),
    md("## 6. Train/test split and the 10,000-customer campaign universe\n"
       "Stratified 70/30 split. The campaign universe is drawn **only from the test set**, so "
       "every propensity score used for decisions is out-of-sample (D-014)."),
    code("split_summary = pd.DataFrame({\n"
         "    'rows': [len(parts.train), len(parts.test), len(parts.universe)],\n"
         "    'positive_rate': [parts.train['y'].mean(), parts.test['y'].mean(), parts.universe['y'].mean()],\n"
         "    'eligible_share': [parts.train['eligible'].mean(), parts.test['eligible'].mean(),\n"
         "                       parts.universe['eligible'].mean()],\n"
         "}, index=['train', 'test', 'universe']).round(4)\n"
         "assert len(parts.universe) == 10_000\n"
         "assert set(parts.universe['customer_id']) <= set(parts.test['customer_id'])\n"
         "split_summary"),
    code("parts.universe['contact'].value_counts().rename('customers by contact channel').to_frame()"),
    md("## 7. Save processed data"),
    code("features.save_processed(feats, parts)\n"
         "sorted(p.name for p in cfg.PROCESSED_DIR.glob('*.parquet'))"),
    md("**Milestone 1 output:** a clean feature table with RFM-inspired features and eligibility "
       "flags, a leakage-guarded feature allow-list, and stratified train/test sets plus a "
       "10,000-customer out-of-sample campaign universe in `data/processed/`."),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_preparation.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
