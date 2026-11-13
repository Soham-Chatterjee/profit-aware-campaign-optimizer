# Raw data

| File | Description |
|---|---|
| `bank-full.csv` | UCI Bank Marketing dataset, full version: 45,211 rows, 17 columns, `;`-separated |
| `bank-names.txt` | Original variable descriptions from the dataset authors |

- **Source:** UCI Machine Learning Repository, *Bank Marketing* (dataset id 222), https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Download:** `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`. The file is at `bank.zip` → `bank-full.csv`. To re-download, run `uv run python -c "from pacopt import data; data.download_raw(force=True)"`.
- **SHA-256** of `bank-full.csv`: `D1513EC63B385506F7CFCE9F2C5CAA9FE99E7BA4E8C3FA264B3AAF0F849ED32D`
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- **Context:** direct-marketing phone campaigns of a Portuguese bank, May 2008 to November 2010. Balances are in euros. The dashboard shows them as £ for illustration only.

This folder is read-only input. Processed outputs go to `data/processed/`, which git ignores and the pipeline regenerates.
