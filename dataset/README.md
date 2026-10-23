# Dataset

The scripts read three daily Excel files from this folder:

| File | Index | Close column | Period |
|---|---|---|---|
| `dax_1day_final2.xlsx` | DAX | `cp` | 2016-09-21 to 2024-01-19 |
| `u30_1day.xlsx` | Dow Jones Industrial Average (DJIA) | `closed` | 2017-01-03 to 2024-01-19 |
| `u500_1day.xlsx` | S&P 500 | `closed` | 2017-01-03 to 2024-01-19 |

These are the files used to produce the results in `results/`. Questions
about the data can be sent to the corresponding author
(theepakorn@econ.tu.ac.th).

## Format

Only the first worksheet of each file is read; the other worksheets are not
used. Each row is one trading day.

* `date`: integer `yyyymmdd`. The DAX file uses Buddhist-era years
  (year + 543), for example `25630331` is 31 March 2020; the scripts convert
  these where calendar dates are needed.
* Close price: `cp` (DAX) or `closed` (DJIA, S&P 500).
* Further numeric source columns (daily return, direction indicator, crash
  indicator, drawdown). Each one has been checked for look-ahead and is either
  kept or excluded; the decisions and reasons are listed in `SOURCE_AUDIT` in
  `koopman_revision.py` and in `results/csv/predictor_sets/source_column_audit.csv`.
  Text columns (for example old model outputs) are ignored.

`koopman_revision.py` stops with an error if a file contains a numeric column
that is not listed in `SOURCE_AUDIT`, so that no unchecked column can enter the
predictors. To run the code on other data, add the new columns to
`SOURCE_AUDIT` with a keep or exclude decision.
