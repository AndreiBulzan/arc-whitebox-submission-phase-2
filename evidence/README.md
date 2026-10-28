# Evidence accompanying the report

The files here retain numerical results without hostnames, local paths,
credentials, process logs, network names or private evaluation labels.

| File | What can be checked |
| --- | --- |
| [mainline_rows.csv](mainline_rows.csv) and [mainline_history.json](mainline_history.json) | Seven complete historical mini100 runs, 700 numerical rows, from the V29 rebuild to bundle_l6; versions, dataset revisions and source/receipt hashes are recorded. |
| [effective_birth_rows.csv](effective_birth_rows.csv) and [effective_birth.json](effective_birth.json) | The matched eight-network effective-edge comparison, including its exposure history and the later failed mini100 qualification. |
| [graded_public_rows.csv](graded_public_rows.csv) | The 50 displayed public results and timing data for each of submissions #337327 and #337559. Private holdout rows are omitted. |
| [report_data/PROVENANCE.json](report_data/PROVENANCE.json) | The small numerical files used for several report figures, including moment accuracy and the error profile across layers. |
| [../research/pc2/RESULTS.json](../research/pc2/RESULTS.json) | The two independent PC2 predictors, their exact source identities and both 24-network cohorts. |

Run the lightweight arithmetic/integrity check from the repository root:

```sh
python evidence/verify.py
```

The historical CSVs preserve the full recorded floating-point scores. The
AIcrowd public-row scores were displayed with limited precision; their CSV
therefore cannot reproduce the final aggregate to its last digit. The exact
aggregate from the submission status is in the root `RESULTS.json`.

Every historical row is scored as its own MSE times its own compute multiplier,
then averaged. None of these exported cohorts had failures. The report's
other failed runs are not silently treated as passing versions of these runs.

This is a compact evidence collection, not the entire experiment archive.
It does not include every oracle injection script, fitting pipeline, rejected
candidate or Monte Carlo bank. [REPORT_MAP.md](../REPORT_MAP.md) makes those
boundaries explicit. Earlier meter versions and reused panels remain labelled;
no historical result has been recertified as a current-runtime full-bank run.
