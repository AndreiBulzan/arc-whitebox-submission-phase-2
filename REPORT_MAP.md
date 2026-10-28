# From the report to the implementation

This index uses the PDF's printed page numbers. The release contains the exact
final predictor, two independent historical predictors and selected numerical
evidence. It is not a complete archive of every experiment described in the
research narrative.

| Report pages | Subject | Where to look | Coverage |
| --- | --- | --- | --- |
| 3–8 | Task, score and why sampling is expensive | [README](README.md), [RESULTS](RESULTS.json), [baseline figure data](evidence/report_data/baselines.json) | Final results and saved baseline comparisons. Cloud drawings are explanatory examples, not benchmark measurements. |
| 9–18 | ReLU integrals, cumulants and the Gaussian reference | [estimator.py](solution/estimator.py), its derivative/term tables and `_central_moments`; [gate data](evidence/report_data/gate.json) | Mathematical explanation and the formulas actually evaluated. Four moments do not determine a general distribution; the report's readout-accuracy claim is empirical. |
| 19–27 | Births, source legs, transport, queries and old basis | [estimator.py](solution/estimator.py), `_predict_core`, `_dslices`, `_source_mm`, `_young_mm` and the source transport helpers | Complete final inference implementation, including support choices and recurrence. |
| 28–32 | Fast multiplication, layout, cuts and the cost history | [estimator.py](solution/estimator.py), the Strassen classes and `_sc_run`; [historical rows](evidence/mainline_rows.csv), [history](evidence/mainline_history.json), [operator breakdown](evidence/report_data/stock_ops_xfin.json) | Seven full historical local banks plus final arithmetic. Earlier code snapshots are not all included. |
| 33–34, 61 | Oracles and the origin of error across layers | [moment/error figure data](evidence/report_data/physics.json) and the report's tables | Saved figure inputs and the published explanation. Not all oracle injection scripts, source states or the 160-network forcing study are included; this release does not independently reproduce every oracle percentage. |
| 35 | The fourth-diagonal model and `(3,1)` term | [estimator.py](solution/estimator.py), `_k4model_features`, `_k4model_apply`, `Q31_C2`; [data_02.npz](solution/data_02.npz) | Complete fitted inference code and parameters. Historical cross-validation jobs are not included. |
| 36 | Fourth-order births folded into a cubic query | [module_03.py](solution/module_03.py), `left_polynomials`, `LiveFold`, `ScaleLawRuleFold`; [data_04.npz](solution/data_04.npz) | Complete final implementation and weights. The fitted rule approximates a response; it is not a general exact fourth-tensor representation. |
| 37–38 | Covariance-anchored companion | [estimator.py](solution/estimator.py), `_u_comp`, `_u_comp_post` and their injection/query paths | Complete final companion. Historical rank and companion ablations are described in the report, not all repackaged as separate candidates. |
| 39–40 | Deferred quartic response, graph head and recurrent corrector | [module_07.py](solution/module_07.py), [module_rgc.py](solution/module_rgc.py), data_01/data_03 and rgc_rnn_weights | Complete inference and frozen parameters. Training code, raw corpora and fitting captures are not distributed. `module_06.py` is a retained earlier head, not the active graph-head implementation. |
| 41–42, 51, 60, 62 | Paired tests, trade-offs and rejected ideas | [evidence index](evidence/README.md), [effective-edge comparison](evidence/effective_birth.json), report tables | Selected original numerical evidence; not every rejected candidate or parameter sweep. The proposed dense pair successor on page 51 was not built and is not presented as released code. |
| 43–50, 63 | Setup, residual time and adaptive recursion | [estimator.py](solution/estimator.py), `_tac_choose`, `_tac_layer`, `_TacStrassen`; [public grade rows](evidence/graded_public_rows.csv) | Exact shipped controller, real public execution records and final timing maxima. Full local slow-clock rehearsal jobs are not included. |
| 52–53 | Independent PC2 implementation and its transfer | [PC2 source and results](research/pc2/README.md); [module_04.py](solution/module_04.py), `effective_leg`; [transfer comparison](evidence/effective_birth.json) | Two complete historical PC2 predictors and 48 original score rows. Over-budget cubature has checkpoint summaries only; see [reference_frontier.json](research/pc2/reference_frontier.json). |
| 54–59, 64 | Timeline, attribution, release and glossary | [history](evidence/mainline_history.json), [provenance](solution/ATTRIBUTION.txt), licences, [RESULTS](RESULTS.json), [SHA256SUMS](SHA256SUMS) | Source identities, original notices and final package. Attribution contains dated development notes; the report and this index describe the final release. |

The entrypoint is `solution/estimator.py`. Its later class definitions and
installation calls extend earlier implementations; reading only the first
`Estimator` definition misses the final multiplication policy and learned
modules. The supplementary predictors are separate entrypoints, not modules
loaded by the graded solution.

The original training datasets and large Monte Carlo banks are not bundled.
External public methods remain attributed and linked in the report. A claim
being discussed in the report does not mean it is an active mechanism in the
final predictor, or that this small release includes its entire experimental
history.
