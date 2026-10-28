# Credits and data

The core extends [Team 504aldo's V29 estimator](https://github.com/504aldo/whest-p2-cumulant-k3)
and its factored third-cumulant representation. The fourth-order `(3,1)`
closure credits [kai_leal_gonzalez's write-up](https://discourse.aicrowd.com/t/18246).
The recurrent corrector design follows [Susupro's public work](https://github.com/logxio/fourth-cumulant).

Most networks used to train `rgc_rnn_weights.npz` came from
[Keenan Pepper's public corpus](https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k).
As disclosed on report page 56, that corpus stated no licence, and no
separate permission was obtained beyond its public release. The corpus
files are not redistributed here.

The covariance multiplication scheme credits Khoruzhii, Gelss and Pokutta;
its MIT notice is preserved in `solution/LICENSE_flip_cpd.txt`.

The report explains the added birth feedback, quartic response, covariance-anchored
companion, learned corrections, multiplication layouts and time-aware scheduling.
It also describes training data, uncertainty and negative results. The shipped
`solution/ATTRIBUTION.txt` is the original cumulative provenance record; its
dated entries include earlier development stages. The report describes the
final package. Selected earlier results and the independent PC2 sources are
included; full training corpora, every oracle script and rejected candidate
are not. The [coverage index](REPORT_MAP.md) marks these boundaries.

Andrei Bulzan directed the work, with extensive assistance from Claude, Codex,
Argon and argon_2 for implementation, experiments, analysis and writing.
This release uses the [MIT licence](LICENSE), with the original third-party
notices retained. No retraining is required to use the supplied parameters.

The README banner reuses the report's cover figure. Its sixteen rings show
sampled mean activations from one of our generated networks, one ring per
layer and 1,024 neurons per ring. The figure's paths are unchanged.
Its lettering uses Source Serif 4, as in the report; no font files are bundled.
