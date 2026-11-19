# grpcio drift investigation — 2026-11-19

User-reported grpcio 1.85.3 violates the experiment freeze 1.85.1, despite a
passing pip check. Exact pinning is retained for every distribution. No patch
compatibility exception has been introduced or claimed tested.

The pin is declared in `configs/surt_base/smoke-observed-freeze.txt` and executable
`requirements-kaggle.txt`. No separate Mini requirements or hash lock exists.
The Base fresh-install cell uses the full freeze; its existing-environment branch
only repairs NumPy/SciPy through `requirements-compat.txt`. Mini env.sh only
activates the venv and sets paths; neither runtime installs dependencies.

The legacy synthetic runbook's final install previously requested unpinned
tensorboard, which can resolve grpcio transitively. That reproducibility gap now
requests tensorboard 2.21.0 and grpcio 1.85.1 explicitly. It is a plausible source
of drift in recreated environments, not a proven account of this session.
**The command that actually installed 1.85.3 is unknown without installation
logs.** Archived 1A logs show only NumPy/SciPy repairs, no 1.85.3 install.
Generic notebook setup is separate from this isolated SURT venv.

TensorBoard requires grpcio; recipe training imports SummaryWriter even when
logging is disabled. grpcio has no direct role in SURT tensor computation, but
its installed reverse requirements must still be respected. pip check verifies
declared dependency relationships, not experiment freeze equality or GPU kernels.

`scripts/surt_base/dependencies.py` reports expected/actual versions, critical
classification, installer, location, direct-URL presence and declared reverse
dependencies. Ordinary installed metadata cannot reconstruct the package index
or installing command; unknown origin is explicitly reported. Mini cell 2 writes
this diagnostic before GPU validation; successful Mini metadata includes it.
Base preserves its critical-version set using the same error diagnostics.

Explicit `--repair-grpcio` requires an isolated Python 3.12.10 venv. It runs
pip check, resolves with the full freeze as constraints, and refuses any plan
changing anything except grpcio to 1.85.1. Only after successful resolution does
it install that wheel with --no-deps. Before/after versions, pip resolver/install
reports, post-install pip check and grpc import/version checks are recorded.
Normal validation never installs anything. Other drift still fails strictly.

Primary references inspected: https://pip.pypa.io/en/stable/cli/pip_install/ and
https://github.com/tensorflow/tensorboard/blob/master/tensorboard/pip_package/requirements.txt.
Upstream master is context, not verification of the pinned environment; actual
installed requirements and the constrained resolver plan determine compatibility.

No access to the Kaggle venv/GPU was supplied. No local package/audio/model
installation occurred. Actual grpc import, pip check, Base GPU preflight, Mini
runtime and wheel availability remain **unverified on Kaggle**. Offline fixtures
cannot establish compiled-kernel compatibility.

Local validation: 64 tests passed in the dependency-policy, Mini, Base English,
Kaggle fixture, environment and workflow suites. New tests cover strict corrected
versus drifted metadata, missing packages, installer/reverse-dependency evidence,
conflicting reverse requirements, forbidden resolver plans, resolution-before-
no-deps ordering, pip-check orchestration and persistence of failure evidence.
These are offline fixtures; subprocess pip calls and grpc imports in repair tests
are mocked. All six Mini shell cells passed syntax checks and git diff --check
passed. There was no actual corrected Kaggle environment test.
