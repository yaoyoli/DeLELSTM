# Historical execution environment

CPython 3.9.6, macOS arm64 (Apple Silicon), CPU; PyTorch 2.0.0, NumPy 1.23.5, Pandas 1.5.3. The exact 31-package versions and wheel hashes are supplied alongside this file. GPU/CUDA and other operating systems have not been validated by this release.

The three environment records are byte-identical copies of the original environment evidence. The lock's header retains its historical “Resolved Lock” label; `verification_status.json` records the later verification state. Its `model_executed: false` belongs to the dependency-installation stage, before the subsequent experiments. It is not a statement about whether the later Track A runs occurred.

No environment is installed or rebuilt during publication. The historical record explicitly says `independent_rebuild_tested: false`. Wheels, virtual environments and cached packages are excluded. The exact pinned artifacts must be obtained from their publishers or the separately retained local wheelhouse, verified against the lock, and checked for the correct Python ABI/platform before any fresh installation. This release does not distribute third-party packages or grant rights to them.

For the two new read-only tools in `tools/`, Python 3.9+ and the standard library suffice. These tools do not import PyTorch or model code.
