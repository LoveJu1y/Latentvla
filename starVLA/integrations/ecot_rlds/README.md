# ECOT RLDS Integration (Skeleton)

This directory contains the scaffolding for integrating Prismatic/ECoT RLDS
datasets into StarVLA. The current placeholder files establish the module
structure without implementing the full data pipeline yet.

Implementation will proceed in incremental steps:

1. Wire the dataset adapter to the Prismatic RLDS pipeline.
2. Map RLDS batches to StarVLA sample dictionaries.
3. Expose configuration validation and DataLoader builders.
4. Add smoke and contract tests to ensure compatibility.

Refer to `docs/ecot_rlds_implementation_plan.md` for the detailed roadmap.

