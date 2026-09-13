# Program Flow

This document captures the full ZephyrMLForge iteration flow.

```mermaid
flowchart TD
  A[Start] --> B[Load config.yaml]
  B --> C[Generate synthetic data]
  C --> D[AI proposes model]
  D --> E[Train model]
  E --> F["Reuse prior weights - optional"]
  F --> G["Convert to TFLite - int8"]
  G --> H[Build Zephyr app]
  H --> I["Run inference - QEMU/HIL/Sim/Auto"]
  I --> J[Collect results]
  J --> K[Record metrics]
  K --> L{Meets constraints and targets?}
  L -- no --> M[Reject or fail]
  L -- yes --> N[Record best iteration]
  M --> D
  N --> D
  N --> O{Stop conditions met?}
  O -- yes --> P[Generate graphs]
  O -- no --> D
  P --> Q[Complete]
```

Notes:
- Synthetic data is generated once per run, then reused across iterations.
- Constraints include accuracy, latency, RAM, and flash limits.
- The pipeline stops when targets are met or max iterations are reached.
