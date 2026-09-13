<!--
  Copyright (C) 2025 Rufilla Ltd
-->

# Neural accelerators

Some boards carry an NPU. That is a fact about the hardware, so it is declared once in
`pyproject.toml` rather than in every run configuration:

```toml
[tool.zephyr-ml-forge.boards.frdm_mcxn947]
accelerator = "neutron"
neutron_target = "mcxn94x"     # neutron_compiler --show-targets lists them
```

A run asks for it with `execution.npu: true`. Asking on a board with none is refused at
start-up, naming the boards that have one — an unlisted board is assumed to have none, so
it runs on its CPU rather than failing. Under `--simulate` the setting is ignored, because
no firmware is built.

## Installing the NXP eIQ Neutron path

Two steps, both scriptable:

```bash
cd "$(dirname "$ZEPHYR_BASE")" && west blobs fetch hal_nxp -l 'neutron/.*'
pip install --index-url https://eiq.nxp.com/repository \
            --extra-index-url https://pypi.org/simple eiq-neutron-sdk eiq_nsys
```

Where:

- `west blobs fetch` collects NXP's NPU driver & firmware, which are binary blobs & carry
  a click-through licence
- `eiq-neutron-sdk` provides `neutron_compiler`; the eIQ Toolkit GUI is **not** required,
  & the index needs no account

With both present the pipeline compiles each quantised model for the NPU before building,
recording how many operators the accelerator took in `result.json`. The firmware embeds
the compiled model & the host scores the uncompiled one, because the host interpreter
cannot resolve the accelerator's custom operator.

## Status on the FRDM-MCXN947: blocked on an NXP component version mismatch

The accelerator itself works — NXP's own `samples/boards/nxp/tflm_neutron` classifies
correctly on the board at 5.62 ms an inference. Models from this pipeline compile for it
& the firmware builds, links & reports `accelerator=neutron`, but neither converter
version tested produces a model this board will execute:

| Converter | Symptom |
|---|---|
| 3.2.2 | Operator resolves, then `Internal Neutron NPU driver error 109 in model run` |
| 3.0.0 | Microcode version matches, but `unresolved custom op: NeutronGraph` |

The driver blobs report version **3.0.0**, & the driver carries the string *"Microcode
version mismatch! Runtime (driver, firmware) should be aligned with converter"*, which is
the 3.2.2 symptom exactly. Three components have to agree — the converter, the `hal_nxp`
driver blobs & the TFLM Neutron integration — & the combination NXP currently ships does
not.

Excluded by test: the FPU setting, arena size from 8 KB to 64 KB, the console subsystem,
`k_msleep`, `InitializeTarget()`, the operator resolver contents & every Kconfig
difference against the working sample.

The CPU path is unaffected & fully working; `execution.npu` stays `false` until NXP's
blobs & converter line up.
