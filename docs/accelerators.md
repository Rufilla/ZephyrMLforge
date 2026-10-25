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

`execution.npu: true`, the default, measures each model on the NPU after the CPU; the
CPU result decides the iteration & the NPU's is reported beside it, with its accuracy
scored on the board. A board with none, an unlisted board, or a machine without the
converter measures the CPU alone & says so. Turning `execution.cpu` off makes the NPU the
only target, & is refused on a board with no accelerator. Under `--simulate` or in `qemu`
mode there is no NPU pass, because no hardware firmware is built.

## Installing the NXP eIQ Neutron path

Two steps, both scriptable:

```bash
cd "$(dirname "$ZEPHYR_BASE")" && west blobs fetch hal_nxp -l 'neutron/.*'
pip install --index-url https://eiq.nxp.com/repository \
            --extra-index-url https://pypi.org/simple 'eiq-neutron-sdk==3.0.0' eiq_nsys
```

Where:

- `west blobs fetch` collects NXP's NPU driver & firmware, which are binary blobs & carry
  a click-through licence
- `eiq-neutron-sdk` provides the converter, `neutron_converter` in 3.0.0; its version
  must match the driver blobs (below). The eIQ Toolkit GUI is **not** required, & the
  index needs no account

With both present the pipeline compiles each quantised model for the NPU before building,
recording how many operators the accelerator took in `result.json`. The firmware embeds
the compiled model & the host scores the uncompiled one, because the host interpreter
cannot resolve the accelerator's custom operator.

## Status on the FRDM-MCXN947: working with converter 3.0.0

Three components have to agree: the converter, the `hal_nxp` driver blobs & the TFLM
Neutron integration. The blobs are version **3.0.0**, & so must the converter be:

| Converter | Result |
|---|---|
| 3.0.0 | Runs; a 16-8 sine MLP & a vibration CNN match the host |
| 3.0.1 | Runs a 16-8 sine MLP correctly; larger models run wrongly (below) |
| 3.2.2 | Operator resolves, then `Internal Neutron NPU driver error 109 in model run` |

The 3.2.2 failure is the driver's microcode check: it carries the string *"Microcode
version mismatch! Runtime (driver, firmware) should be aligned with converter"*. 3.0.1
passes that check, so the driver reports nothing; the pipeline instead refuses an NPU
model whose converter version, recorded in the model, differs from the driver blob's in
`hal_nxp`'s `zephyr/module.yml`. On the 3.0.0 driver, 3.0.1's models give the same
output for every input, or hard-fault the board: one with a 384-input dense layer faulted in the
scheduler after its first inference, & with the tensor arena moved, in the driver's
`arm_pad_s8` copying from `0x30068000`, past the end of SRAM. An earlier note recorded
3.0.0 failing with `unresolved custom op: NeutronGraph`; that did not recur on
2026-10-25.

Tested on 2026-10-25 with Zephyr v4.4.0-14188-gcec156efc292 & `hal_nxp` at `14d44be6e`.
A `west update` that moves the blobs to a new version needs a converter of the same
version.

Measured on one board with the pseudo agent's first model, a 16-8 MLP on the synthetic
sine task, one iteration each way:

| | NPU | CPU |
|---|---|---|
| Latency, mean of 100 runs | 13.1 us | 28.2 us |
| Tensor arena used | 692 B | 1060 B |
| Flash | 209 KB | 175 KB |

The converter mapped 3 operators to the NPU. On 41 inputs spanning the task's range the
board's outputs matched the host interpreter's to within 0.016, two steps of the output
quantisation scale, & to it exactly on most.

## Convolutional models

A vibration CNN of four `(k, 1)` convolutions, three followed by max pooling, then global
average pooling & two dense layers, compiled by converter 3.0.0, ran on the NPU in
0.52 ms against 8.40 ms on the CPU. On 36 validation windows, 12 of each class, its
outputs were identical to the host interpreter's. Compiled by 3.0.1, a similar CNN of
three convolutions answered every window with `[0.332, 0.332, 0.332]`, the softmax of
three equal scores.

Until 2026-10-25 the pipeline scored the board on the first `on_device_samples`
validation windows. A vibration validation set is ordered by class, so the first 64
were 60 'off' & 4 'balanced', & a model answering 'off' to everything scored 60 of 64,
93.75%. An earlier CNN reported here as 60 of 64 correct was very likely such a model.
The pipeline now spreads the samples across the set & fails a model that gives every
sample the same output. NPU accuracy is still scored on the board, never taken from
the host.

`PADV2`, which the converter emits for padding it leaves on the CPU, is registered in the
firmware's resolver; without it such a model fails at start-up with
`Didn't find op for builtin opcode 'PADV2'`.
