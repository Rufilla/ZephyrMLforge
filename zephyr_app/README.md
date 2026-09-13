# ZephyrMLForge Zephyr application

TFLite Micro inference firmware for Zephyr RTOS. The structure mirrors the official
[TFLite Micro hello_world sample](https://github.com/zephyrproject-rtos/zephyr/tree/main/samples/modules/tflite-micro/hello_world).

This tree is the single source for the firmware. The pipeline copies it per run &
overwrites three files in its copy — `src/model.cpp`, `src/model.hpp` &
`src/app_config.h`. Nothing else differs between a pipeline build & a manual one, so a
failure seen in the pipeline reproduces with a plain `west build` here.

## Directory structure

```
zephyr_app/
├── CMakeLists.txt          # Build configuration
├── prj.conf                # Zephyr configuration (no overlay; the app is always TFLite)
└── src/
    ├── app_config.h        # Arena size, benchmark run count & interactive flag
    ├── main.c              # Minimal entry point
    ├── main_functions.h    # setup()/run_benchmark()/serve_request() declarations
    ├── main_functions.cpp  # Interpreter setup, benchmark & host request handling
    ├── model.hpp           # Model data declarations
    ├── model.cpp           # Compiled model (placeholder sine network)
    ├── output_handler.hpp  # Output handler declarations
    └── output_handler.cpp  # Metrics output for pipeline parsing
```

## Prerequisites

- A Zephyr workspace & SDK, per the
  [Zephyr Getting Started Guide](https://docs.zephyrproject.org/latest/develop/getting_started/index.html),
  working as far as the **Build the Blinky Sample** section
- The TFLite Micro module fetched into that workspace:
  ```bash
  cd "$(dirname "$ZEPHYR_BASE")"
  west config manifest.project-filter -- +tflite-micro
  west update tflite-micro
  ```
- The project environment activated from the project root: `source activate.sh`

## Build & run

Run every command from the project root, not from inside `zephyr_app/`.

Under QEMU:

```bash
west build -b qemu_cortex_m3 zephyr_app --pristine
west build -t run
```

On hardware:

```bash
west build -b frdm_mcxn947/mcxn947/cpu0 zephyr_app --pristine
west flash
minicom -D /dev/ttyACM0 -b 115200
```

Where:

- `frdm_mcxn947/mcxn947/cpu0` is the full board qualifier Zephyr 4.x requires; run
  `west boards` for the target's own spelling
- `/dev/ttyACM0` is the port observed on this machine after the MCU-LINK enumerates;
  yours may differ, & it may change after a flash

## Expected output

```
=== ZephyrMLForge inference ===
model_bytes=2488
input_count=1 output_count=1
STATUS=ready
METRICS_START
latency_ns=41200
latency_us=41
latency_ms=0.041200
arena_used=2064
num_runs=100
output_val=0.479019
METRICS_END
Inference complete
```

`STATUS=ready` is printed once the interpreter has allocated its tensors. A run that
never reaches it prints `STATUS=error` with a `reason=` field instead, & the pipeline
reports that reason rather than a timeout.

## Metrics fields

| Field | Description |
|---|---|
| `latency_ns` | Mean time for one `Invoke()` call, in nanoseconds |
| `latency_us` | The same figure in microseconds, truncated |
| `latency_ms` | The same figure in milliseconds, six decimal places |
| `arena_used` | Tensor arena bytes the interpreter claimed |
| `num_runs` | Inferences the mean was taken over |
| `output_val` | First output element of the final inference, as a sanity value |

Each `Invoke()` is timed alone & the readings are accumulated in 64 bits, so a long
benchmark cannot wrap the 32-bit cycle counter. The input is rewritten before every call
— the memory planner may overlay the input buffer with an intermediate tensor, so an
input written once is consumed by the first inference & overwritten before the second —
but that write sits outside the timed region, along with every line of console output. On
a 784-input model the quantisation arithmetic for one write is roughly 100 us, which
inside the window would be reported as inference.

Where the target has no DWT cycle counter — QEMU's Cortex-M3 is one — the timing
subsystem returns zero & the firmware falls back to the kernel cycle counter. A zero
would otherwise reach the host as an inference that took no time at all, satisfying
every latency limit.

## Host evaluation protocol

After the benchmark the firmware reads commands from the console, one per line, so the
host can measure accuracy against predictions the hardware actually produced. The
pipeline uses this when `accuracy.evaluate_on_device` is true.

| Command | Reply | Meaning |
|---|---|---|
| `PING` | `PONG` | Link check |
| `BENCH` | A METRICS block | Re-runs the latency measurement |
| `INFER 0.5` | `OUT=0.479019` | One inference; inputs comma-separated, outputs likewise |

The block printed at boot is easily missed, because the USB port is still enumerating
when it is sent; the pipeline asks for a fresh one with `BENCH` rather than racing it.

Malformed input is answered with `ERR=` & a reason (`unknown_command`, `not_ready`,
`expected_inputs=<n>`, `invoke`). Set `MLFORGE_INTERACTIVE` to 0 in `src/app_config.h`
to compile the command loop out; QEMU builds do this, as the runner has no channel to
the guest console.

## Changing the model by hand

`src/model.cpp` holds a placeholder sine network. To substitute another model:

```bash
xxd -i model.tflite > /tmp/model_data.c
```

Copy the array body into `src/model.cpp`, keeping the `g_model` & `g_model_len` names &
the `const unsigned int` type of the length, then rebuild. A mismatched length type is
a link error, not a warning.

## Troubleshooting

**`STATUS=error reason=allocate_tensors`** — the model needs more arena than
`MLFORGE_TENSOR_ARENA_SIZE` in `src/app_config.h`. Raise it, or raise
`hardware.tensor_arena_kb` in the pipeline configuration.

**`Invoke failed` or a hard fault on the first inference** — the model uses an operation
the resolver does not register. The registered set is in `setup()` in
`src/main_functions.cpp`; widening it also means widening the permitted operation list in
`pipeline/agents/base.py`, or the agent will keep proposing what the firmware rejects.

**Build errors about missing TensorFlow headers** — the TFLite Micro module is not in the
workspace. Run the `west update tflite-micro` step in the prerequisites above.

**No serial output at all** — the port may have re-enumerated under a different name after
the flash. Check `ls /dev/ttyACM*`.
