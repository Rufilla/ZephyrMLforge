# TODO

Cleanups raised in review on 2026-10-25. None changes today's results; 1 & 2 are the
likeliest to cause a bug later.

1. **Flash & port wait written twice.** `HILRunner.run()` repeats what
   `flash_and_connect()` does, each with its own error messages, so a change to one
   misses the other. Have `run()` call `flash_and_connect()` & turn its `DeviceError`
   into an `ExecutionResult`. (`pipeline/runners/hil.py`)
2. **`npu_enabled` corrected at one call site.** `_build_agent_context` overrides it,
   but `get_constraints_summary()` still reports `execution.npu`, so
   `zephyr-ml-forge graph` shows the NPU enabled for QEMU runs & boards without one.
   Derive it in one place. (`pipeline/core/pipeline.py`, `pipeline/core/config.py`)
3. **Windows per round computed twice.** The `capture` banner repeats `record()`'s
   `math.ceil(windows_per_class / rounds)`. Add a `VibrationConfig.windows_per_round`
   property for both. (`pipeline/cli.py`, `pipeline/data/vibration.py`)
4. **Unused NPU build directory.** `_measure_on_npu` creates `iteration_NNNN_npu`
   every iteration, but builds go to the shared build directory & `run()` ignores it.
   (`pipeline/core/pipeline.py`)
5. **Slow LOAD encoding.** Inputs go as `.6f` text in LOAD lines of at most 240
   characters, one round trip each: about 13 per vibration window & 30 per
   Fashion-MNIST image, so 64 on-device samples of the latter take about 2,000. A
   shorter format, such as `%.4g` or quantized int8, would cut that by a large factor.
   (`pipeline/runners/serial_link.py`, `zephyr_app/src/main_functions.cpp`)
