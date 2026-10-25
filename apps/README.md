<!--
  Copyright (C) 2025 Rufilla Ltd
-->

# Standalone apps

Three Zephyr applications that each run one trained model on the FRDM-MCXN947 &
print what it predicted. They need no pipeline, no host & no serial commands: flash one,
open the console, & it runs.

| App | Does | Measured on one board |
|---|---|---|
| `sine_predict` | 100 predictions of sin(x) across [-π, π], each against the true value | MSE 0.0012, worst error 0.097, 33 µs each |
| `fashion_mnist_test` | Classifies 100 Fashion-MNIST test images held in flash | 89 of 100 correct, 39.6 ms each |
| `vibration_guess` | Runs a randomly chosen motor on the ML Vibro Sens Click, 20 times, & says which it was | 20 of 20 correct (12 balanced, 8 unbalanced) |

**Before you start**
- The project environment, from the project root: `source activate.sh`
- For `vibration_guess`, an ML Vibro Sens Click in the mikroBUS socket

## Build & run

From the project root:

```bash
west build -b frdm_mcxn947/mcxn947/cpu0 -d build/sine apps/sine_predict --pristine
west flash -d build/sine
minicom -D /dev/ttyACM0 -b 115200
```

Where:

- `apps/sine_predict` is the app; `apps/fashion_mnist_test` & `apps/vibration_guess`
  build the same way
- `build/sine` is any directory, one per app, so that switching apps does not force a
  full rebuild of TFLite Micro
- `/dev/ttyACM0` is the port observed on this machine; it may differ, & it may change
  after a flash

Each app waits a second after reset before printing, while the USB serial port comes
back, then runs once & stops. Press the board's reset button to run it again.

```
=== Sine predictions ===
  1  x=-3.1416  predicted=-0.0806  sin(x)=+0.0000  error=-0.0806
...
predictions=100 mse=0.001216 max_abs_error=0.0967 mean_latency_us=33
```

## What the numbers mean

- **Sine.** The model was trained on sin(x) plus noise with a standard deviation of 0.1,
  so the pipeline scored it against noisy labels (MSE 0.0125). Here it is scored against
  sin(x) itself, which is why the figure is ten times lower. The error is largest at the
  ends of the range.
- **Fashion-MNIST.** These are the first 100 images of the test split. The pipeline
  validated on that split, so they steered early stopping but were never trained on.
  Across all 10,000 the model scored 86.15%; at 100 images the standard error is about 3
  points, so 89% agrees with it.
- **Vibration.** The motor is chosen by a xorshift generator seeded from the
  accelerometer's noise, & runs for one second before its window is taken, as in
  `zephyr-ml-forge capture`. Both motors never run together.

## Where the models come from

Each app embeds a model the pipeline trained, as `src/model.cpp`; the Fashion-MNIST app
also embeds its test images as `src/test_images.cpp`. Both are generated files &
committed, so the apps build without running the pipeline. To replace a model with
another iteration's:

```bash
python apps/export_models.py --vibration artifacts/iteration_0003/model.tflite
```

Where:

- `--sine`, `--fashion` & `--vibration` each name the `model.tflite` of the iteration
  to embed, for that app; apps not named are left as they are
- `--fashion` also rewrites the test images, which needs the Fashion-MNIST download
  Keras keeps in `~/.keras/datasets`

The apps run the model on the CPU. The NPU-compiled `model_npu.tflite` would need
`CONFIG_TFLM_NXP_NEUTRON=y` & the `NeutronGraph` operator registered in
`common/inference.cpp`; that has not been done.

## Layout

| Path | Contents |
|---|---|
| `common/inference.cpp` | TFLite Micro set-up & float input & output, with the operators the pipeline firmware registers |
| `common/tflm.conf` | Kconfig shared by all three: TFLite Micro, CMSIS-NN kernels, float printing |
| `common/app.cmake` | Adds the shared code to each app |
| `vibration_guess/CMakeLists.txt` | Takes the click's overlay, Kconfig & sensor code from `zephyr_app/`, so both read the accelerometer the same way |

## If something goes wrong

**`error: the model needs a tensor arena larger than ... bytes`** — a replacement model
needs more working memory. Raise `kTensorArenaBytes` in that app's `src/main.cpp`.

**`error: model takes ... inputs`** — the embedded model was trained for another task or
window size; export the right iteration's model.

**`SENSOR=error reason=accel_not_ready`** from `vibration_guess` — the click is not
answering on I2C. Check it is seated in the mikroBUS socket & that the ADDR SEL jumper is
in its default left position, address 0x18.
