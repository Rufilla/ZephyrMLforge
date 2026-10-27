# TODO

1. **Add the Fashion-MNIST CNN to `docs/accelerators.md`.** The README's NPU table has
   its row. Measured on 2026-10-27 on one
   FRDM-MCXN947 with converter 3.0.0. The model is the one `apps/fashion_mnist_test`
   embeds, `iteration_0001/model.tflite` in `artifacts_20261024_172534.zip` (51,952 B);
   the converter mapped 13 operators to the NPU. Accuracy is over the same 200 images,
   spread across the 10,000-image validation split; latency is the mean of 200 runs.

   | | Host | Board CPU | Board NPU |
   |---|---|---|---|
   | Latency | — | 39.47 ms | 1.25 ms |
   | Accuracy, 200 images | 83.0% | 83.0% | 83.5% |
   | Outputs against the host | — | Identical on 200, within 5e-7 | Identical on 152; same class on 199 |
   | Tensor arena used | — | 19,108 B | 17,316 B |
   | Flash / RAM | — | 223 / 109 KB | 252 / 110 KB |

   The host scores 86.15% over all 10,000, as `apps/README.md` records. The NPU's
   outputs differ from the host's by at most 14 output steps of 1/256, & change the
   class on one image (#142), a close call on the host (0.41); the NPU's answer was the
   correct one, so its extra point is noise. No fault & no constant output.

   The model is three 3×3 convolutions (16, 32, 64 filters), each followed by 2×2 max
   pooling, then dense layers of 32 & 10. `docs/accelerators.md` describes each model in
   prose under "Convolutional models"; add a paragraph there. The deck (29 October) can
   replace "Not measured" with 1.25 ms.

   The measurement was a one-off script outside the pipeline, because `run` trains new
   models rather than scoring an existing one; it is not in the repository.
