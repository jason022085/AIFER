## 2024-04-22 - [TFLite TensorImage Reallocation]
**Learning:** Reallocating `TensorImage.fromBitmap(bitmap)` on every inference causes unnecessary object creation and potential GC overhead. A single `TensorImage()` instance can be reused across inferences safely, as long as it's processed on a single thread.
**Action:** Always cache and reuse `TensorImage` (using `load(bitmap)`) and `TensorBuffer` when dealing with TensorFlow Lite inference, especially in continuous flows.
