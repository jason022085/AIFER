## 2024-04-19 - [TensorImage Allocation]
**Learning:** Reusing a single `TensorImage` instance using `.load(bitmap)` instead of `.fromBitmap(bitmap)` avoids repeated object allocations during inference.
**Action:** Use `.load(bitmap)` for recurring inference tasks to reduce memory churn.
