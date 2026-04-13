## 2024-04-06 - Caching TFLite Model in Android Activity
**Learning:** Loading a TFLite model from disk and allocating native memory on every inference call (`recognizeImage`) creates a severe performance bottleneck and memory churn in Android ML applications.
**Action:** When implementing ML models in Android, always hoist the model instantiation to an instance property (e.g., using Kotlin's `by lazy`) and ensure native resources are properly released by calling `.close()` in the Activity's `onDestroy()` lifecycle method.
## 2024-04-08 - TFLite Thread Safety
**Learning:** Closing a TFLite model from the main thread (`onDestroy()`) while an inference (`model.process()`) is actively running on a background executor causes a native crash (SIGSEGV) due to concurrent resource access.
**Action:** Always synchronize the model lifecycle by queuing the `.close()` operation on the same single-thread executor that handles inferences to guarantee sequential execution.
## 2024-04-10 - TensorImage instance caching
**Learning:** Using `TensorImage.fromBitmap(bitmap)` allocates a new TensorImage and its internal buffers on every inference, causing unnecessary garbage collection and memory churn in hot paths like continuous image recognition.
**Action:** Cache a `TensorImage` instance at the class level and use `.load(bitmap)` instead to reuse the existing memory allocation.
