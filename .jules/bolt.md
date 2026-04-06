## 2024-04-06 - Caching TFLite Model in Android Activity
**Learning:** Loading a TFLite model from disk and allocating native memory on every inference call (`recognizeImage`) creates a severe performance bottleneck and memory churn in Android ML applications.
**Action:** When implementing ML models in Android, always hoist the model instantiation to an instance property (e.g., using Kotlin's `by lazy`) and ensure native resources are properly released by calling `.close()` in the Activity's `onDestroy()` lifecycle method.
