
## 2023-10-25 - [Empty State for Dynamic Lists]
**Learning:** The Android application lacks guidance when the results list is empty initially. Implementing `emptyView` for dynamic ListViews significantly improves UX by giving users clear next steps.
**Action:** Always implement an `emptyView` (via `listView.emptyView = emptyView`) for dynamic ListViews to provide user guidance when the list is initially empty.
