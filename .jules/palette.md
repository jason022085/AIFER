## 2026-04-20 - Empty View for ListView
**Learning:** The AIFER app uses native Android `ListView`s to display recognition results. Since results only appear after a photo is processed, the list is initially empty, showing just blank space.
**Action:** Always implement `listView.emptyView = emptyView` with a descriptive localized message (in Traditional Chinese) to provide initial user guidance for dynamic lists.
