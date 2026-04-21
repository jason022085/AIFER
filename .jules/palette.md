
## 2024-05-18 - [Add Empty View to ListView]
**Learning:** For Android UI development, when a `ListView` is initially empty (e.g., waiting for an image recognition task to complete), it presents a blank space which can be confusing. Using the `emptyView` property to show a placeholder `TextView` is an effective pattern to provide guidance, especially ensuring the language matches the rest of the application.
**Action:** Always implement an `emptyView` (via `listView.emptyView = emptyView`) for dynamic `ListView`s to provide user guidance when the list is initially empty.
