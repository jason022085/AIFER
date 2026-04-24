## 2024-06-25 - [Add Empty State for ListView]
**Learning:** The application uses a dynamic `ListView` to display facial expression recognition results. When the app is initially opened, this list is completely empty, which might leave users wondering what to do. Adding an `emptyView` natively handles the display logic for a helpful prompt.
**Action:** Use `listView.emptyView = emptyView` for all dynamic list views in Android to provide an intuitive UX when the data set is initially empty or cleared.
