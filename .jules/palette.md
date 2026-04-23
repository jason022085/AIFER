## 2024-04-23 - Add empty view for dynamic ListViews
**Learning:** For dynamic ListViews in Android that are initially empty, it's crucial to provide user guidance (an empty view) rather than just a blank screen. This makes the interface more intuitive by explicitly stating what the user needs to do (e.g. take a photo or select an image from the album) before the list populates.
**Action:** Always implement an `emptyView` (via `listView.emptyView = emptyView`) for dynamic ListViews to provide explicit user instructions when the data list is initially empty.
