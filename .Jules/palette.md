## 2024-05-20 - Adding Android Content Descriptions
**Learning:** Android accessibility requires `android:contentDescription` on ImageViews similar to alt text in HTML for TalkBack screen readers to function properly. It is best practice to define these in `strings.xml` instead of hardcoding.
**Action:** When working on Android XML layouts, always verify if interactive or informative `ImageView` elements have localized content descriptions.
