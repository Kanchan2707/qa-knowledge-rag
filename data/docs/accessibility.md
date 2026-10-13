# Accessibility Testing

## Standard
Northwind targets WCAG 2.1 level AA. Text must have a colour contrast ratio of at least 4.5:1 against its background.

## How it is tested
Automated checks with axe run in the CI pipeline. Before every major release, a QA engineer also does a manual screen reader test using VoiceOver and NVDA.
