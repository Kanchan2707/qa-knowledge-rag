# Test Automation Standards

## Design rules
UI tests use the Page Object pattern. Tests must never use fixed sleep calls; they use explicit waits instead. Each test must be independent and must not rely on the result of another test.

## Naming and tooling
Test IDs follow the format TC-AREA-NNN, for example TC-LOGIN-014. Backend tests are written in Python with pytest. Web UI tests are written with Playwright.

## Test data in automation
Tests create their own data through factories. They must not depend on shared rows in a common database.
