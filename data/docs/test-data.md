# Test Data Management

## Rules
Production data must never be copied into test environments. Any personal information has to be masked or replaced with synthetic data. Synthetic data is created by the shared data factory library.

## Accounts and resets
Test accounts use the prefix qa_ so that they are easy to find and clean up. Test environments are reset to a clean state every night at midnight Eastern.
