# Flaky Test Policy

## Definition
A flaky test is one that fails and then passes on a rerun with no code change. Flaky tests hide real failures and waste engineering time, so they are treated as defects in the test suite.

## Quarantine rule
A test is quarantined after 3 flaky failures within its last 10 runs. A quarantined test is moved out of the blocking pipeline into a separate quarantine suite. The owner must open a ticket within 2 business days of quarantine.

## Review and retries
Quarantined tests are reviewed every Friday. The CI pipeline may automatically retry a failed test at most 2 times. More retries than that are not allowed because they mask instability.
