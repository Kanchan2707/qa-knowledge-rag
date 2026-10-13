# Release Process

## Schedule
Northwind ships a release every second Tuesday. Code freeze starts the Monday before at 5 PM Eastern. Only fixes approved by the QA lead may be merged after the freeze.

## Regression and sign-off
The regression suite contains 240 tests and all of them must pass before release. Release sign-off needs approval from both the QA lead and the product owner.

## Rollback
If a P1 bug appears after release, the team must decide within 30 minutes whether to roll back. The on-call engineer makes the call together with the QA lead.
