# Test Strategy

## Test pyramid
Northwind aims for a test pyramid of roughly 70% unit tests, 20% integration tests and 10% end-to-end tests. Unit tests are cheap and fast, so most coverage should live there. End-to-end tests are slow and brittle, so they are kept for critical user journeys only.

## Test plans
Every new feature needs a written test plan before development starts. The plan lists the risks, the test levels that will cover them, and any test data that is needed. The QA engineer assigned to the feature owns the plan and reviews it with the developer.

## Release gate
A build can only be released when there are no open P1 bugs and unit test coverage on the changed code is at least 80%. The release gate is checked by the QA lead on release day.
