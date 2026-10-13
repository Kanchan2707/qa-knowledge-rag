# CI Pipeline

## Stages
The pipeline runs these stages in order: lint, unit tests, integration tests, end-to-end tests, and deployment to staging. A failure in any stage stops the pipeline.

## Time limits and schedule
The unit test stage must finish in under 10 minutes. If it grows beyond that, the team splits or parallelises the suite. The full end-to-end suite runs nightly at 2 AM Eastern and also on every release branch.

## Broken builds
If the main branch build fails, the on-call QA engineer is paged. A broken main build is fixed or reverted before any new work is merged.
