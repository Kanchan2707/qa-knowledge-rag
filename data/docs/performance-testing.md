# Performance Testing

## Tools and timing
Load tests are written in k6. They run before every major release and after any change to database queries or caching.

## Targets
Under a load of 500 concurrent users, the p95 API latency must stay below 300 ms and the error rate must stay below 1%. Tests run on a staging environment that mirrors production at 50% capacity, so results are scaled when compared with production limits.
