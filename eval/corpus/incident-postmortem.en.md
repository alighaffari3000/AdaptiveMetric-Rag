# Postmortem: checkout latency regression, 2026-05-07

Severity: SEV-2 — Duration: 3 hours 42 minutes — Author: Reliability Engineering

## Impact

Checkout p99 latency rose from 380 ms to 6.2 seconds between 09:14 and 12:56 UTC. Approximately 41,000 checkout attempts were affected and 2,900 were abandoned. Revenue impact is estimated at EUR 74,000.

## Timeline

At 09:14 a routine deployment enabled connection pooling on the payments service. At 09:31 the first latency alert fired. At 10:05 the on-call engineer ruled out the payment provider after confirming upstream response times were normal. At 11:40 the pool exhaustion hypothesis was confirmed from thread dumps. At 12:56 the deployment was rolled back and latency recovered within ninety seconds.

## Root cause

The connection pool was sized at 20 connections per instance, inherited from a staging configuration where the instance count is four. Production runs 34 instances behind a database that accepts 800 connections. The pool did not exhaust the database; instead each instance queued requests behind its own 20-connection limit while holding the pool lock during a synchronous fraud check that averages 210 ms.

The fraud check had been moved inside the pooled section three weeks earlier without a load test.

## Contributing factors

Staging load tests use synthetic traffic with a fraud-check stub that returns in under 5 ms, so the lock contention never appeared. Alerting thresholds were tuned for error rate rather than latency, delaying detection by seventeen minutes.

## Corrective actions

Move the fraud check outside the pooled section. Size the pool from measured concurrency rather than copying staging values. Add a p99 latency alert at 800 ms. Make the fraud-check stub reproduce realistic latency in staging.
