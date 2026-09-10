# Architecture

The ranking service is written in Python and deployed on Kubernetes.

It depends on the Waypoint SDK version 0.3.1 for ETA prediction, and on Postgres 16 for
the load and carrier tables. Planning runs are idempotent: replaying a load id returns the
same ranking unless carrier availability changed.

The planning path holds a p95 latency of 380 ms, measured at the ranking endpoint.
