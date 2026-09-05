# Information Security Policy

Version 4.2 — Effective 1 January 2026 — Owner: Security Engineering

## Access control

Access follows least privilege. Production access requires a named approver and expires automatically after twelve hours. Shared accounts are prohibited. Every privileged action is logged to an append-only audit store retained for twenty-four months.

## Authentication

Multi-factor authentication is mandatory for all staff. Hardware security keys are required for anyone with production access; time-based one-time passwords are acceptable elsewhere. Passwords must be at least fourteen characters and are checked against a breached-password corpus at set time.

## Secrets management

Secrets are stored in the managed vault and injected at runtime. Secrets must never appear in source control, container images, environment files committed to a repository, or log output. A leaked secret must be rotated within four hours of discovery.

## Vulnerability management

Critical vulnerabilities are remediated within seven days, high within thirty days, and medium within ninety days. Dependency scanning runs on every pull request and blocks merge on a critical finding.

## Incident response

Security incidents are declared by any engineer without approval. The response team assembles within thirty minutes for a confirmed breach. Customer notification for a personal data breach is sent within seventy-two hours of confirmation.

## Encryption

Data is encrypted in transit with TLS 1.3 and at rest with AES-256. Internal service-to-service traffic uses mutual TLS with certificates rotated every ninety days.

## Exceptions

Policy exceptions require written approval from the security lead, carry a maximum duration of six months, and are reviewed monthly.
