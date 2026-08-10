# Security Policy

## Scope

This is research software. It is not hardened for production use and it should not be
exposed to untrusted input in a networked service.

Areas where reports are especially welcome:

- unsafe deserialization of checkpoints or experiment records;
- dataset adapters that execute or evaluate downloaded content;
- dependency vulnerabilities affecting the default install;
- accidental disclosure of credentials, tokens, or personal data in committed artifacts.

## Supported versions

The `main` branch is the only supported version. Fixes are not backported to tags.

## Reporting a vulnerability

Please **do not** open a public issue for a suspected vulnerability.

1. Use GitHub's private vulnerability reporting on this repository
   (**Security** → **Report a vulnerability**), or
2. email `hansolo.dj@gmail.com` with the subject `SECURITY: modern-neural-networks-lab`.

Please include a description, affected files or versions, reproduction steps, and the impact
you believe it has.

## What to expect

- Acknowledgement within 7 days.
- An assessment and remediation plan within 30 days.
- Public disclosure through a GitHub Security Advisory once a fix is available, with credit
  unless you ask otherwise.

## Known accepted risks

- `torch.load` is used only for artifacts produced by this repository. Do not load
  checkpoints from untrusted sources.
- Optional reference integrations (for example TabPFN or official architecture packages)
  pull third-party code and checkpoints under their own licenses and threat models.
