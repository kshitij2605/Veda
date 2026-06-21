# Licensing

Veda is **open source** and offered under a **dual-licensing** model. You can use it
for free under the open-source terms below, or buy a commercial license when those
terms don't fit your use case.

> This document is informational, **not legal advice**. Before launching, have an
> open-source attorney review the dual-license + CLA setup. Licensor / copyright holder:
> **Kshitij Singh** · commercial & licensing contact: **puku1304@gmail.com**.

---

## 1. The Veda engine — AGPL-3.0-only + commercial

Everything in this repository **except** the `niriksha/` directory (the Rust crates under
`crates/`, the server, SDKs, configs) is licensed under the **GNU Affero General Public
License, version 3** (see [`LICENSE`](LICENSE)).

You may use, study, modify, self-host, and redistribute Veda for free under the AGPL.
The key obligation: **if you run a modified version of Veda as a network service, you
must make the complete corresponding source of your modified version available to that
service's users** (AGPL §13). Unmodified self-hosting carries the normal GPL obligations
(provide source on distribution, keep notices, license derivatives under AGPL).

**A commercial license is available** if the AGPL doesn't work for you — for example if
you want to:
- embed Veda in a **proprietary/closed-source product** you distribute, or
- offer Veda **as a managed/hosted service** without releasing your modifications, or
- otherwise use it without the AGPL's copyleft and network-source obligations.

Contact **puku1304@gmail.com** for commercial terms.

## 2. Niriksha dashboard — FSL-1.1-ALv2 (Functional Source License)

The `niriksha/` directory (the dashboard UI) is licensed under the **Functional Source
License, Version 1.1, Apache-2.0 Future License** (see [`niriksha/LICENSE`](niriksha/LICENSE)).

You may use, copy, modify, and redistribute Niriksha **for any purpose except a
"Competing Use"** — i.e. you may not repackage Niriksha's UX as a product or service that
substitutes for or competes with Veda/Niriksha. Each released version **automatically
becomes Apache-2.0 two years after its release**. For competing-use terms, contact
**puku1304@gmail.com**.

*(Why a different, tighter license for the dashboard? The polished console is the part a
competitor would most want to lift, so — like MongoDB Compass, RedisInsight, and Grafana
— it's held a little tighter than the engine while staying free for normal use.)*

---

## "Can I use it for free?" — quick guide

| Your situation | Free? |
|---|---|
| Individual, hobby, research, or education | ✅ Yes |
| Internal use inside your company (you don't redistribute or offer it as a service) | ✅ Yes |
| Self-hosting Veda to power **your own** application | ✅ Yes — comply with AGPL (publish modifications you serve to users) |
| Using the Niriksha dashboard for any of the above | ✅ Yes (FSL Permitted Purpose) |
| Embedding Veda in a **proprietary product you ship** | 💵 Commercial license |
| Offering Veda **as a managed service** to others | 💵 Commercial license |
| Reselling/rehosting **Niriksha's UI** as a competing product | 💵 Commercial license |

See [`PRICING.md`](PRICING.md) for paid tiers.

---

## Contributing

To keep the dual-/commercial-licensing option open, **all contributions require a signed
Contributor License Agreement** (see [`CLA.md`](CLA.md)). Without a CLA the project cannot
legally offer your contributed code under the commercial license. The CLA can be
administered automatically via [cla-assistant.io](https://cla-assistant.io).

## Trademarks

"Veda" and "Niriksha" (and associated logos) are trademarks of Kshitij Singh. The code
licenses above grant **no** trademark rights — see [`TRADEMARK.md`](TRADEMARK.md). You can
take the code, but you can't ship it under our name.

## SPDX identifiers

- Engine: `AGPL-3.0-only`
- Dashboard: `LicenseRef-FSL-1.1-ALv2`
- Commercial: `LicenseRef-Veda-Commercial` (proprietary; by agreement)
