# Historical acquisitions patch report

## Result

The patch supplies **10 event records and 10 acquisition edges**, all using A-grade issuer, regulatory, or official-acquirer evidence. Four local organization nodes are included because the canonical snapshot lacks the historical transaction counterparties Mellanox, NVIDIA, II-VI, and Lightwire.

| Target → acquirer | Public announcement | Effective / close | Announced headline | Closing/accounting distinction | Explicit workforce fact |
|---|---|---|---:|---|---|
| Luxtera → Cisco | 2018-12-18 | 2019-02-06 | $660M | Cash plus assumed awards; split undisclosed | Employees planned to join Cisco Optics |
| Acacia → Cisco | 2019-07-09 | 2021-03-01 | $4.5B amended | Original $2.6B / $70-share deal amended to $4.5B / $115 cash | Raj Shanmugaraj and employees stated to join Cisco Optics |
| Kotura → Mellanox | 2013-05-15 | 2013-08-15 | ~$82M cash | Subject to adjustments; final paid amount unresolved | Engineering team addition and U.S. R&D-center plan stated; no names |
| Mellanox → NVIDIA | 2019-03-11 | 2020-04-27 | $6.9B EV | $7.049B cash plus $85M assumed-award fair value; $7.134B closing consideration | No separate optical-subteam evidence |
| Aurrion → Juniper | unresolved | 2016-08-09 | $101.9M accounting consideration | $74.3M cash for remaining 82%; pre-existing interests remeasured | $55M future-service awards for continuing employees; no names |
| Elenion → Nokia | 2020-02-19 | 2020-03-25 | undisclosed | 100% ownership; individual purchase price undisclosed | No explicit team-retention claim |
| NeoPhotonics → Lumentum | 2021-11-04 | 2022-08-03 | $918M equity value | $934.4M closing accounting total, including $867.3M shareholder cash | Unvested RSUs assumed on similar vesting terms; no names |
| Finisar → II-VI | 2018-11-09 | 2019-09-24 | ~$3.2B equity value | $2.908503B closing fair value: $1.879086B cash, $987.707M shares, $41.710M awards | No named-team evidence |
| Oclaro → Lumentum | 2018-03-12 | 2018-12-10 | ~$1.8B equity value | $1.4249B closing fair value: $964.8M cash, $457.4M shares, $2.7M awards | Awards converted on similar terms; no named-team evidence |
| Lightwire → Cisco | 2012-02-24 | 2012-03-19 | ~$271M | Cash plus retention incentives; split undisclosed | Employees integrated into Transceiver Modules and Supply Chain Operations |

## Modeling judgments

- Headline values retain their issuer-defined basis. Enterprise value, announced equity value, actual cash paid, and ASC 805 purchase consideration are not treated as interchangeable.
- Finisar's acquirer is `org_ii_vi`. The later Coherent name is successor context, not the 2019 counterparty.
- An assumed or converted award supports an issuer's continuing-employee mechanism, not a claim that every employee stayed.
- Generic whole-team language is not expanded into person edges or `team_transfer` edges.
- The Cisco Luxtera closing release contains an apparent body-date typo (“2018”); Cisco page metadata and its Form 10-Q establish 2019-02-06.

## Remaining highest-value gaps

1. Retrieve the canonical SEC-hosted Kotura closing filing and final adjusted cash paid.
2. Find Aurrion's signing/public-announcement date, if one was publicly disclosed.
3. Obtain Nokia's Elenion purchase price or confirm formally that it remained undisclosed.
4. Build named post-close person edges only from company bios, filings, patents, or conference biographies; the transaction-level workforce statements are insufficient.
5. Add an explicit II-VI → Coherent successor/rebrand event in the canonical corporate-history layer rather than redirecting the Finisar acquisition edge.
