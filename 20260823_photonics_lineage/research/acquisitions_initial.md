# Phase-1 Acquisition-First Research

As of 2026-08-23. This is a human-readable companion to `acquisitions_initial.yaml`, which contains the auditable records, source URLs, evidence grades, confidence, and unresolved fields.

## Executive summary

| Target | Acquirer | Announced | Closed | Value captured | Core lineage signal | Confidence |
|---|---|---:|---:|---:|---|---|
| InfiniLink | GlobalFoundries | 2025-11-25* | 2025-11-14 | $48M | Cairo SerDes/optical-chip design capability; founders Ahmed Aboul-Ella and Botros George | High |
| Hyperlume | Credo | 2025-09-29 | 2025-09-29 | $92.017M accounting consideration | MicroLED links; Mohsen Asad and Hossein Fariborzi | High |
| Cloud Light | Lumentum | 2023-10-30 | 2023-11-07 | ~$750M announced | High-volume 400G/800G modules and automated packaging; Dennis Tong | High |
| AMF | GlobalFoundries | 2025-11-17 | 2025-11-18 | $453M | A*STAR/IME spin-off; Guo-Qiang Lo; SiPh foundry manufacturing | High |
| Inphi | Marvell | 2020-10-29 | 2021-04-20 | $9.918B accounting consideration | Loi Nguyen/Cornell high-speed device lineage; electro-optics platform | High |
| Celestial AI | Marvell | 2025-12-02 | 2026-02-02 | $3.25B upfront announced + up to $2.25B earnout | Photonic Fabric for AI scale-up; David Lazovsky | High |
| DustPhotonics | Credo | 2026-04-13 | 2026-05-28 | Up to ~$1.2B; $770M cash + shares at close | Israeli SiPho PIC team; Yoel Chetrit, Ben Rubovitch, Avigdor Willenz | High |
| Nubis | Ciena | 2025-09-22 | 2025-10-07 | ~$270.5M aggregate | Bell Labs founders Peter Winzer and Guilhem de Valicourt; CPO/NPO + ACC | High |
| Enosemi | AMD | 2025-05-28 | Exact date undisclosed | Undisclosed | Luminous IP plus Elenion/Nokia, Ayar and A*STAR/IME personnel lineage | High |
| Polariton | Marvell | 2026-04-22 | Completed by announcement | Undisclosed | Formal ETH Zurich/Jürg Leuthold spin-off; plasmonic modulators | High |

\* GF's first dated public item found post-dates InfiniLink's filing-established close; the precise first public announcement date remains unresolved.

## Most important genealogy findings

1. **A*STAR/IME → AMF → GlobalFoundries is a formal institutional spin-off chain.** A*STAR identifies AMF as an IME silicon-photonics spin-off. Guo-Qiang Lo's conference biography places him at IME from 2004–2017 with silicon-photonics program responsibility before co-founding AMF.

2. **Bell Labs → Nubis → Ciena is unusually clean person-level evidence.** Ciena's transaction deck identifies Peter Winzer (20 years at Bell Labs) and Guilhem de Valicourt (9 years at Bell Labs) as Nubis founders. Post-close, Winzer continued at Ciena as VP Systems Architecture.

3. **ETH Zurich / Jürg Leuthold → Polariton → Marvell is another formal academic spin-off chain.** ETH states that Claudia Hoessbacher, Wolfgang Heni, and Benedikt Baeuerle founded Polariton in 2019 out of Leuthold's Institute of Electromagnetic Fields.

4. **Enosemi is a recombination node, not merely a standalone startup.** Its own launch release ties the team to Luminous Computing IP, Elenion/Nokia product experience, early Ayar Labs, and Gennum/Semtech. Matt Streshinsky additionally documents University of Washington and A*STAR/IME training/work.

5. **Inphi remains the largest and most mature platform acquisition in this set.** Marvell recorded $9.918B total consideration. Founder Loi Nguyen continued into Marvell leadership; CEO Ford Tamer joined Marvell's board. The clean upstream thread is Cornell/Lester Eastman and Honeywell for Nguyen, with Broadcom and Agere/Lucent for Tamer.

## Transaction-value cautions

- **Celestial AI:** $3.25B is the signing-date upfront value ($1B cash + shares valued at $2.25B), plus an announced earnout of up to $2.25B. Actual close used about $1.3B gross cash ($1B net of acquired cash) and 24.5M shares, versus 27.2M shares announced.
- **DustPhotonics:** terms evolved between signing and closing. The signing release said $750M cash + ~0.92M shares and up to ~3.21M earnout shares. Credo later reported $770M cash + ~0.8M shares and potential ~2.8M shares + $31.6M cash. A seller-investor filing characterized total consideration as about $1.2B.
- **Nubis:** the $270.5M aggregate includes $37.9M of future employment-linked arrangements that Ciena accounts for as post-combination compensation; cash purchase price at close was $232.6M.
- **Cloud Light:** ~$750M is the announced transaction value before adjustments, paid in cash plus assumption/substitution of unvested options.

## Explicit source gaps

- No disclosed values were found for Enosemi or Polariton.
- No exact legal close date was found for Enosemi; AMD only said on May 28, 2025 that it had welcomed the team the prior week.
- InfiniLink's exact first public announcement date remains unclear because GF's dated public article followed the November 14 close.
- Acquired employee counts are missing for InfiniLink, Hyperlume, Cloud Light, AMF, Inphi, Celestial AI, Enosemi, and Polariton. Nubis disclosed 50+ engineers; DustPhotonics disclosed approximately 70 employees.
- Founder upstream histories remain incomplete for InfiniLink, Hyperlume, Cloud Light, DustPhotonics, and parts of Celestial AI. These are left unresolved rather than filled from weak aggregators.
- Subsequent founders emerging from each acquired team need a separate people-first expansion pass.

## Recommended next pass

1. Expand person records from patents, conference bios, archived team pages and pre-acquisition papers, especially for Cloud Light, InfiniLink and DustPhotonics.
2. Resolve corporate genealogy edges for Enosemi's licensed Luminous IP versus personnel transfer.
3. Reconcile final purchase accounting for Celestial AI and DustPhotonics after later annual reports.
4. Build a normalized person/organization/edge export from the evidence captured here, preserving the distinction between formal spin-out, employment lineage, and IP license.
