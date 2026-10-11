# Research: 400G QSFP112 FR4 (candidate hero A, not chosen for short 1; kept for a later short)

Date: 2026-10-09. Method: research subagent, read-only. Not fetched: full QSFP112 MSA PDF (too large), Broadcom/Marvell briefs (403), Yole sample (403). "via search" = read from a search summary.

## Existence and specs

| Claim | Value | Source | Conf |
|---|---|---|---|
| Major vendor product | Cisco QSFP-400G-FR4: 400G QSFP112, 400GBASE-FR4, duplex LC SMF, 2 km | cisco.com/c/en/us/products/collateral/interfaces-modules/400g-qsfp-transceiver-modules-ds.html | high |
| Other datasheets | Edgeoptic 400G-QSFP112-2.1; Naddod Q112-400G-FR4; ProLabs QSFP112-400GB-FR4-C | edgeoptic.com; resource.naddod.com/files/2026-08-11/q112-400g-fr4-datasheet-017571.pdf | high |
| Wavelengths | 1271/1291/1311/1331 nm (1264.5-1337.5 nm) | Edgeoptic | high |
| Power | max 9 W (Cisco), max 10 W (Edgeoptic, Naddod) | as above | high |
| Host | 400GAUI-4 C2M (802.3ck), 4x106.25G PAM4, host FEC | Naddod, Cisco | high |
| CMIS | 5.0 (datasheets); MSA Rev 2.1 references 5.1 | Edgeoptic, MSA | high |

## Optics

| Claim | Value | Source | Conf |
|---|---|---|---|
| Laser | DML parts and EML parts both exist | Edgeoptic; lightwaveonline.com 14200566 | high |
| TEC | Uncooled common on CWDM grid (e.g. NeoPhotonics uncooled CWDM4 EML, 20-70 C) | semiconductor-today.com 2018-09 | med |
| TX MUX | TFF Z-block common (Coherent CWDM4 Z-blocks) | coherent.com Z-block datasheet | med |
| TX train (patent) | prism + lens array -> 4-ch Z-block -> isolator -> fiber collimator; 0.75 mm laser pitch | USPTO 12212364 | med |
| Monitor PD | not found for FR4 | - | low |
| Driver | separate quad driver (DML) or DSP-integrated (Broadcom BCM8741x, MaxLinear MxL93642) | - | med |
| RX PD | PIN (InGaAs/InP) | Edgeoptic; Hengtong; Yole (via search) | high |
| RX DEMUX + fold | TFF demux, or AWG with 40-42 deg polished facet as mirror onto PD array | m.researching.cn/articles/OJeedfa381d34b396c | med |
| TIA | 4-ch TIA (seen in 100G CWDM4 teardown; assumed for FR4) | outside5sigma.com/teardown/100G-200G-400G-800G/ | med |

## DSP examples

Broadcom BCM8741x "Jesko" (2021-11-08; integrated TIA + driver; 7 W 400G DR4/FR4 claim); Broadcom BCM87400 (2018, 7 nm, 4x112G); MaxLinear MxL93642/3 Keystone; Credo Robin 400 (4:4 retimer for QSFP112 FR4; Robin 402 with EML driver); Marvell Spica Gen2 (8x100G, 5 nm, integrated EML/SiPh drivers, sub-12 W 800G).

## Management, mechanics, thermal (QSFP112 MSA Rev 2.1, 2022-08-24)

- MCU monitors supply, temperature, laser bias (networking-docs.nvidia.com/mms4x50nm800g2kmpub/pin+description). DDM: temperature, Vcc, per-lane TX bias, TX power, RX power.
- 2-wire management, address A0h, 400 kHz (1 MHz optional); ID byte 1Eh = CMIS.
- 38 contacts; 4 TX + 4 RX diff pairs up to 112 Gb/s.
- Low-speed: ModSelL, ResetL, LPMode/TxDis, ModPrsL, IntL/RxLOSL, SCL, SDA.
- Mating: 1st GND, 2nd power (VccRx, Vcc1, VccTx), 3rd signal (MSA Table 3).
- Power class 8 (>5 W). Case 0-70 C. Heat path: case to cage-riding heatsink; thermal gel on DSP and TOSA; TOSA hotter than ROSA (USPTO 11415764).

## Comparison

- 100G QSFP28 CWDM4: 4x25G, cooled DML TOSA + PIN ROSA (one vendor), CDR not DSP, 3.5 W max.
- 800G OSFP DR8: 8x100G PAM4, MPO-16 APC, 500 m, max 16-16.5 W; SiPh or EML (see `hero_800g_osfp_dr8.md`).

## Teardown references

Own: outside5sigma.com/teardown/100G-200G-400G-800G/ and outside5sigma.com/ofs/weekly-OFS-44/ (100G CWDM4, 400G DR4, 200G QSFP56 FR4, 800G PSM8). ServeTheHome 100G SR4 teardown. No free 400G FR4 internal photo found.
