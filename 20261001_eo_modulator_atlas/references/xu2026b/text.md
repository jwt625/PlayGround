---
paper_id: xu2026b
source_url: https://doi.org/10.1364/ofc.2026.th4a.2
doi: 10.1364/ofc.2026.th4a.2
license: 
extracted_on: 2026-10-01
extraction_method: pre-extracted text copied from a local corpus; no PDF and no figure images available; figures and tables must be treated as unavailable
---

# C-band 110-GHz-Bandwidth Thin-Film Lithium Tantalate Modulator Enabling 768 (536) Gbit/s Line (Net) Data Rates

Mengyue Xu, 1, \*,† Yang Lan,1, † Di Che,2,† Jinze Shi,<sup>1</sup> and Di Liang1,\*

*<sup>1</sup>Department of Electrical and Computer Engineering, University of Michigan, Ann Arbor, MI 48109 USA <sup>2</sup> Nokia Bell Labs, Murray Hill, NJ 07974, USA*

*†Equal contributions*

*\*mengyuex@umich.edu, liangdi@umich.edu*

Abstract: We demonstrate a C-band thin-film lithium tantalate (TFLT) Mach–Zehnder modulator with *V*<sup>π</sup> of 1.35 V and >110 GHz electro-optic bandwidth, enabling record-high line (net) data rate of 768 (536) Gbit/s on the TFLT platform.

# 1. Introduction

Driven by the AI era, rapidly growing data-center traffic is pushing optical transceivers toward higher data rates, lower energy per bit, and wider bandwidth. Integrated Pockels-effect electro-optic (EO) modulators have emerged as a leading solution because they offer femtosecond-timescales response, good linearity, and large extinction ratio. Thin-film lithium niobate (TFLN) has demonstrated strong overall performance, yet practical deployment continues to face challenges such as DC bias drift and non-negligible photorefractive effects, particularly under high optical power. More recently, thin-film lithium tantalate (TFLT) has attracted increasing attention due to reduced DC drift and improved resistance to optical damage [\[1\]](#page-2-0), providing a promising path toward robust, high-performance modulation. Figure 1 summarizes recent record TFLT modulators across four key EO metrics. As shown, the best reported voltage–bandwidth metric is below 20 GHz/V 2 [\[1–](#page-2-0)[7\]](#page-2-1) and the highest demonstrated net data rate reaches 423 Gbit/s [\[1](#page-2-0)[–8\]](#page-2-2) , suggesting substantial headroom remains for the TFLT platform.

In this paper, we report the first >500 Gbit/s net data rate on a TFLT platform. The demonstrated C-band TFLT Mach–Zehnder modulator (MZM) achieves a half-wave voltage (*V*<sup>π</sup> ) of 1.35 V and a 3-dB EO bandwidth exceeding 110 GHz, corresponding to > 60 GHz/V 2 . Leveraging this device, we realize a 768 Gbit/s line rate and a 536 Gbit/s net rate using 226 Gbaud probabilistically shaped (PS)-16PAM. To the best of our knowledge, these results establish state-of-the-art voltage–bandwidth performance as well as the highest demonstrated symbol rate and net data rate on the TFLT platform.

![](_page_0_Figure_11.jpeg)

Fig. 1. Recent performance records on the TFLT platform: (a) *V*<sup>π</sup> versus 3-dB EO bandwidth [\[1](#page-2-0)[–7\]](#page-2-1), and (b) net data rate as a function of symbol rate [\[1–](#page-2-0)[8\]](#page-2-2).

# 2. C-band 110-GHz-bandwidth TFLT modulator

A fundamental trade-off exists between EO bandwidth and *V*<sup>π</sup> : longer modulators reduce the drive voltage but suffer higher RF loss and velocity mismatch, limiting bandwidth. To mitigate this trade-off, the RF loss per unit length in the traveling-wave electrodes must be minimized, where the dominant contributions typically come from conductor loss and substrate-related dielectric loss. Here, we employ a capacitance-loaded traveling-wave electrode (CL-TWE) and use 1 µm-thick gold to reduce conductor loss. In addition, we partially removed the silicon beneath the electrodes to suppress substrate-related dielectric loss. Figs. 2(a–b) show a schematic and micrograph of the fabricated MZM. Si removal is performed by patterning a series of "pocket holes" in waveguide trenches and etching through the TFLT membrane and BOX layer, followed by isotropic XeF<sup>2</sup> etching of the silicon substrate (Fig. 2(c)). Here, the silicon undercut depth of 25 µm is precisely controlled using multiple short

![](_page_1_Picture_2.jpeg)

Fig. 2. (a) Schematic of the locally silicon-removed TFLT MZM. (b) Microscope image of the fabricated device. (c,d) SEM images of the RF electrodes and the optical waveguide cross section.

etch cycles. As shown in Fig. 2(d), the fabricated optical waveguide has sharp sidewalls (76°). The measured optical propagation loss is  $\sim 0.1$  dB/cm.

We characterized the low-frequency  $V_{\pi}$  of an 18-mm-long TFLT modulator down to 1 Hz at  $\lambda = 1550$  nm to assess dynamic stability (Fig. 3(a-b)). The measured  $V_{\pi}$  remains stable across 1 Hz to 10 kHz, varying by only  $\pm 4\%$  around an average value of 1.35 V. Our device exhibits a large extinction ratio (ER) of  $\sim 42$  dB and a fiber-to-fiber insertion loss of  $\sim 12$  dB, including 2.4 dB on-chip loss. The insertion loss is dominated by fiber-to-chip coupling and can be further reduced using optimized edge couplers.

We then measured the high-frequency EO frequency response of the same device. As shown in Fig. 3 (c), the EO 3-dB bandwidth exceeds 110 GHz. This translates to a voltage–bandwidth performance above 60 GHz/V<sup>2</sup>, enabling high-baud-rate IM-DD transmission with reduced drive voltage.

![](_page_1_Figure_7.jpeg)

Fig. 3. 18-mm-long TFLT MZM: (a-b) Measured optical transmission and corresponding  $V_{\pi}$ , and (c) measured EO  $S_{21}$  response.

#### 3. 226-Gbaud Modulation

We performed high-baud-rate experiments with our TFLT modulator using 226 Gbaud PAM4, PAM6, PAM8 and PS 16-PAM data signals in a back-to-back configuration. The experimental setup is shown in Fig. 4(a). A continuous-wave (CW) laser from an external-cavity laser (ECL) was amplified by an EDFA to 23 dBm and then launched into our TFLT modulator chip. The RF drive signals generated by a digital-band-interleaving digital-to-analog converter (DBI-DAC) [9] were delivered to the TFLT chip through a 110-GHz RF probe. The modulated optical output power was  $\sim$  9 dBm and was detected by a 110-GHz photodetector. The received waveform was sampled by a 256 GS/s real-time oscilloscope. The modulated optical spectrum measured on an optical spectrum analyzer (OSA) is shown in Fig. 4(b). Notably, no linear pre-equalization was applied in the transmitter DSP (Tx-DSP) because the TFLT modulator exhibits small high-frequency roll-off. The recovered eye diagrams are shown in Fig. 4(c). The measured BER for PAM4 is below the HD-FEC threshold, and the BER for PAM6 is below the 15.32% overhead with a threshold of  $2 \times 10^{-2}$  [10].

Next, we employ PAM8 signaling and sweep the entropy of PS 16-PAM to maximize the achievable net bit rate while keeping the normalized generalized mutual information (NGMI) above the decoding threshold. We assume a concatenated FEC scheme with code rate c = 0.7436 and an NGMI threshold of 0.8105 [11]. The net bit rate is calculated as  $R_{\text{net}} = B(H - (1 - c) \log_2 M)$ , where B is the symbol rate, H is the entropy, c is the FEC code rate, and M is the modulation order. For PAM8, we achieve an NGMI of 0.8299, corresponding to a net bit rate exceeding 500 Gbit/s. As shown in Fig. 4(d), the maximum entropy meeting the NGMI threshold is H = 3.4 bits/symbol, yielding the highest net data rate of  $R_{\text{net}} = 226 \left(3.4 - (1 - 0.7436) \log_2 16\right) = 536.6$  Gbit/s.

![](_page_2_Figure_2.jpeg)

Fig. 4. (a) Experimental setup for the optical modulation. (b) Received optical spectrum. (c) Recovered eye-diagram of 226 Gbaud PAM4, PAM6 and PAM8 signal. (d) NGMI and net bit rate as a function of 226 Gbuad PS- PAM16 entropies.

# 4. Conclusion

We demonstrated a 110-GHz-bandwidth C-band TFLT MZM with CMOS-compatible drive voltage, achieving a line (net) rate of 768 (536) Gbit/s using 226 Gbaud PS-PAM16. Our TFLT modulator with good DC-stability is a promising candidate for next-generation >500 Gbit/s IM-DD short-reach links, and can potentially scale toward 2 Tb/s/lane in dual-polarization coherent formats with reduced requirements for DC bias control. In addition, its high optical power handling capability, good stability, low *V*<sup>π</sup> and insertion loss make it also attractive for applications including microwave generation, high-linearity analog photonic links, and terahertz systems.

### <span id="page-2-0"></span>References

- 1. H. Wang *et al.*, "Thin-film lithium tantalate modulator operating at high optical power," ACS Photonics 12, 5345 (2025).
- 2. C. Wang *et al.*, "Ultrabroadband thin-film lithium tantalate modulator for high-speed communications," Optica 11, 1614–1620 (2024).
- 3. J. Cai *et al.*, "Heterogeneously integrated lithium tantalate-on-silicon nitride modulators for high-speed communications," arXiv:2508.06265 (2025).
- 4. M. Niels *et al.*, "A high-speed heterogeneous lithium tantalate silicon photonics platform," Nat. Photonics 20 (2026).
- 5. L. Lu *et al.*, "A broadband lithium tantalate-on-silicon nitride heterogeneous modulator for optical and terahertz communications and radar sensing," doi:10.21203/rs.3.rs-8004021/v1 (2026).
- 6. J. Su *et al.*, "Low-loss and high-speed heterogeneous lithium tantalate-on-si3n4 modulator via micro-transfer printing," CLEO, PD104 4 (2025).
- <span id="page-2-1"></span>7. M. Lin *et al.*, "Copper damascene process-based high-performance thin-film lithium tantalate modulators," Nat. Commun. (2026).
- <span id="page-2-2"></span>8. A. Kotz *et al.*, "O-band lithium tantalate-on-insulator mach–zehnder modulators operating at line rates of 480 gbit/s," CLEO, SS183 2 (2025).
- <span id="page-2-3"></span>9. D. Che, X. Chen, C. Deakin, and G. Raybon, "2.4-tb/s single-wavelength coherent transmission enabled by 114-ghz all-electronic digital-band-interleaved dacs," ECOC (2023).
- <span id="page-2-4"></span>10. R. Nagarajan *et al.*, "Recent advances in low-power digital signal processing technologies for data center applications," J. Light. Technol. 42, 4222–4232 (2024).
- <span id="page-2-5"></span>11. J. M. Gene´ *et al.*, "Experimental demonstration of widely tunable rate/reach adaptation from 80 km to 12,000 km using probabilistic constellation shaping," OFC, M3G.3 (2020).