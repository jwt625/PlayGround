---
paper_id: valdez2026
source_url: https://doi.org/10.1364/ofc.2026.w1a.6
doi: 10.1364/ofc.2026.w1a.6
license: 
extracted_on: 2026-10-01
extraction_method: pre-extracted text copied from a local corpus; no PDF and no figure images available; figures and tables must be treated as unavailable
---

# High-Efficiency Ring-Assisted Mach-Zehnder Modulator on a Lithium Tantalate-on-Silicon Nitride Platform

Forrest Valdez,<sup>1</sup> Boris Zabelich,<sup>1</sup> Viphretuo Mere,<sup>1</sup> Pragati Aashna,<sup>1</sup> Radha Krishnan,<sup>1</sup> Camiel Op de Beeck,<sup>1</sup> Arif Rahman,<sup>2</sup> Shayan Mookherjea, <sup>2</sup> and Pieter Wuytens1,\*

*<sup>1</sup>LIGENTEC SA, EPFL Innovation Park, Batiment L, Chemin de la Dent d'Oche 1B, 1024 Ecublens VD, Switzerland*

*<sup>2</sup>University of California, San Diego, Department of Electrical and Computer Engineering, La Jolla, California 92093-0407, USA*

*\*pieter.wuytens@ligentec.com*

Abstract: We present a heterogeneously integrated SiN/LTO ring-assisted Mach–Zehnder modulator with a modulation efficiency of 1.36 Vcm, insertion loss of 1.75 dB in the Cband, and 3-dB electro-optic bandwidth over 50 GHz on a wafer-scale platform.

## 1. Introduction

The simultaneous realization of efficient high-frequency modulation and low optical loss has long been a central goal in photonic integrated circuits (PICs). Although these features have been demonstrated separately in different PIC platforms, achieving both in combination together with scalable and cost-effective fabrication remains a significant challenge. This is particularly true for material platforms with no inherent electro-optic (EO) properties, such as silicon nitride (SiN). Wafer-scale heterogeneous integration of EO materials such as lithium niobate (LN) with a SiN platform offers a promising route, enabling each material to contribute its optimal properties to the overall device performance; the SiN platform provides low-loss passive components, while the EO material offers a strong Pockels effect-based modulation capabilities. While thin-film LN has been shown to be a good EO material to integrate with SiN for hybrid-based modulators [\[1,](#page-2-0) [2\]](#page-2-1), thin-film lithium tantalate (LT) is an attractive alternative due to its similar EO coefficient, high optical damage threshold, low birefringence, and reduced DCto-low-frequency bias drift [\[3,](#page-2-2) [4\]](#page-2-3). Recent works have shown LT modulators based on etching [\[5\]](#page-2-4), microtransfer printing [\[6\]](#page-2-5), and a combination of hybrid bonding with optical-grade etching [\[7\]](#page-2-6). The heterogeneous integration of low-loss SiN with an LT layer combines the passive routing advantages of SiN with the efficient active modulation enabled by LT.

In this work, we demonstrate a high-Q ring-assisted Mach–Zehnder modulator (RAMZM) fabricated using a push-pull hybrid SiN/LT structure heterogeneously integrated at wafer-scale with a standard CMOS low-resistivity Si substrate on a commercially available PIC platform. The device exhibits a measured V<sup>π</sup> of 6.8 V, corresponding to a VπL product of 1.36 V·cm, and demonstrates a high-frequency EO response with a 3-dB bandwidth exceeding 50 GHz. Fabricated within the established dual-SiN LIGENTEC process design kit (PDK), the total insertion loss of the device is 1.75 dB. This performance demonstrates the potential of hybrid SiN/LT ring-assisted modulators for compact, low-voltage, and high-speed PICs, fabricated on a commercial multi-project wafer from the LIGENTEC platform.

#### 2. Device design, fabrication, and characterization

Figure [1a](#page-1-0) illustrates the schematic of the proposed ring-assisted RAMZM. Unlike a conventional 2×2 MZM, this design incorporates a loop connection in one arm of the interferometer. Specifically, one output of the combiner is fed back to an input of the splitter to form a ring cavity, while the remaining two ports serve as the input and output of the device. This configuration turns the device into a coupling coefficient modulator, enabling control over the light injection into the ring section [\[8,](#page-2-7)[9\]](#page-2-8). The overall transmission spectrum results from the combined response of the Mach–Zehnder interferometer (MZI) and the ring resonator, governed by the optical path difference between the MZI arms and the round-trip length of the ring. Figure [1b](#page-1-0) schematically shows the hybrid bonded duallayer SiN/LT wafer cross-section. The device incorporates an 800 nm thick silicon nitride layer, SiN1, which is employed for low-loss edge coupling, optical routing to the modulator, multimode interference couplers, the lowloss bends to form the ring cavity, as well as the optical path length difference in the asymmetric MZI structure.

![](_page_1_Figure_2.jpeg)

<span id="page-1-0"></span>Fig. 1. (a) Schematic of the RAMZM. (b) Wafer cross-section. S, signal; G, ground. (c) Image of fabricated 100 mm wafer after bonding and LTOI silicon substrate removal. (d) Stitched microscope image of the EO modulators. MZM, Mach–Zehnder modulator; RAMZM, ring-assisted Mach–Zehnder modulator; Lm, length of the active hybrid modulation section.

Dual adiabatic tapers are integrated along the MZM arms to enable efficient mode transfer from the SiN1 layer to the hybrid SiN/LT waveguide. The hybrid waveguide consists of a thinner silicon nitride layer, SiN2, with a thickness of 350 nm, which is nominally 200 nm above SiN1, and the heterogeneously integrated 300 nm thick LT slab above it with a thin amount of oxide between (see Figure [1b](#page-1-0)). For efficient light modulation in the LT, the SiN2 waveguide is adiabatically tapered such that a large fraction of the optical power resides in the LT slab, which provides the Pockels effect in the modulated region, while also maintaining single-mode propagation without interacting with the metal electrodes. The hybrid mode has a confinement factor Γ, with Γ*LT* ≈ 53% of the optical power located in the LT. The final wafer stack is based on a low-resistivity silicon substrate with a manufacturer specification of 5 - 30 Ω·cm.

A wafer-to-wafer bonding process was employed to heterogeneously incorporate the LT film with the bilayer SiN wafer, similar to our previous work [\[1\]](#page-2-0). Figure [1c](#page-1-0) is an image of the bonded wafers after the LT's silicon substrate was removed. The microscope image in Figure [1d](#page-1-0) shows the device under investigation, labeled as RAMZM, alongside a conventionally structured MZM included for comparison of modulation efficiency. Both devices share identical modulation region cross-sections and electrode lengths. The co-planar aluminum electrodes were designed using Ansys Lumerical software. The signal-ground gap is 5 µm, the signal electrode width is 10 µm, the electrode length is 2 mm, and the electrode thickness is 1 µm. The design was optimized to ensure matching between the RF phase index and the optical group index over a wide frequency range, while maintaining 50 Ω impedance matching with the transmission line, source, and load.

The measured optical transmission (fiber-to-fiber) of the fabricated RAMZM is shown in Figure [2a](#page-2-9). Resonance regions with a high extinction ratio correspond to the critical coupling regime of the ring, whereas shallow dips arise from under- or over-coupled conditions. The principle of coupling modulation is to dynamically tune the device between the undercoupled and critically coupled states, which enables high modulation depth [\[8,](#page-2-7) [9\]](#page-2-8).

An arbitrary waveform generator was used to apply a sinusoidal waveform with 20 V*pp* at 1 MHz to both the MZM and RAMZM devices. Both were wavelength-biased at quadrature and critical coupling regimes, respectively. A high-speed photodiode and an oscilloscope were then used to monitor the modulated signals. Figure [2b](#page-2-9) shows the measured responses: the standard MZM (blue curve) and the RAMZM (red curve), each with 2 mm long phase shifters. A cosine-squared fit to the MZM response (cyan curve) yields a V<sup>π</sup> of 20.9 V, corresponding to a VπL of 4.2 V·cm. In contrast, the RAMZM exhibits a significantly reduced V<sup>π</sup> of 6.7 V, corresponding to a VπL of 1.4 V·cm. The factor of 3.1 improved V<sup>π</sup> between MZM and RAMZM is related to the round-trip loss of light in the cavity, corresponding to a round-trip loss of the device of 1.0 dB (including MMI, phase shifter, propagation, and transition losses).

The high-frequency EO response of both devices are shown in Figures [2c](#page-2-9)-d using an optical spectrum analyzer (OSA) characterization method [\[1,](#page-2-0)[2\]](#page-2-1). In the case of the RAMZM, the bias wavelength was chosen near the critical coupling condition at around 1550 nm. The red fit line in [2d](#page-2-9) shows that the RAMZM follows the theoretical EO response of a standard travelling wave MZM [\[10\]](#page-2-10), without strong perturbation from the ring resonances. Although

![](_page_2_Figure_2.jpeg)

<span id="page-2-9"></span>Fig. 2. (a) Transmission spectrum near 1550 nm. (b) Normalized transmission versus applied voltage under a sinusoidal-wave scan at 1 MHz for the MZM (blue) with a cosine-squared fit (cyan) and the RAMZM (red) devices. (c)-(d) EO frequency response of the MZM (blue) and RAMZM (red) up to 50 GHz with bias wavelengths near 1550 nm.

the modulators are built upon a low-resistivity Si substrate, which has high RF propagation losses, the EO response is not strictly RF loss-limited here because the phase shifter length is short (2 mm), resulting in 3-dB bandwidths greater than 50 GHz.

# 3. Conclusions

In summary, we report a low-voltage, low-loss, and high-speed hybrid EO modulator utilizing heterogeneously integrated LT on a bilayer LPCVD SiN wafer. To overcome the low-speed limitations of the low-resistivity substrate, a ring-assisted configuration was adopted, resulting in a coupling-coefficient-based modulator with a VπL of 1.36 V·cm and a 3-dB bandwidth over 50 GHz. This hybrid SiN/LT PIC allowed the use of low-loss and long-tested LIGENTEC PDK components for edge-coupling, routing, and splitting to achieve an on-chip insertion loss of 1.75 dB. This work furthers the capabilities of the LPCVD SiN foundry-level photonic platform with high-performance and efficient modulation.

## <span id="page-2-0"></span>References

- 1. M. A. Rahman, F. Valdez, V. Mere, C. O. de Beeck, P. Wuytens, and S. Mookherjea, "High-performance hybrid lithium niobate electro-optic modulators integrated with low-loss silicon nitride waveguides on a wafer-scale silicon photonics platform," arXiv preprint arXiv:2504.00311 (2025).
- <span id="page-2-1"></span>2. F. Valdez, V. Mere, X. Wang, N. Boynton, T. A. Friedmann, S. Arterburn, C. Dallo, A. T. Pomerene, A. L. Starbuck, D. C. Trotter *et al.*, "110 Ghz, 110 mW hybrid silicon-lithium niobate mach-zehnder modulator," Sci. reports 12, 18611 (2022).
- <span id="page-2-2"></span>3. C. Wang, Z. Li, J. Riemensberger, G. Lihachev, M. Churaev, W. Kao, X. Ji, J. Zhang, T. Blesin, A. Davydova *et al.*, "Lithium tantalate photonic integrated circuits for volume manufacturing," Nature 629, 784–790 (2024).
- <span id="page-2-3"></span>4. X. Yan, Y. Liu, L. Ge, B. Zhu, J. Wu, Y. Chen, and X. Chen, "High optical damage threshold on-chip lithium tantalate microdisk resonator," Opt. Lett. 45, 4100–4103 (2020).
- <span id="page-2-4"></span>5. K. Powell, X. Li, D. Assumpcao, L. Magalhaes, N. Sinclair, and M. Lon ˜ car, "DC-stable electro-optic modulators using ˇ thin-film lithium tantalate," Opt. Express 32, 44115–44122 (2024).
- <span id="page-2-5"></span>6. M. Niels, T. Vanackere, E. Vissers, T. Zhai, P. Nenezic, J. Declercq, C. Bruynsteen, S. Niu, A. Moerman, O. Caytan *et al.*, "A high-speed heterogeneous lithium tantalate silicon photonics platform," arXiv preprint arXiv:2503.10557 (2025).
- <span id="page-2-6"></span>7. J. Cai, A. Kotz, H. Larocque, C. Wang, X. Ji, J. Zhang, D. Drayss, X. Ou, C. Koos, and T. J. Kippenberg, "Heterogeneously integrated lithium tantalate-on-silicon nitride modulators for high-speed communications," arXiv preprint arXiv:2508.06265 (2025).
- <span id="page-2-7"></span>8. W. D. Sacher and J. K. Poon, "Dynamics of microring resonator modulators," Opt. express 16, 15741–15753 (2008).
- <span id="page-2-8"></span>9. Y. Xue, R. Gan, K. Chen, G. Chen, Z. Ruan, J. Zhang, J. Liu, D. Dai, C. Guo, and L. Liu, "Breaking the bandwidth limit of a high-quality-factor ring modulator based on thin-film lithium niobate," Optica 9, 1131–1137 (2022).
- <span id="page-2-10"></span>10. G. Ghione, *Semiconductor devices for high-speed optoelectronics*, vol. 116 (Cambridge University Press Cambridge, 2009).