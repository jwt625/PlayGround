---
paper_id: li2026a
source_url: https://doi.org/10.1364/ofc.2026.w1a.5
doi: 10.1364/ofc.2026.w1a.5
license: 
extracted_on: 2026-10-01
extraction_method: pre-extracted text copied from a local corpus; no PDF and no figure images available; figures and tables must be treated as unavailable
---

# **Lithium-Tantalate-on-Fused Silica Mach-Zehnder Modulators**

**Zihan Li1,2\* , Alexander Kotz3\* , Adrian Schwarzenberger3 , Christian Koos3,4,5† , Tobias Kippenberg1,2,5‡**

*1 Institute of Electrical and Micro Engineering, EPFL, CH-1015 Lausanne, Switzerland <sup>2</sup> Institute of Physics, Swiss Federal Institute of Technology Lausanne (EPFL), CH-1015 Lausanne, Switzerland <sup>3</sup> Inst. of Photon. and Quant. Electron. (IPQ), Karlsruhe Inst. of Technol. (KIT), 76131 Karlsruhe, Germany 4 Inst. of Microstruct. Technol. (IMT), Karlsruhe Inst. of Technol. (KIT), 76131 Karlsruhe, Germany <sup>5</sup> LUXTELLIGENCE SA, CH-Saint Sulpice, Switzerland* \* *These authors contribute equally. † [christian.koos@kit.edu;](mailto:christian.koos@kit.edu) ‡[tobias.kippenberg@epfl.ch;](mailto:tobias.kippenberg@epfl.ch)* 

**Abstract:** We demonstrate the lithium-tantalate-on-fused silica (LT-on-FS) Mach-Zehnder modulator manufactured in 4-inch wafer-scale. The modulator features a 3 dB electro-optic bandwidth of 67 GHz and a modulation efficiency of 1.6 V. We achieve net data rates of 437 Gbit/s with PAM8 signaling. © 2025 The Author(s)

## **1. Introduction**

Lithium tantalate (LT), an emerging electro-optical platform, exhibits promising improvements compared to the current widely explored material, lithium niobate [1~4]. Its superior electro-optical coefficients make it a compelling candidate for modulators, which empower modern communication networks, data centers, and various academic research applications. Concurrently, the more mature fabrication process in foundries potentially unlocks the cost limit for more widespread applications [1]. Nonetheless, monolithic LT electro-optical modulators were demonstrated with a silicon substrate, which limited the microwave waveguide design for a broader electro-optic (EO) modulation bandwidth [2~4].

![](_page_0_Figure_8.jpeg)

Fig. 1. Wafer-scale fabricated LTOFS MZM. (a) The microscope picture of three 1.8 cm long LTOFS MZM. (b) The photo for the fabricated LTon-FS wafer. (c)-(f) Scanning electron microscope pictures for the electrodes, cross-section, and chip facets. LT is colored in blue, SiO2 in purple, and Au in yellow. (g) The fabrication process flow includes optical waveguide etching and the electrode lift-off method.

In this paper, we extend our previous research [3] by replacing the silicon substrate with a fused silica (FS) wafer. In contrast to our earlier efforts, the current device employs a wafer-scale fabrication process and achieves the state-ofthe-art performance of integrated EO modulators [5].

## **2. Lithium-Tantalate-on-Insulator (LTOI) Mach-Zehnder Modulators (MZM)**

The modulator utilizes a 2 µm wide, 400 nm high waveguide with a 200 nm slab to guide the light in the modulation arms. A co-planar microwave waveguide with a T-segment structure is placed directly on the LT slab, serving as the push-pull configuration electrodes (Fig. 1 (c)-(d)). A 1.5 µm space between the metal structure and optical waveguide can improve the modulation efficiency while not significantly degrading the optical loss. By carefully selecting the height of the T-segment [5], the microwave effective index is matched to the optical group index, which enables broad operation bandwidth (Fig. 2 (c)).

The fabrication begins with a fused silica wafer and an X-Cut LTOI wafer, consisting of a 600 nm LT thin-film single crystal on 4.7 µm of thermal silicon dioxide, and a 525 µm silicon carrier (Fig. 1 (g)). To prepare the LT-on-FS substrate, we transfer the LT thin-film to the fused silica wafer by direct wafer bonding with oxygen plasma surface activation. The following fabrication process for the modulators is reported in our previous work [1]. Additionally, the electrodes are also cladded with a thin layer of SiO2 to protect the soft metal surface, while the pad area is opened for the microwave probe connection. Finally, the wafer is singulated using stealth dicing with a femtosecond laser (General Intelligent Equipment Co., Ltd.) to form a smooth facet for butt coupling (Fig. 1 (e)-(f)).

To evaluate the electro-optical performance in wafer-scale, we measured the half-wave voltage-length product (VπL) of an 18-mm long modulator by applying a low frequency drive signal from an arbitrary waveform generator and recording the optical output intensity with a photodetector. With a 100 Hz driven frequency, a half-wave voltagelength product (VπL) of around 2.8 V·cm for 1550 nm and 2.3 V·cm for 1300 nm is measured on the same device (Fig. 2 (a)), which corresponds to 1.56 V and 1.28 V half-wave voltage for an 18-mm long modulator, respectively. Meanwhile, we characterize the modulation efficiency with 100 Hz driven frequency across the whole wafer, which shows consistent VπL for both C-band and O-band (Fig. 2 (b)). The microwave transmission characteristics are measured using a vector network analyzer and 110 GHz GSG probes (Fig. 2 (c)). The electro-optical response of the modulator is further performed by terminating one probe with a 50 Ω resistor and collecting the optical signal with a calibrated photodiode. The MZM features a 3-dB bandwidth of 67 GHz (Fig. 2 (d)).

![](_page_1_Figure_7.jpeg)

Fig. 2. Electro-optic performance and IMDD signaling experiment using an LT-on-FS modulator. (a) The dependence between VπL and the driven signal in the low frequency range. (b) The distribution of VπL on a 4-inch wafer. (c) Microwave loss and effective index of the CPW. (d) Measurement of electro-optic response for an 18-mm long modulator. (e) Experimental setup for IMDD signaling experiment: Tx-DSP: transmitter-side digital signal processing, AWG: arbitrary waveform generator, ECDL: external-cavity diode laser, FPC, fiber-based polarization controller, EDFA: erbium-doped fiber amplifier, PD: photo diode, RTO: real-time oscilloscope, Rx-DSP: receiver-side DSP. (f) Measured bit error ratios (BER) and (g) achievable information rates (AIR, dashed lines, asterisks), as well as net data rates (NDR, solid lines, circles) as a function of symbol rates for PAM8 (green), PAM6 (orange), and PAM4 (blue) signals.

## **3. High-speed IMDD signaling using LTOI MZM**

To demonstrate the exceptional performance of our LTOI MZM, we perform a high-speed IMDD signaling experiment using the setup shown in Fig. 2(e). An external-cavity diode laser (ECDL, 17.8 dBm @ 1550 nm) provides the optical carrier. The electrical drive signals are generated by a high-speed arbitrary-waveform generator (AWG, M8199B, Keysight) and fed to the CPW of the MZM via a 20 cm-long RF cable, a broadband RF amplifier, and a 110 GHz RF probe. Pulse-amplitude modulated (PAM) signals with four, six and eight levels (PAM4, PAM6, PAM8) based on pseudo-random bit sequences are synthesized by using offline digital signal processing (Tx-DSP) comprising root-raised cosine pulse-shaping (roll-off β = 0.05), predistortion to account for the frequency-dependence of the electrical transmitter (excluding the modulator) and clipping of the signal to reduces the peak-to-average power ratio to 9 dB. The MZM is terminated by a 50 Ω resistor via a second 110 GHz RF probe, which also enables quadrature biasing through a bias-Tee. The optical power at the output fiber of the MZM operated at the quadrature point is 3.7 dBm, exceeding typical Ethernet transceiver specs [6]. An EDFA is used to reach a sufficiently high-power level of 10 dBm at the receiver, where a photodiode (PD) is directly connected to a real-time oscilloscope (RTO, UXR 1004A, Keysight). In real-world systems, a broadband RF amplification after the PD could eliminate the need for the EDFA. After digitization by the RTO, the data is finally extracted using offline DSP (Rx-DSP), which contains resampling to 2 samples per symbol, timing recovery, linear Sato equalization, and an additional least-mean-square (LMS) equalizer.

In the data transmission experiment, we generated and received PAM4, PAM6 and PAM8 data signals with symbol rates between 144 GBd and 208 GBd. Figure 2(f) shows the measured bit error ratios (BER) of the various PAM signals as a function of symbol rate. The horizontal dashed lines indicate the thresholds for typical soft-decision (SD) forward-error correction (FEC) with 25 % and 15 % overhead, and for hard-decision (HD) FEC with 7% overhead. The results show that we can transmit PAM8 signals at 184 GBd, PAM6 signals at 192 GBd and PAM4 signals at 208 GBd with BER values below the 25% SD-FEC limit. Fig. 2 (g) shows the achievable information rate (AIR, dashed line, asterisks), calculated form the generalized mutual information (GMI) of our measurements assuming an additive white Gaussian noise channel model [7]. The solid curves in Fig. 2(g) show the net data rates for measurements with BER below the 25% SD-FEC limit, calculated using the code rates that are associated with the normalized GMI threshold given in [8]. The highest AIR of 466 Gbit/s is achieved with PAM8 at 180 GBd, yielding a net data rate of 437 Gbit/s, comparable to high-bandwidth LNOI MZM results [9].

### **4. Summary**

We demonstrate high-speed, scalable-manufactured LT-on-FS Mach-Zehnder modulators. The modulator exhibits a DC Vπ of 1.56 V for 1550 nm and 1.28 V for 1300 nm, and a 3-dB EO bandwidth of 67 GHz. We reach PAM4 symbol rates of 208 GBd, PAM6 symbol rates of 192 GBd and PAM8 symbol rates of 184 GBd, all with measured BER below the 25% SD-FEC threshold. This leads to a single-carrier line rate of 552 Gbit/s (184 Gbd PAM8) and a net data rate of 437 Gbit/s (180 GBd PAM8) The results of this paper highlight the significant potential of the LTOI platform in highly scalable IMDD transceivers.

### **5. Acknowledgements**

This work was supported by the Swiss National Science Foundation under grant agreement No. 216493 (HEROIC) and the Horizon Europe EIC Transition programme under grant agreement No. grant agreement 101131069 (ELLIP-TIC), as well as funding from the Swiss State Secretariat for Education, Research and Innovation (SERI). This work was further supported by the DFG via the project PACE (#403188360) and Gospel (# 403187440), and by the BMBF project Open6GHub (#16KISK010).

### **6. References**

- [1] Wang, Chengli, Zihan Li, *et al.* "Lithium tantalate photonic integrated circuits for volume manufacturing." *Nature* (2024): 1-7.
- [2] Powell, Keith, *et al.* "Stable electro-optic modulators using thin-film lithium tantalate." *arXiv preprint* arXiv:2405.05169 (2024).
- [3] Wang, Chengli, *et al.* "Ultrabroadband thin-film lithium tantalate modulator for high-speed communications." *Optica* 11, 1614-1620 (2024).
- [4] Lin, Mengxin, *et al.* "Copper Damascene Process-Based High-Performance Thin Film Lithium Tantalate Modulators." *arXiv preprint* arXiv: 2505.04755 (2025).
- [5] Kharel, Prashanta, *et al.* "Breaking voltage–bandwidth limits in integrated lithium niobate modulators using micro-structured electrodes." *Optica* 8.3 (2021): 357-363.
- [6] "IEEE Standard for Ethernet Amendment 10: Media Access Control Parameters, Physical Layers, and Management Parameters for 200 Gb/s and 400 Gb/s Operation," IEEE Std 802.3bs-2017.
- [7] M. Ivanov *et al.*, "On the Information Loss of the Max-Log Approx-imation in BICM Systems," *IEEE Trans. on Info. Theory*, 62 (6), 3011- 12976 (2020).
- [8] Q. Hu *et al.*, "Ultrahigh-Net-Bitrate 363 Gbit/s PAM-8 and 279 Gbit/s Polybinary Optical Transmission Using Plasmonic Mach-Zehnder Modula-tor," *J. Light. Technol.*, 40 (10), 3338–3346 (2022).
- [9] E. Berikaa *et al.*, "TFLN MZMs and Next-Gen DACs: Enabling Beyond 400 Gbps IMDD O-Band and C-Band Transmission," *IEEE Photon. Technol. Lett.*, 35 (15), 850–853 (2023).