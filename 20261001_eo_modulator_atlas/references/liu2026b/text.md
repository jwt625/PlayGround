---
paper_id: liu2026b
source_url: https://doi.org/10.1364/ofc.2026.w2a.14
doi: 10.1364/ofc.2026.w2a.14
license: 
sha256: a3b0fe4215635414889c04ab64959c03150748adc6f17de95907d211bff07aff
pages: 3
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
High-Bandwidth Serpentine Segmented Silicon Photonic 
Mach-Zehnder Modulator for 192 Gbaud Transmission 
 
Qian Liu1,2, Chao Cheng1,2, Yangming Ren1, Xinyu Li1, Wenfu Zhang1,2, and Binhao Wang1,2* 
1State Key Laboratory of Ultrafast Optical Science and Technology, Xi’an Institute of Optics and Precision Mechanics, Chinese Academy of 
Sciences, Xi’an 710119, China 
2School of Future Technology, University of Chinese Academy of Sciences, Beijing 100049, China 
*wangbinhao@opt.ac.cn 
 
Abstract: We demonstrate a serpentine segmented silicon Mach-Zehnder modulator with a 1 dB 
bandwidth >67 GHz at 0 V bias and a modulation efficiency of 1.14 V∙cm, supporting 192 Gbaud 
transmission. 
 
1.Introduction 
Driven by the explosive growth of artificial intelligence, high-performance computing, and cloud computing, global 
data traffic is expanding at an unprecedented pace, with data rates advancing toward 1.6 Tbps and beyond. Silicon 
photonics, offering low cost, high integration, and low power consumption, has emerged as a key technology to 
address these transmission bottlenecks [1]. As the core component enabling the conversion between electrical and 
optical signals in communication systems, the electro-optic (EO) modulator plays a crucial role [2]. With the 
continuous growth in communication capacity and bandwidth density, modulators are facing increasingly stringent 
demands on speed, power efficiency, and integration [3]. 
Silicon photonic modulators primarily rely on the plasma dispersion effect [4] within PN junctions. However, 
due to the relatively weak effect in silicon, a long modulation region is often required to achieve low-voltage 
operation. This constraint not only limits the EO bandwidth but also poses challenges for compact integration. To 
overcome these limitations, various modulator architectures have been proposed to enhance performance. For 
instance, a 124 μm-long silicon modulator leveraging the slow-light effect of Bragg gratings demonstrated a 3 dB 
bandwidth of 110 GHz and supported OOK transmission at 112 Gbaud [5]. In another approach, a segmented 
electrode silicon modulator achieved a 3 dB bandwidth exceeding 67 GHz, enabling 120 Gbaud 8ASK signaling [6]. 
An EO bandwidth above 110 GHz has also been reported using tunable time-frequency equalization technique [7]. 
Furthermore, impedance peaking in silicon traveling-wave modulators enables bandwidths up to 63.5 GHz [8], 
while distributed micro-capacitor designs can significantly extend the bandwidth without increasing the Vπ 
compared with conventional structures [9]. Although these approaches have considerably advanced modulator 
performance, further enhancement in speed, efficiency, and integration remains a critical research focus. 
To enable high-speed transmission within a compact footprint, we demonstrate a serpentine segmented-
electrode Mach-Zehnder modulator (MZM) at O-band. The proposed device occupies only half the area of a 
conventional MZM with the same length, exhibiting a modulation efficiency of 1.14 V·cm and a 1 dB EO 
bandwidth exceeding 67 GHz at 0 V bias, supporting 192 Gbaud OOK transmission. 
 
 
Fig. 1. (a) Schematic of the serpentine segmented MZM. (b) Cross-sectional view of the serpentine segmented MZM. 
W2A.14.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 2 -->
2. Device Design 
Fig. 1(a) illustrates the schematic of the proposed serpentine segmented MZM structure. The device consists of 
three modulation segments with serpentine-arranged ground-signal (GS) traveling-wave electrodes and waveguides. 
Each phase shifter has a length of 950 μm and the overall modulator footprint is 2.25 mm × 0.45 mm. This 
configuration is designed for flip-chip integration with a distributed driver chip, which significantly enhances 
integration density and enables higher bandwidth compared with conventional wire-bonded photonic-electronic 
integration. The device is fabricated on a silicon-on-insulator (SOI) platform with a 220 nm-thick top silicon layer 
and a 2 μm-thick buried oxide (BOX). The cross-sectional structure of the modulator phase shifter is illustrated in 
Fig. 1(b). The phase shifters employ laterally structured PN junctions with three-level doping concentrations in 220 
nm × 380 nm silicon rib waveguides with a 150 nm etch depth. To optimize device performance, the P and N 
regions are doped to 6 × 1017 cm-3 and 1 × 1018 cm-3, with a 100 nm junction offset introduced to balance optical 
loss and modulation efficiency. Intermediate p+ and n+ doping concentrations of 1 × 1019 cm−3 are used to 
simultaneously mitigate optical loss, PN-junction-limited bandwidth, and microwave attenuation, while heavily 
doped p++ and n++ regions (1 × 1020 cm−3) form the ohmic contacts. The modulator adopts a series push-pull 
electrode configuration, with mismatched terminating resistors (40 Ω) integrated at the end of the 52 Ω traveling-
wave electrodes to enhance bandwidth. A 100 μm optical path imbalance is introduced between the two modulation 
arms to facilitate DC characterization and bias tuning via wavelength adjustment. Additionally, thermal phase 
shifters are integrated on both arms to precisely control the bias point and maintain optimal linearity. 
3. Experimental Results 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Fig. 2. (a) Measured transmission spectra of the modulator at 0 V and -4 V. (b) EO responses of the three segments at 0 V and -2 V. 
Fig. 2(a) shows the static transmission spectra of the modulator when all segments are driven simultaneously, 
revealing a measured insertion loss of 9 dB. To extract the modulation efficiency, one junction was bypassed while 
monitoring the resulting phase shift, and procedure was repeated for the other junction. The average of the two 
measurements yields a Vπ of 4 V under a -2 V bias, corresponding to a modulation efficiency of 1.14 V·cm. To 
evaluate the high-frequency performance, the small-signal EO response was measured using a 67 GHz photonic 
network analyzer (PNA) and a 100 GHz photodetector. The optical signal was coupled in and out of the modulator 
via grating couplers, while RF signal was applied to one segment of the modulator through a 67 GHz GS RF probe. 
A bias-tee was used to set the DC offset of the modulation signal. Fig. 2(b) presents the measured EO responses of 
three modulation segments at bias voltages of 0 V and -2 V. Because of the mismatched terminating resistors, a 
low-frequency peaking effect was observed. Under a 0 V bias, all segments exhibit a 1 dB EO bandwidth exceeding 
67 GHz. Moreover, the bandwidth of all segments shows almost no degradation when a -2 V bias is applied, 
indicating a 3 dB EO bandwidth exceeding 100 GHz. 
W2A.14.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 3 -->
Fig. 3. (a) Experimental setup for the MZM eye diagram measurement. (b)-(g) Measured optical eye diagrams at 64, 96, 112, 128, 160, and 192 
Gbaud OOK signals. 
Fig. 3(a) illustrates the schematic of the large signal experimental setup for the serpentine segmented MZM. RF 
signals are generated by channels of a 256 GSa/s arbitrary waveform generator (AWG), amplified by a 67 GHz, 11 
dB amplifier, and applied to a single segment of the MZM via a 67 GHz GS RF probe. The output optical signal 
from the MZM is amplified through a praseodymium-doped fiber amplifier (PDFA) and detected with a 110 GHz 
oscilloscope. Fig. 3 (b)-(g) show the measured optical eye diagrams of the MZM for 64, 96, 112, 128, 160, and 192 
Gbaud with PRBS7 OOK signals. The corresponding extinction ratios are 3.67 dB, 2.87 dB, 2.51 dB, 2.23 dB, 1.68 
dB and 1.17 dB, respectively. 
4. Conclusion 
We report a serpentine segmented MZM with a footprint of 2.25 mm × 0.45 mm, achieving a 1dB EO bandwidth 
exceeding 67 GHz under 0 V bias. Experimental results demonstrate 192 Gbaud NRZ optical transmission. By 
employing a serpentine segmented design, the modulator not only achieves higher bandwidth but also significantly 
enhances compactness and integration density. This structural optimization meets the demands of optical 
communication systems for high integration and data rates, highlighting the promising potential of silicon photonics 
for 400 Gbps/λ data transmission. 
5. Acknowledgement 
This work was supported by the National Key Research and Development Program of China (2022YFB2803100). 
6. References 
[1] Y. Shi, Y. Zhang, Y. Wan, et al., “Silicon photonics for high-capacity data communications,” Photonics Research. 10, A106–A134 (2022). 
[2] G. T. Reed, G. Mashanovich, F. Y. Gardes, et al., “Silicon optical modulators,” Nat. Photonics 4, 518–526 (2010). 
[3] Q. Cheng, M. Bahadori, M. Glick, et al., “Recent advances in optical technologies for data centers: a review,” Optica 5, 1354–1370 (2018). 
[4] R. Soref, B. Bennett, “Electrooptical effects in silicon,” IEEE J. Quantum Electron. 23(1), pp. 123–129, (1987). 
[5] C. Han, Z. Zheng, H. Shu, et al., “Slow-light silicon modulator with 110-GHz bandwidth,” Sci. Adv. 9, eadi5339 (2023). 
[6] A. Mohammadi, Z. Zheng, X. Zhang, et al., “Segmented silicon modulator with a bandwidth beyond 67 GHz for high-speed signaling,” 
Journal of Lightwave Technology. PP, 1–8 (2023). 
[7] H. Yue, J. Fu, H. Zhang, et al., "Silicon modulator exceeding 110 GHz using tunable time-frequency equalization," Optica 12, 203-215 
(2025). 
[8] A. Mohammadi, S. Levasseur, W. Shi, "Impedance Peaking in Silicon Traveling Wave Mach-Zehnder Modulators," 2023 Conference on 
Lasers and Electro-Optics (CLEO), 2023, pp. 1-2. 
[9] A. Mohammadi, L. A. Rusch, W. Shi, “Bandwidth enhancement of silicon traveling-wave modulators using distributed micro-capacitors,” in 
49th European Conference on Optical Communications (ECOC 2023), 2023, pp. 1512–1515. 
W2A.14.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

