---
paper_id: zhang2026b
source_url: https://doi.org/10.1364/ofc.2026.w1a.3
doi: 10.1364/ofc.2026.w1a.3
license: 
extracted_on: 2026-10-01
extraction_method: pre-extracted text copied from a local corpus; no PDF and no figure images available; figures and tables must be treated as unavailable
---

# **Thin-Film Lithium Niobate Modulators with 110 GHz Bandwidth and 1.9 V·cm Efficiency on 200-mm Silicon Substrate**

**Y. Zhang1,2, C. Lau1,2 , B. Li1,2 , S. N. Foo1,2, J. O. Yoo1,2 , B. Pan1,2 , W. Q. Tan1,2 , N. Singh1,2 , L. W. Lim1,2 , L. Tobing1,2,\*, X. Luo1,2,\*, Y. C. Yeo1,\***

*<sup>1</sup>National Semiconductor Translation and Innovation Centre (NSTIC), 4 Fusionopolis Way, Kinesis #09-11, Singapore 138635 Institute of Microelectronics (IME), Agency for Science, Technology and Research (A\*STAR), 2 Fusionopolis Way, Innovis #08-02, Singapore 138634, Republic of Singapore*

> *\* [landobasa\\_tobing@a-star.edu.sg;](mailto:landobasa_tobing@a-star.edu.sg) [luo\\_xianshu@a-star.edu.sg;](mailto:luo_xianshu@a-star.edu.sg) [yeo\\_yee\\_chia@nstic.sg](mailto:yeo_yee_chia@nstic.sg)*

**Abstract:** We demonstrate thin-film lithium niobate modulators on 200-mm wafers in a back-endof-line CMOS foundry. We achieve O-band propagation loss of <0.5 dB/cm and TFLN modulators with 110 GHz bandwidth and 1.9 V·cm modulation efficiency. © 2026 The Author(s)

## **1. Introduction**

Thin-film lithium niobate (TFLN) platform has been reported to exhibit high-speed modulation beyond 100 GHz as the material possesses high electro-optic coefficient and strong optical confinement [1]. While chip-scale demonstrations are aplenty, large-scale manufacturing of TFLN modulators are impeded by few factors: 1) limitations in commercial wafer sizes, 2) process uniformity across the wafer and 3) lithium contamination handling in a CMOS line. TFLN modulators have used commercially available 6-on-8-inch lithium niobate-oninsulator (LNOI) wafers to leverage on mainstream commercial 200-mm foundries with advanced lithography which has achieved TFLN modulators with >67 GHz bandwidth, <0.5 dB/cm propagation loss and 3.12 V·cm modulation efficiency [2]. An alternative approach towards large-scale, high-performance TFLN modulators is to use wafer-to-wafer bonding of LNOI substrates onto patterned 8-inch silicon nitride substrates [3]. Such methods have yielded hybrid TFLN modulators with voltage-length product (VπL) of 3.8 V·cm and bandwidth >110 GHz but they rely on pre-processing steps such as chemical-mechanical polishing (CMP) and wafer bonding of mismatched wafer sizes, leading to material wastage.

In this work, we design and fabricate high-performance TFLN devices directly on 8-inch TFLN-on-Insulator (TFLNoI) wafers using a back-end-of-line (BEOL) CMOS foundry. This is the first report of TFLN modulators realized on 8-inch TFLNoI platform fabricated in a CMOS line. An average waveguide propagation loss of <0.5 dB/cm across the O-band with high uniformity is obtained. The TFLN modulators shows a maximum bandwidth of 110 GHz and lowest modulation efficiency VπL of 1.87 V·cm. Such 8-inch wafer-scale fabrication using BEOL CMOS fab marks an essential step towards high-volume manufacturing of TFLN modulators.

## **2. Device fabrication**

The fabrication starts with an 8-inch TFLNoI wafer with 350 nm X-cut LN film, 4.7 µm-thick buried oxide layer and high-resistivity 725 µm-thick Si handle (commercially available from NanoLN). SiO2is deposited through a plasma-enhanced chemical vapor deposition (PECVD) process to act as the hardmask for LN etching. Waveguide patterns are defined using 248-nm deep ultraviolet (DUV) lithography. LN etching is performed using inductivelycoupled-plasma reactive-ion-etching (ICP-RIE) followed by by-product removal. The process is repeated to define the LN slab layer. The waveguides have LN rib and slab thicknesses of 200 and 150 nm respectively. An optimized SiO<sup>2</sup> cladding of 1 µm thickness is used as the top cladding. The metallization scheme consists of 1 µm deep vias and Al electrodes of 1.5 µm thickness in a GSG configuration. Fig. 1(a) shows the 8-inch wafer under test in a probe station (FormFactor Summit200 Wafer). Devices structures such as a multi-mode interferometer (MMI) and ring resonators are shown in tilted-view scanning electron microscope images (SEM) in Fig. 1(b) and 1(c), respectively. Fig. 1(d) shows the cross-sectional SEM image of one arm of the modulator and Fig. 1(e) shows the optical microscope image of the fabricated TFLN modulator.

![](_page_1_Figure_2.jpeg)

Fig. 1. (a) Photo of the 8-inch wafer-under-test in an optical probe station. Tilted SEM images of the fabricated passive structures: (b) MMI and (c) microring resonator. (d) XSEM and (e) optical microscope images of a Mach-Zehnder modulator.

# **3. Results and Discussion**

The waveguide propagation loss of single-mode ridge waveguides with 1.0 µm width are measured using cutback structures and vertical grating couplers over the entire 8-inch wafer using a wafer-level optical tester (FormFactor Summit200 Wafer). Figure 2(a) shows the propagation loss from 1280 to 1360 nm with an average loss of <0.5 dB/cm across the O-band. The error bars here illustrate the standard deviation where a larger spread at short wavelengths due to the weaker grating coupler response. Figure 2(b) shows the wafer-scale propagation loss at 1330 nm, indicating a uniform optical loss of 0.37 ± 0.06 dB/cm. The exact cause of the discrepancy here is under investigation but could be due to a combination of the following: i) total thickness variation of incoming LNOI wafer and ii) non-uniformity of ICP etch across wafer. We then measure the transmission spectra of a typical unbalanced Mach-Zehnder interferometers as shown in Fig. 2(c) with > 25 dB extinction ratio, which indicates the designed MMI splitters have been well-fabricated. We also measure the transmission of microring resonator and extract a loaded Q-factor ~4 × 10<sup>5</sup> . This corresponds to an intrinsic Q-factor in the order of ~1.65 × 10<sup>6</sup> and an extracted loss of ~0.3 dB/cm which matches well with the propagation loss extracted from the cutback method (data not shown here).

![](_page_1_Figure_6.jpeg)

Fig. 2. Passive TFLN device characterization: (a) optical propagation loss of single-mode waveguides in the O-band, showing <0.5 dB/cm averaged loss. Inset: Linear fitting of the cutback measurement at 1330 nm wavelength. (b) Wafer-scale propagation loss at 1330 nm with propagation loss of 0.37 ± 0.06 dB/cm. (c) Normalized transmission spectrum a MZI modulator, showing > 25 dB extinction ratio.

The DC V<sup>π</sup> of a 5-mm-long modulator with metal space of 5 μm is measured using a 1 kHz sinusoidal signal. Figure 3(a) shows a typical measurement with V<sup>π</sup> of 4.29 V at 1310 nm, which corresponds to a VπL of 2.15 V·cm. Figure 3(b) shows the wafer-scale measurement results of the modulation efficiency, which shows an average of 2.37 ± 0.35 V·cm modulation efficiency, with the best result of 1.87 V·cm. Figure 3(c) shows the EE S-parameter of the co-planar waveguide (CPW) electrodes measured using a 110 GHz vector network analyzer (VNA). From this, the microwave phase index is extracted to be 2.2 which is close to the optical phase index of 2.27 extracted from MZI structures. This allows for good phase matching of the RF and optical wave to achieve a high EO modulation efficiency. The EO response of few exemplary MZI modulators measured on chip-level is illustrated in Fig. 3(d) which shows a 3-dB bandwidth ranging between 93 to 110 GHz.

![](_page_2_Figure_2.jpeg)

Fig. 3 (a) The measured DC V<sup>π</sup> for a 5-mm long modulator at the wavelength of 1310 nm with a 1 kHz triangular voltage sweep. (b) Wafer map of VπL for 5-mm long modulator, showing an average of 2.37 ± 0.35 V·cm and the best result of 1.87 V·cm. (c) Calculated RF loss and microwave index from EE S<sup>21</sup> response. (d) Measured EO S<sup>21</sup> response of modulators at chip-level showing maximum bandwidth of 110 GHz.

Ref. Year Wafer size Wavelength (µm) Loss (dB/cm) VπL (DC) (V·cm) EO Bandwidth (GHz) [1] 2025 4" LNoI on quartz C-band 0.24 2.98 >110 [2] 2024 6" LN on 8" Si C-band 0.47 3.12 >67 [4] 2025 6" LN on 8" Si C-band NA 2.9 >50 [5] 2025 6" LN on 8" Si C-band NA 3.8 >110 **This work** 2025 8" LNoI O-band < 0.5 1.87 110

Table 1. Benchmark of wafer-scale thin-film LN photonic devices

## **4. Conclusion**

We demonstrated TFLN modulators on an 8-inch BEOL CMOS fabrication line, showcasing excellent uniformity in optical propagation loss of < 0.5 dB/cm across the O band. TFLN modulator shows highest bandwidth of 110 GHz and the best VπL of 1.87 V·cm. This work represents a significant milestone toward the mass production of TFLN photonic integrated circuits using an 8-inch CMOS-compatible manufacturing line.

## **5. Acknowledgements**

We acknowledge funding support from National Semiconductor Translation and Innovation centre (NSTIC).

## **6. References**

- [1] Yang Liu, et al. "Volume manufacturing of thin-film lithium niobate modulators with bandwidth > 110 GHz based on 4-inch wafer with a quartz handle," Light: Advanced Manufacturing 6, 17 (2025).
- [2] J. Zhou et al., "High-Performance Thin-Film Lithium Niobate Mach-Zehnder Modulator on 8-Inch Silicon Substrate," 2024 Optical Fiber Communications Conference and Exhibition (OFC), pp. 1-3 (2024).
- [3] Tahman. Md et al, "High-performance Hybrid Lithium Niobate Electro-optic Modulators Integrated with Low-loss Silicon Nitride Waveguides on a Wafer-scale Silicon Photonics Platform", arXiv:2504.00311, 2025.
- [4] Z. Li et al, "Lithium niobate electro-optical modulator based on ion-cut wafer scale heterogeneous bonding on patterned SOI wafers" Photonics Research, Vol. 13, No. 1 (2025).
- [5] Tahman. Md et al, "High-performance Hybrid Lithium Niobate Electro-optic Modulators Integrated with Low-loss Silicon Nitride Waveguides on a Wafer-scale Silicon Photonics Platform", arXiv:2504.00311, 2025.