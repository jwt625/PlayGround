---
paper_id: liu2026c
source_url: https://doi.org/10.1364/ofc.2026.th2a.11
doi: 10.1364/ofc.2026.th2a.11
license: 
extracted_on: 2026-10-01
extraction_method: pre-extracted text copied from a local corpus; no PDF and no figure images available; figures and tables must be treated as unavailable
---

# **A 280 Gbps PAM6 Silicon Photonic Tabbed-Electrode Mach-Zehnder Modulator with Co-Optimized Modulation Efficiency and Electro-Optic Bandwidth**

**Qian Liu1,2† , Jintao Xue1,2† , Chao Cheng1,2 , Yangming Ren<sup>1</sup> , Xinyu Li<sup>1</sup> , Wenfu Zhang1,2, and Binhao Wang1,2\*** *<sup>1</sup>State Key Laboratory of Ultrafast Optical Science and Technology, Xi'an Institute of Optics and Precision Mechanics, Chinese Academy of Sciences, Xi'an 710119, China*

*<sup>2</sup>School of Future Technology, University of Chinese Academy of Sciences, Beijing 100049, China † These authors contributed equally to this work \*wangbinhao@opt.ac.cn*

**Abstract:** A tabbed-electrode silicon Mach-Zehnder modulator is demonstrated, achieving a 1 dB electro-optic bandwidth of 66 GHz and a Vπ of 4.5 V through co-optimization of modulation efficiency and bandwidth, enabling 280 Gbps PAM6 transmission.

#### **1.Introduction**

The exponential growth of artificial intelligence (AI) and machine learning is driving unprecedented demand for computational resources, necessitating high-performance optical interconnects to support next-generation computing systems. As AI workloads continue to scale, the limitations of electrical interconnects become increasingly apparent, motivating the adoption of silicon photonics for energy-efficient, high-bandwidth data transmission [1]. Within this context, electro-optic (EO) modulators serve as critical components, converting electrical signals into optical domains and defining the overall link performance.

Among various modulator technologies, lithium niobate devices offer high performance but face challenges in cost and integration with complementary metal-oxide-semiconductor (CMOS) processes, limiting their use in costsensitive, short-reach applications [2]. In contrast, silicon-based modulators provide a promising alternative, combining CMOS compatibility with the potential for large-scale photonic integration. Two dominant architectures in this space are Mach-Zehnder modulators (MZMs) and micro-ring modulators (MRMs). While MRMs offer compact footprints suitable for dense integration [3], MZMs deliver superior thermal stability and linear modulation characteristics, making them favorable in applications where footprint constraints are relaxed [4,5].

As noted in [6], the current research and development (R&D) challenges for pure silicon MZMs primarily concern achieving both high EO bandwidth and modulation efficiency. To overcome these limitations, we propose an optimized periodic electrode design incorporating tabbed routing, enabling a synergistic improvement in both bandwidth and efficiency. Compared with conventional traveling-wave electrode (TWE) MZMs, the proposed tabbed-electrode design increases the 1 dB EO bandwidth by 27 GHz, from 39 GHz to 66 GHz, and reduces the half-wave voltage (Vπ) by 1.4 V, from 5.9 V to 4.5 V. Moreover, 280 Gbps PAM6 data transmission is experimentally demonstrated, highlighting the potential of this approach to overcome the performance bottlenecks of pure silicon MZMs in single-lane 200G/400G applications.

## **2. Device Design**

![](_page_0_Picture_11.jpeg)

Fig. 1. (a) Schematic and (b) Cross-sectional view of the silicon photonic tabbed-electrode MZM.

As illustrated in Fig. 1(a), the proposed MZM employs a tabbed TWE configuration designed to achieve high EO bandwidth and low drive voltage. Conventional TWE-MZMs often face a trade-off in which shortening the phase shifter to extend bandwidth inherently degrades modulation efficiency. The proposed design overcomes this limitation by embedding a symmetric PN junction in each arm as the phase modulation section, above which the tabbed TWE is implemented. Unlike conventional electrodes, the tabbed structure consists of a series of periodically arranged metal "tabs" adjacent to a main electrode on both sides. These tab pairs function as capacitive elements, forming a synthetic impedance matching network that lowers the microwave effective index for improved velocity matching with the optical mode, thereby extending the EO bandwidth. At the same time, this network tunes the characteristic impedance toward 50 Ω to minimize reflections while maintaining low loss. Moreover, the tabbed-electrode design enhances the overlap between the microwave electric field and the optical mode field, leading to a reduced Vπ and improved modulation efficiency.

As shown in Fig. 1(b), the MZM is fabricated on a silicon-on-insulator (SOI) platform with a buried oxide layer thickness of 2 μm and a top silicon layer thickness of 220 nm. The modulator employs a lateral PN junction embedded in a 220 nm × 380 nm silicon ridge waveguide with an etch depth of 150 nm. A three-level doping scheme is employed. The adjacent p+ and n+ regions are doped to 5.3 × 10¹⁸ cm⁻³ and 1.0 × 10¹⁹ cm⁻³ to balance optical loss, junction-limited bandwidth, and microwave attenuation, while heavily doped p++ (7.4 × 10¹⁹ cm⁻³) and n++ (1.1 × 10²⁰ cm⁻³) regions form low-resistance ohmic contacts. The device employs a series push-pull electrode configuration with an integrated termination resistor, and thermal phase shifters are incorporated on both arms to enable precise bias control for optimal linear operation.

#### **3. Experimental Results**

![](_page_1_Figure_5.jpeg)

Fig. 2. (a) Micrographs of conventional- and tabbed-electrode MZM. (b) Normalized optical transmission of the tabbed-electrode MZM and the conventional TWE-MZM. (c) EO responses of the tabbed-electrode MZM and the conventional TWE-MZM at a -3 V bias.

As shown in Fig. 2(a), the micrographs of the conventional and tabbed-electrode MZMs reveal identical optical and doping designs, differing only in the electrode configuration. Both devices have a phase shifter length of 2 mm and the same ground-signal (GS) pad layout, ensuring compatibility with high-bandwidth-density optical engines (in Gbps/mm of shoreline). Fig. 2(b) presents the DC characterization results, comparing the optical transmission responses of the tabbed-electrode MZM and the conventional TWE-MZM under varying reverse bias voltages. The tabbed-electrode MZM achieves a π phase shift at a bias voltage of 4.5 V, corresponding to a modulation efficiency of 0.9 V·cm, whereas the conventional TWE-MZM requires 5.9 V to achieve the same phase shift, corresponding to a modulation efficiency of 1.18 V·cm. This represents an approximate 24% improvement in modulation efficiency for the proposed design. The EO responses of both modulators are shown in Fig. 2(c). Small-signal S₂₁ measurements were conducted using a photonic network analyzer (Keysight N5227A) and a 100 GHz photodetector. Optical input/output coupling was facilitated via grating couplers, and RF signal was applied using a 67 GHz GS probe. Under a -3 V bias, the conventional MZM exhibits a 1 dB EO bandwidth of 39 GHz, while the tabbedelectrode MZM demonstrates a significantly enhanced 1 dB bandwidth of 66 GHz—corresponding to a 69% improvement. Although the 3 dB cutoff frequency could not be directly measured due to instrument bandwidth limitations, curve fitting of the acquired data indicates a 3 dB bandwidth exceeding 80 GHz for the proposed device.

Fig. 3(a) shows a schematic of the large-signal experimental setup for the tabbed-electrode MZM. The RF signal generated by a 256 GSa/s arbitrary waveform generator (AWG, Keysight M8199B) is amplified by a 67 GHz, 11 dB amplifier (SHF M827B), and then fed into the modulator via a 67 GHz GS RF probe. The output optical signal is amplified by a praseodymium-doped fiber amplifier (PDFA) and detected by a broadband oscilloscope

(Keysight N1000A). Notably, the experimental link was not AWG-calibrated, and due to cable losses, the electrical drive swing was approximately 2.4 Vpp at 64 Gbaud and decreased to 1.8 Vpp at 128 Gbaud. Fig. 3(b) and (c) show the optical eye diagrams for PRBS7 NRZ modulation at 64 Gbaud and 128 Gbaud, with extinction ratio (ER) of 5.08 dB and 2.21 dB, respectively. Fig. 3(d) and (e) present the eye diagrams for 128 Gbps and 256 Gbps PAM4 PRBS7 patterns, corresponding to transmitter dispersion eye closure quaternary (TDECQ) values of 0.33 dB and 2.88 dB. The open 256 Gbps PAM4 (Fig. 3(e)) and 280 Gbps PAM6 (Fig. 3(f)) eye diagrams demonstrate robust signal integrity under unoptimized test conditions, indicating sufficient system bandwidth and confirming that the MZM is fully capable of supporting single-lane 200 Gbps applications.

![](_page_2_Figure_3.jpeg)

Fig. 3. (a) Experimental setup for the MZM eye diagram measurements. Measured optical eye diagrams at (b) 64 Gbps NRZ, (c)128 Gbps NRZ, (c) 128 Gbps PAM4, (d) 256 Gbps PAM4, and (f) 280 Gbps PAM6.

### **4. Conclusion**

We demonstrated a low-voltage (Vπ = 4.5 V) silicon photonic MZM incorporating a tabbed electrode, which significantly enhances device performance, achieving a 1 dB EO bandwidth of 66 GHz and a modulation efficiency of 0.9 V·cm—a 69% increase in EO bandwidth and a 24% improvement in modulation efficiency. The modulator supports single-lane 256 Gbps PAM4 and 280 Gbps PAM6 transmission. These performance gains result from the tabbed electrode's superior microwave–optical phase velocity matching and reduced microwave loss. This work highlights the continued relevance of all-silicon MZMs in the 200G/400G era and provides an effective electrode design strategy for high-performance optical interconnects.

#### **5. Acknowledgement**

This work was supported by the National Key Research and Development Program of China (2022YFB2803100).

# **6. References**

- [1] R. Mahajan et al., "Co-Packaged Photonics for High Performance Computing: Status, Challenges and Opportunities," J. Lightwave Technol, 40, 2, pp. 379-392 (2022).
- [2] F. Valdez et al., " 110 GHz, 110 mW hybrid silicon-lithium niobate Mach-Zehnder modulator, " Sci. Rep., 12, pp. 18611 (2022).
- [3] J. Xue et al., "An 8×256 Gbps Silicon Photonic DWDM Transmitter with Thermally Stable Microring Modulators," OFC2025, pp. Th1E.3.
- [4] K. Li et al., " An integrated CMOS–silicon photonics transmitter with a 112 gigabaud transmission and picojoule per bit energy efficiency," Nat. Electron., 6, pp. 910–921 (2023).
- [5] X. Li et al., "1.6 Tbps FOWLP-Based Silicon Photonic Engine for Co-Packaged Optics," J. Lightwave Technol., 43, 4, pp. 1979-1986 (2025).
- [6] S. Shekhar et al., "Roadmapping the next generation of silicon photonics," Nat. Commun., 15, pp. 751 (2024).