---
paper_id: dong2026
source_url: https://doi.org/10.1364/ofc.2026.th4a.4
doi: 10.1364/ofc.2026.th4a.4
license: 
sha256: f807fc1d68074532bed25bd2715ccc0f034e7f6fa76be1de2eaea16d66b5a7ba
pages: 3
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
400G/lane PAM4 Modulation Using Silicon Mach-Zehnder 
Modulators  
Po Dong, Venkatesh Doddapaneni, Michael Kossey, Craig Schulz, Andrei Kaikkonen, Roberto Rodes, Dun 
Mao, Jaime Kwan, Taylor Fryett, Bo Feng, Argishti Melikyan, Chengpin Yu and Jack Xu 
Coherent Corp., 48800 Milmont Dr., Fremont, CA 94538, USA 
E-mail address: po.dong@coherent.com 
 
Abstract: Silicon Mach–Zehnder modulators (MZMs) are widely used for 100G/200G PAM4, but 
their scalability to IMDD 400G/lane is uncertain. We demonstrate a silicon MZM with a 
commercial SiGe driver enabling 400G/lane PAM4 transmission. 
 
1. Introduction 
The transition to 400G/lane is a critical step for next-generation optical interconnects, driven by the explosive 
bandwidth demands of AI/ML workloads and hyperscale data centers. As port capacities scale to 3.2T and beyond, 
higher per-lane data rates are essential to control system complexity. A 400G/lane architecture reduces fiber count, 
component density, footprint, power, and cost, improving both energy efficiency and cost per bit in large-scale 
deployments. Recent demonstrations have validated 400G/lane PAM4 operation using advanced modulator 
technologies such as InP electro-absorption modulators (EAMs) [1-2], InP Mach-Zehnder modulators (MZMs) [3], 
thin-film lithium niobate [4], hybrid Si/III-V EAMs [5], plasmonic modulators [6], and silicon slow-light modulators 
[7], which offer high bandwidth and strong modulation efficiency. In contrast, standard silicon MZMs have been 
viewed as less suitable due to lower intrinsic bandwidth and higher Vπ, raising concerns about drive voltage and 
design margin. Recent work has demonstrated 300-350 Gbps PAM4 modulation with silicon MZMs [8–11] using 
offline processing typically with unpractical equalizations.  The feasibility of achieving 400G/lane with a practical 
driver has yet to be established. 
     Here, we demonstrate, for the first time to our knowledge, 400G PAM4 modulation using a silicon MZM without 
exotic materials or unconventional structures. The result is achieved with a SiGe driver providing a 2.5 V swing 
within a practical system interface.  This work shows that 400G/lane performance is achievable using Si MZMs, 
preserving the advantages in high-volume manufacturability, proven reliability, and a well-established supply chain. 
The demonstrated performance supports silicon MZMs as a scalable solution for pluggable transceivers, near-
packaged optics (NPO), and co-packaged optics (CPO), helping resolve the debate over their suitability for 
400G/lane interconnects. 
2. Silicon MZM 
The Si MZM employs reverse-biased lateral PN-junction phase shifters fabricated in 220-nm silicon on a 3-µm 
BOX substrate. The electro-optic (EO) bandwidth is fundamentally limited by the junction RC time constant, which 
is determined by the junction capacitance and series resistance. The junction doping profile is carefully engineered 
to minimize resistance while preserving modulation efficiency. Traveling-wave electrodes with 2.8-µm-thick 
aluminum enable both velocity and impedance matching. The modulator adopts a dual-drive differential 
configuration with a target differential impedance of 75 Ω. The measured EO response demonstrates a >70 GHz 3-
dB bandwidth in Fig. 1, while the S11 response indicates low reflection over a broad frequency range. The response 
oscillation at high frequency may be due to the limited bandwidths for RF components/cables in the setup that 
introduces calibration difficulty.  The effective Vπ, defined as the modulation efficiency (Vπ·L) divided by the 
phase-shifter length in the MZM, is approximately 7 V. A sufficiently low Vπ is required to achieve the target 
modulation extinction ratio when driven by a practical modulator driver.  
3. Transceiver prototype and test results 
To verify the modulation capability of the silicon MZM, we designed and assembled a transceiver prototype. A 
commercial DAC/ADC chipset capable of generating and receiving 400G/lane PAM signaling [3] was employed 
and flip-chipped on a printed circuit board (PCB). The DAC output drives a SiGe modulator driver featuring a 
maximum differential output swing of 2.5 V, a small-signal gain of 18 dB, 4 dB of gain peaking, a 3-dB bandwidth 
of 135 GHz, and a power consumption of approximately 500 mW [11]. The driver is wire-bonded to a silicon 
photonic integrated circuit (PIC) containing four MZMs. Since the driver is single-channel, only one MZM is 
characterized in this work. A 70 mW continuous-wave (CW) laser is coupled into the PIC through free-space lenses. 
Th4A.4.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 2 -->
A 1×2 power splitter integrated within the PIC splits the optical input power to a pair of MZMs. At the transmitter 
output, a fiber array unit (FAU) is butt-coupled to the PIC to collect the modulated optical output. On the receiver 
side, a SiGe transimpedance amplifier (TIA) integrated with top-illuminated photodiodes (PDs) is used. It should be 
noted that PD and TIA used in the experiment were designed for 200G/lane operation, with an electrical bandwidth 
below 80 GHz. Commercially available PD/TIA solutions specifically specified for 400G/lane operation were not 
available at the time of this study. The TIA output is wire-bonded to the PCB for signal acquisition and processing. 
Fig. 2(a) shows the block diagram of the transceiver prototype. Fig. 2(b) presents a photograph of the packaged 
device, along with several inset images highlighting key components and assembly details. 
 
Fig. 1. Measured s-parameter of the Si MZM used in this paper. 
 
 
Fig. 2. 400G/lane transceiver prototype with Si MZMs. (a) Block diagram of the packaged device; (b) Photo 
of the packaged device with inset images highlighting key components.  
We evaluated PAM4 signal performance in an optical back-to-back (B2B) configuration at data rates of 300, 360, 
and 420 Gb/s, as shown in Fig. 3. The DAC supports pattern generation with integrated FIR filtering and a look-up 
table (LUT) for level optimization. Random patterns were generated by the DAC, comprising five patterns, each 
with lengths of 1280, 1152, and 1344 symbols at 300 Gb/s, 360 Gb/s, and 420 Gb/s, respectively. Key performance 
metrics were characterized as a function of data rate, including extinction ratio (ER), linearity (RLM), and link bit 
error rate (BER). The BER was measured using the 200G/lane receiver in conjunction with the ADC and offline 
digital signal processing. Eye diagrams were captured using a digital communication analyzer (DCA).  
Figure 3 summarizes the measured results. At 300 Gb/s, the transmitter achieved a modulation extinction ratio of 
4.71 dB and a linearity of 0.93. The link demonstrated excellent performance, with a measured BER of 3 × 10⁻⁶ 
(limited by the total pattern length) even when evaluated using the 200G/lane receiver and ADC. As the data rate 
increased to 360 Gb/s and 420 Gb/s, the ER decreased to 4.22 dB and 3.71 dB, respectively. The corresponding link 
BER increased to 2 × 10⁻² at 360 Gb/s and 8 × 10⁻² at 420 Gb/s, while the RLM remained relatively stable (≈0.94–
0.95). We attribute the BER degradation primarily to bandwidth limitations of the receiver employed in the 
experiment. With increasing data rate, DCA measurements show progressive eye closure and vertical compression, 
indicating bandwidth constraints and enhanced inter-symbol interference under back-to-back operation. Signal 
averaging was applied in both DCA eye catching and BER measurements at 360 Gb/s and 420 Gb/s. For eye 
Th4A.4.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 3 -->
diagram visualization and BER characterization, 15-tap feed-forward equalization (FFE) was used at 300 Gb/s, 
while 31-tap FFE was applied at 360 Gb/s and 420 Gb/s.  
 
Fig. 3. Back-to-back optical transmitter performance at 300–420 Gb/s showing ER, RLM, Link BER, and 
DCA eye diagrams. 
 
4. Discussions and conclusion 
Silicon MZMs are one of the most mature and manufacturable modulator platforms for 100G and 200G/lane 
systems. Enabled by CMOS-compatible fabrication, excellent wafer-level uniformity, proven reliability, and a well-
established supply chain, tens of millions of units have been deployed in hyperscale data centers. Their scalability, 
cost efficiency, and compatibility with high-volume packaging make them the preferred solution for large-scale 
optical interconnects. The 400G/lane demonstration in this work, achieved with a practical SiGe driver, further 
demonstrates their suitability for next-generation optical interconnects. 
Additional performance improvements are feasible. The current wirebond interface between the driver and PIC 
limits high-frequency response and can be significantly improved through flip-chip integration. The TIA and PD 
used in this experiment were designed for 200G/lane operation and constrained to <80 GHz bandwidth. A higher-
bandwidth TIA/PD would further enhance link performance. Moreover, continued scaling of CMOS technology 
nodes is expected to increase DAC and ADC bandwidths, providing additional system margin for 400G/lane 
operation. The bandwidth of the silicon MZM could be further improved by using copper electrodes instead of 
aluminum electrodes. 
 
References 
[1] 
Coherent 
Corp, 
“Coherent 
demonstrates 
industry’s 
first 
400G 
differential 
electro-absorption 
modulated 
laser,” 
https://www.coherent.com/news/press-releases/400g-differential-eml, OFC 2025. 
[2] Keysight, NTT Innovative Devices, and Lumentum, “Achieve new benchmark 448 Gbps data transmission for AI data centers,” 
https://www.keysight.com/us/en/about/newsroom/news-releases, OFC 2025. 
[3] Marvell, “Industry’s first 400G/lane PAM4 electrical-to-optical link technology,” https://www.marvell.com/company/newsroom.html, OFC 
2025. 
[4] TeraHop “Industry 1st 400G per lane technology demo”, https://terahop.com/newsinfo/135.html, OFC 2025. 
[5] OpenLight and Tower Semi, “400G/lane silicon photonics modulators,” https://openlightphotonics.com/newsroom/news-releases, OFC 2025. 
[6] T. Blatter et al., "Plasmonic ring resonator modulator demonstrating IM/DD >400G per lane," ECOC 2024, pp. 418-421. 
[7] C. Han et al., “Exploring 400 Gbps/λ and beyond with AI-accelerated silicon photonic slow-light technology,” Nat Commun. 16, 6547 
(2025). 
[8] X. Li, J. C. Davies, E. Sing Chee Koh and J. T. -Y. Liow, "Silicon photonic traveling wave Mach-Zehnder modulators for 400G/Lane 
transmitters," 2025 IEEE Silicon Photonics Conference (SiPhotonics), London, United Kingdom, 2025, pp. 1-2. 
[9] K. Li et al, “Beyond 300Gb/s from an integrated single-channel silicon photonics modulator driver combination,” 2024 IEEE Silicon 
Photonics Conference (SiPhotonics), Tokyo Bay, Japan, 2024, TuB2. 
[10] A. Ostrovskis et al., "SiP modulators for 400 Gbps on single wavelength in IM/DD systems: is this a reality?," 2025 25th Anniversary 
International Conference on Transparent Optical Networks (ICTON), Barcelona, Spain, 2025, pp. 1-4. 
[11] https://www.macom.com/products/optical/laser-and-modulator-drivers/pam. 
 
Th4A.4.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

