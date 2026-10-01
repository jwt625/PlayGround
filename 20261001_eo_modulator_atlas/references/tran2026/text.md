---
paper_id: tran2026
source_url: https://doi.org/10.1364/ofc.2026.w3e.6
doi: 10.1364/ofc.2026.w3e.6
license: publisher-copyright
sha256: ce7d51f49a2d4de89be3724897a81a267e7834bc27eadd2f8f1d5a4e301ab74e
pages: 3
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
180 GBaud PAM4 Driver-Modulator Engine for IM/DD 
Transmissions in the O-Band 
 
Son Tran1, Sebastian Lauck1, Gerrit Fiol1, Jonathan Andree1, Felix Ganzer1, Robert Emmerich1, Colja 
Schubert1, Patrick Runge1, Martin Schell1,2, Ronald Freund1,2, Shawn Parker3, Trong Phan3, Billy Allen3 
1Fraunhofer Heinrich-Hertz-Institute, Einsteinufer 37, 10587 Berlin, Germany 
2Technical University Berlin, 10623 Berlin, Germany 
3MACOM Technology Solutions, Santa Clara, CA 95050, USA 
thanh.son.tran@hhi.fraunhofer.de 
 
Abstract: A co-packaged 76 GHz InP-based Mach-Zehnder modulator and a 224 GBaud-class 
linear differential EML driver was demonstrated, achieving 180 GBaud PAM4 back-to-back in an 
IM/DD transmission for O-Band operation. 
 
1. Introduction 
The continuous increase in data traffic within short-reach and metro optical networks demands electro-optical 
transmitters capable of supporting high symbol rates while maintaining low power consumption and cost-effective 
packaging and integration [1-11]. Electro-absorption modulated lasers (EMLs) combine low energy consumption 
and compact footprint, making them a well-established choice for short-reach intensity modulation and direct 
detection (IM/DD) links [1, 6]. However, their modulation bandwidth, which is typically limited to several tens of 
gigahertz, together with the inherent chirp and limited extinction ratio (ER), constraint the transmission distance and 
degrade the signal integrity. Therefore, Mach-Zehnder modulators (MZM) have emerged as a key transmitter 
architecture for high-bandwidth and advanced modulation formats due to their chirp-free operation and high ER [2-
5]. While MZMs on thin-film lithium niobate (TFLN) have recently demonstrated excellent bandwidth performance 
[7, 8], their electrode geometry typically leads to large footprint and restricts differential operation. Moreover, due 
to the well-known DC-drift effect [9], stable bias operation in TFLN devices requires additional permanent phase 
control electronics, resulting in increased power consumption. Indium phosphide (InP) platforms stand out through 
their ability to monolithically integrate active and passive photonic components, thus significantly reducing coupling 
losses, footprint and fabrication complexity [11]. Owing to its travelling-wave electrode (TWE) design, InP-based 
MZMs supports differential driving operation [10] and are suited for modern high-speed differential driver designs. 
We introduce an InP-based MZM with a flat electro-optical response (EOR) up to 76 GHz and a modulation 
efficiency of VπL = 0.62 V cm being co-packaged with a 20 dB gain commercial EML differential driver using a low-
complexity assembly approach. The large volume availability of EML drivers allows for a cost-effective and scalable 
transmitter implementation. System-level measurements employing pulse amplitude modulation with 4-levels 
(PAM4) were performed in a back-to-back (B2B) configuration to evaluate the modulation performance in the O-
band. Tab. 1 summarizes the current state-of-the-art transmitter technologies and their performance.  
Reference 
Technology 
Component 
 
Symbol rate/Channel 
Bit rate/Channel 
Lumentum (2025) [1] 
InP 
EML 
113 GBaud PAM4 
226 GBps 
NTT (2025) [2] 
InP 
4x IQ-MZM 
172 GBaud PAM6 
430 GBps 
Nokia (2025) [3] 
InP 
8x MZM 
106 GBaud PAM4 
212 GBps 
D’Lithium (2025) [4] 
SiPh 
MZM+Driver 
128 GBaud PAM8 
384 GBps 
McGill Uni (2025) [5] 
TFLN 
8x MZM+Driver 
225 GBaud PAM4 
450 GBps 
This work 
InP 
MZM+Driver 
180 GBaud PAM4 
360 GBps 
Tab. 1: Performance comparison of the recently presented transmitter technologies.  
2. Design, Fabrication & Assembly 
The p-i-n heterostructure MZM chip was fabricated on HHI’s InP platform. It includes a traveling-wave electrode 
(TWE) design and spot size converters at the facet for simplified fiber coupling. The TWE and the on-chip termination 
resistors, monolithically integrated and matched to the driver’s differential output impedance of 85 Ω, were optimized 
for differential operation with MACOM’s EML driver. The radio frequency (RF) losses between the MZM and the 
driver chip were minimized by optimizing the interconnections to achieve the shortest possible electrical path. 
Therefore, both were co-packaged side-by-side on an aluminum-oxide ceramic, which not only reduced the signal 
W3E.6.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 2 -->
path length but also minimized the parasitic inductance introduced by the bonding wires. The ceramic submount was 
micro-machined to provide DC bias lines for both the MZM and the driver.  
3. Measurement Setup & Results 
 
Fig. 1: Measured and simulated EO-response of MZM chip. The dip at low frequencies is due to the impedance mismatch of the 85 Ω terminated 
MZM and the 50 Ω measurement equipment (left). Measured optical transfer function of the MZM (center). Simulated S21 characteristic of the 
20 dB linear differential driver (right). 
Initially, the modulator was characterized by electro-optical (EO) chip measurements performed with a 110 GHz 
lightwave component analyzer. As expected, the EO-response of the MZM shows a dip at lower frequencies (Fig. 1 
left), being attributed to the impedance mismatch between the 85 Ω termination of the MZM and the standard 50 Ω 
measurement instruments. The 3D- electromagnetic (EM) simulation result shows a flat response in lower frequencies 
with impedance of the MZM input and termination being perfectly matched. Therefore, normalizing the response 
beyond the dip (~10 GHz) is justified and yields an estimated 3 dB-bandwidth of 76 GHz and a Vπ = 3.1 V at 5 V bias 
voltage. Additionally, the MZM exhibits a DC ER of 18 dB as shown in Fig. 1 (center). 
The system-level measurement setup is illustrated in Fig. 2 (left). Measurements were performed at 40 °C under 
temperature control using a thermoelectric cooler (TEC). The 20 dB gain driver (Fig. 1 right) was contacted using a 
110 GHz differential RF probe in GSGSG configuration, while the DC bias lines were connected via a multi-contact 
probe card. The modulator subassembly was driven by a 2-channel 8-bit digital analog converter (DAC) to create a 
higher signal to noise ratio (SNR) for the differential drive signal. The DAC has 80 GHz of analog bandwidth and is 
operated at 260 GSa/s. A 1310 nm optical carrier was coupled from the fiber into the MZM. The modulated optical 
signal was detected by a 100 GHz photodetector (PD), being connected via a 1 mm DC-block to a 110 GHz real-time 
oscilloscope (RTO) for measurement. To control the received optical power at the PD, a praseodymium-doped fiber 
amplifier (PDFA) with optical filter and an optical attenuator were inserted between the transmitter and the PD. 
Furthermore, digital signal processing (DSP) with a digital root-raised cosine pulse-shaping filter with a roll-off factor 
of 1% was applied at both the transmitter and receiver side. Additionally, a digital feed-forward equalizer with 2048 
taps was used at the receiver side to compensate for electrical and optical distortions. 
        
   
Fig. 2: Schematic of the system measurement setup (left). A picture of the driver-modulator assembly (right).  
The input power of the optical carrier signal was kept constant at 15 dBm. The bit error rate (BER) was evaluated at 
various symbol rates as a function of the received optical power at the PD (Fig. 3 left). An optimum point at 8 dBm 
received optical power is observed due to saturation characteristic of the PD. Figure 3 (center and right) shows the 
post-equalized eye diagrams for 150 GBaud and 180 GBaud PAM4 transmission, respectively. At 150 GBaud, the 
eye openings remain well separated, and the BER is approximately two orders of magnitude below the 20 % overhead 
(OH) hard decision forward error correction (20 % HD-FEC) threshold [9]. At 180 GBaud, however, a reduction in 
vertical eye opening becomes evident. This degradation is primarily attributed to the limited EO-bandwidth of the 
0
20
40
60
80
100
120
−18
−15
−12
−9
−6
−3
0
3
 Measurement (norm. at 10 GHz)
 Simulation (impedance matched)
EOR [dB]
Frequency [GHz]
76 GHz
−2
−1
0
1
2
0
50
100
150
200
250
5V bias
Optical Power [mW]
Applied voltage [V]
Vp=3.4 V
ER = 18 dB
0
50
100
150
200
−30
−20
−10
0
10
20
30
Gain [dB]
Freq [GHz]
> 2 Vppd output 
3.5V DC bias
0.42 W max. power consumption
W3E.6.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 3 -->
MZM (76 GHz), which constrains the optical modulation amplitude. Additional bandwidth reduction of 
approximately 3 dB at 80 GHz results from RF losses in the probe and the cable assembly of the measurement setup. 
As a result, the available electrical bandwidth of the modulator engine reached its limit. After digital equalization, 
residual amplitude distortion and timing jitter remain visible, indicating that the DSP has reached its compensation 
capability under the given operating conditions. Nevertheless, a distinguishable eye structure confirms that successful 
symbol detection is achievable. In summary, the co-packaged modulator engine achieved 180-GBaud PAM4 
transmission with a BER below the 20 %-HD-FEC threshold under 0.4 Vppd driver excitation. The total electrical 
energy per bit of the modulator engine is 525 mW. Note that only the power consumption of the driver, MZM and RF 
signal were considered. The power consumption of the DSPs, the external laser and the TEC were excluded from this 
estimation. At a symbol rate of 180 GBaud the electrical energy per bit of the transmitter engine is 1.45 pJ/bit.  
 
 
 
Fig. 3: BER measurement results of the PAM4 modulation scheme at different symbol rates (left). Eye diagram at 150 GBaud (center) and 
180 GBaud (right). 
4. Conclusion 
A 76 GHz InP-based MZM with Vπ = 3.1 V was co-packaged with a 224 Gbaud-class differential linear EML driver. 
The modulator front-end achieved 180 GBaud PAM4 transmission with a BER below the 20%-HD-FEC threshold 
with a low electrical energy per bit of 1.45 pJ/bit. The co-packaged MZM engine showed competitive performance 
[1-5] in terms of high symbol rate for IM/DD transmission links in the O-Band. 
Acknowledgements 
This work was partly funded by the EU Project FLEX-SCALE (grant no. 101096909), and by the German Federal 
Ministry of Research, Technology and Space through the HYPERCORE (grant ID 16KIS2097K) and OPTIMUX 
(grant ID 16KIS2217) projects. 
References 
[1] S. Ohno et al., “226 Gbps PAM4 Operation Using Differential Drive EA-DFB Laser with 2.0-Vppd Swing over 10-km SSMF Transmission 
for 1.6TbE Transceivers”, 2025 OFC, San Francisco, CA, USA, M4K.6, doi: 10.1364/OFC.2025.M4K.6 
[2] Y. Ogiso et al., “Uncooled O-band InP MZ Modulator PIC for 3.2 Tb/s (400 Gb/s/lane) Pluggable Transceiver”, 2025 OFC, San Francisco, 
CA, USA, Th4D.1, doi: 10.1364/OFC.2025.Th4D.1 
[3] A. Rundquist et al., “8-Channel Monolithic InP Transmitter PIC Integrating DFB and MZM Arrays, Capable of Operating 106 GBd PAM4 at 
85 °C”, 2025 ECOC, Copenhagen, Denmark, Tu.04.03.2 
[4] X. Li et al., “Silicon Photonic Traveling Wave Mach-Zehnder Modulators for 400G/Lane Transmitters”, 2025 IEEE Silicon Photonics 
Conference (SiPhotonics), London, United Kingdom, TUA7, doi: 10.1109/SiPhotonics64386.2025.10985400. 
[5] C. St-Arnault et al., “Net 3.2 Tbps 225 Gbaud PAM4 O-Band IM/DD 2 km Transmission Using FR8 and DR8 with a CMOS 3 nm SerDes 
and TFLN Modulators”, 2025 OFC, San Francisco, CA, USA, Th4B.1, doi: 10.1364/OFC.2025.Th4B.1 
[6] M. Theurer et al., "200 Gb/s Uncooled EML with Single MQW Layer Stack Design," 2022 European Conference on Optical Communication 
(ECOC), Basel, Switzerland, 2022, Th1E.5, doi: 10.1364/ECEOC.2022.Th1E.5 
[7] H. Liu et al., “Ultra-High-Efficiency Dual-Band Thin-Film Lithium Niobate Modulator Incorporating Low-k Underfill with 220 GHz 
Extrapolated Bandwidth for 390 Gbit/s PAM8 Transmission”, 2025, Laser & Photonics Reviews. 19., doi:10.1002/lpor.202402053. 
[8] G. Chen, Z. Ruan, H. Wei, et al., “First Demonstration of Differential-Drive Push-Pull Modulators on Thin-Film Lithium Niobate.”, 
2025 Laser Photonics Rev 19, no. 19, doi: 10.1002/lpor.202500018 
[9] Guanbao Zhao et al., “Research on DC-Drift of TFLN modulator”, Proc. SPIE 13806, International Conference on Optoelectronic 
Information and Computer Engineering (OICE 2025), 1380602, doi: 10.1117/12.3087753 
[10] M. Rausch et al., "A performance comparison of single-ended- and differential driving scheme at 64 Gbit/s QPSK modulation for InP-based 
IQ-Mach-Zehnder modulators in serial-push-pull configuration," 2015 ECOC, Valencia, Spain, doi: 10.1109/ECOC.2015.7341728 
[11] J. Klamkin et al., "Indium Phosphide Photonic Integrated Circuits: Technology and Applications," 2018 BCICTS, San Diego, CA, USA, 
201, pp. 8-13, doi: 10.1109/BCICTS.2018.8550947 
0
2
4
6
8
10
10−5
10−4
10−3
10−2
10−1
100
 150 GBd
 160 GBd
 170 GBd
 180 GBd
 200 GBd
 1.4∙10-2
BER
Received optical power [dBm]
20% HD-FEC
W3E.6.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

