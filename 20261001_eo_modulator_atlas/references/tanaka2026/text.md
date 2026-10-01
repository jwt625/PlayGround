---
paper_id: tanaka2026
source_url: https://doi.org/10.1364/ofc.2026.m4d.2
doi: 10.1364/ofc.2026.m4d.2
license: publisher-copyright
sha256: 73a70c97862748b3ad06476128c12414a961a0f50dbb88fe5f2ef418bc473d15
pages: 3
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
IQ Modulators with Two Segments using  
InP/SOI Chip-on-Wafer Bonding Process for Optical DAC 
 
Hajime Tanaka1,2,3, Naotaka Kasuya1,2,3, Taichi Misawa2, Yohei Sobu1,4, Shuntaro Maeda5, 
 Kento Komatsu1,2,3, Naoko Inoue1,2,3, Takehiko Kikuchi1,2,3, Takuya Mitarai1,2,3, Shun Kimura1,2, 
Naoki Fujiwara1,2,3, Nobuhiko Nishiyama1,3, Takuo Tanemura5, Shinsuke Tanaka1,4, Hideki Yagi1,2,3  
1 Photonics Electronics Technology Research Association (PETRA) 1-20-10 Sekiguchi, Bunkyo-ku, Tokyo, 112-0014, Japan 
2 Sumitomo Electric Industries, Ltd., 1 Taya-cho, Sakae-ku, Yokohama-shi, Kanagawa 244-8588, Japan 
3 Dept. of Electrical and Electronic Engineering, Institute of Science Tokyo, 2-12-1, O-okayama, Meguro-ku, Tokyo, 152-8550, Japan 
4 1FINITY Inc., 4-1-1 Kamikodanaka, Nakahara-ku, Kawasaki-shi, Kanagawa 211-8588, Japan 
5 School of Engineering, The University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo 113-8656, Japan 
tanaka-hajime@sei.co.jp 
 
Abstract: We fabricated IQ modulators consisting of two segments using InP/Si chip-on-wafer 
bonding. Electrical/optical 3-dB bandwidth is approximately 60 GHz (V: 5.4 V). The symbol rate 
up to 64 GBaud is demonstrated in 16QAM operation using optical DAC. 
 
1. Introduction 
In recent years, the exponential growth in global data traffic has driven the demand for high-capacity, energy-
efficient, and compact optical components in optical fiber communication systems. To cope with these demands, 
photonic integrated circuits are a key technology, and various approaches, such as InP, thin-film lithium niobate 
(TFLN), and silicon photonics (SiPh), have been developed up to now. However, achieving the aforementioned 
requirements has become increasingly challenging, especially towards multi-Tb/s era, modulator devices based on 
single materials have tough issues. 
InP is well-suited for high-speed and high-efficiency modulation [1], utilizing its superior electro-optic (EO) 
properties. However, InP has limitations in achieving compact device layouts due to its large bending radius in 
passive waveguides. Hence it is not suitable for the high-density integration. On the other hand, SiPh offers the 
advantage of tight bending radii, enabling compact passive waveguide layouts for high-density integration. Despite 
this benefit, SiPh relies on the carrier plasma effect for modulation [2], resulting in a tradeoff between optical losses 
and modulator performance. The carrier plasma effect brings about the increase of optical losses with increasing 
carrier doping for high modulation efficiency and wide bandwidth operation. Hence, SiPh modulators are 
challenging to obtain high modulation efficiency and wide bandwidth operation for multi-terabit applications.   
To overcome these limitations, the heterogeneous integration platform on an SOI wafer has emerged as a 
promising solution, and high-speed, low-voltage MZMs utilizing EO polymers [3] and thin-film lithium niobate [4] 
have been reported as research efforts. Additionally, it has been revealed that the direct bonding process of InP chips 
onto SOI wafers [5,6] is very effective for fabricating wavelength-tunable lasers [7] and high-speed modulators [8]. 
This bonding technology enables high-density integration of multi-functional devices, including lasers and 
modulators, while leveraging the advantages of InP-based QW structures.  
In terms of EO signal conversion scheme to reduce power consumption, optical digital-to-analog converter (oDAC) 
is a promising approach [9]. The concept of oDAC is illustrated in Fig. 1. Conventionally, DACs and linear drivers 
were required to generate multi-level electrical signals in Fig. 1(a). However, the use of binary CMOS drivers 
allows to significantly reduce the power consumption in Fig. 1(b). By dividing the electrodes in the modulator and 
applying binary signals to each segment, multi-level optical signal generation becomes achievable. 
As shown in Fig. 1(b), the block diagram represents the left segments correspond to the most significant bit (MSB) 
and the other segments correspond to the least significant bit (LSB). The process of generating a composite 16QAM 
signal using MSB and LSB is illustrated in Fig. 2. By applying QPSK signals to the MSB and LSB, the modulator 
generates a composite 16QAM signal. In this configuration, the MSB performs intensity modulation with twice the 
strength of the LSB. Divided electrode design allows the achievement of both low half-wave voltage (Vπ) and wide 
EO bandwidth. SiPh-based oDAC implementations have demonstrated the symbol rates of up to 50 GBaud for 
16QAM signals [9]. However, further performance improvements can be realized by utilizing InP-based quantum 
well (QW) structures in the modulation section. 
In this study, we successfully fabricate the IQ modulator designed for oDAC applications through the InP/SOI 
chip-on-wafer (CoW) bonding process for the first time. This device utilizes the high modulation efficiency of InP 
QW with the miniaturization capability of SiPh, addressing the critical limitations of single-material platforms. 
Consequently we present 16QAM operation at 64 GBaud for an InP-based IQ modulator integrated onto a SiPh 
M4D.2.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 2 -->
DSP
DAC
LD
:Driver IC
Traveling-wave mod.
Digital      Analog
(a)
Q
I
DSP
(AWG)
LD
segmented mod.
Digital      Analog
(b)
Q-LSB
Q-MSB
I-MSB
I-LSB
:CMOS inverter driver
(Driver IC)
This demonstration used driver IC and AWG instead 
of CMOS inverter driver and DSP without DAC.
 
 
I
Q
Modulation by MSB
Modulation by LSB
00
10
11
01
0010
0000
0011
0001
00
10
11
01
MSB
LSB
(b)
Driver IC
IQ mod.
Q-LSB
Q-MSB
I-MSB
I-LSB
3.3 mm
3.8 mm
RF pads
Input
Output
Si
SiO2
Si substrate
InGaAsP MQW
n-InP
p-InP
SiO2
Au
20 m
Si WG
InP ridge WG
(a)
(d)
(c)
Modulator section
top view 
Modulator section
cross-sectional view 
TO phase
adjuster
Fig. 1: Transmitter with (a) 
conventional architecture and 
(b) optical DAC architecture. 
Fig. 
2: 
Example 
of 
16QAM modulation using 
MSB and LSB 
Fig. 3: (a) Photomicrograph of IQ modulator 
(mod.) with two segments using InP/SOI CoW 
bonding. (b) Photograph of IQ mod. with driver 
IC. (c) Schematic images of top and cross-
sectional views in modulator section. (d)  SEM 
image of optical transition structures between InP 
ridge and Si waveguides (WGs).  
platform compatible with oDAC technology assembling driver circuits. 
2.  Design and characteristic of IQ modulator with two segments using InP/SOI CoW bonding   
A photomicrograph of the fabricated IQ modulator comprising two segments, is shown in Fig. 3(a). The modulator 
consists of four RF modulation sections (I-LSB, I-MSB, Q-LSB, Q-MSB), which are based on the heterogeneously 
integrated structure utilizing the InP/SOI CoW bonding process. Phase-adjustment electrodes are implemented as 
heaters positioned along Si waveguides. Fig. 3(b) represents the IQ modulator implemented with driver circuits. The 
driver ICs amplify electrical signals to effectively drive each RF modulation sections. The schematic shown in Fig. 
3(c) illustrates the RF modulation sections, indicating both top and cross-sectional views. The modulation design 
incorporates capacitively loaded electrodes to optimize RF impedance adjustment. The filling factor (FF) 
configuration in RF electrodes provides flexibility in achieving desired RF impedance values. Furthermore, vertical 
optical confinement is achieved by the multiple quantum-well (MQW) core layer, while horizontal optical 
confinement is maintained by the upper InP ridge waveguide. Fig. 3(d) depicts a photograph of the taper structure 
for optical transition between the Si waveguide and InP ridge waveguide. Light transitions from the Si waveguide to 
the core layer of the InP waveguide through this taper structure. The taper configuration allows high-efficient optical 
transition through the precise formation achieved using the photolithography process with high positional accuracy 
after bonding [5,6]. Accordingly, this fabrication method minimizes optical coupling loss between Si waveguide and 
InP ridge sections. 
After bonding, ridge waveguides, termination resistors, and RF electrical lines were fabricated using the InP 
process including dry etching, wet-chemical etching, and lift-off process. Fig. 4 depicts the measured optical power 
as a function of heater power for phase adjustments. The heater is used as single-phase driving. The heating power 
for the rotation of π phase (Pπ) is 20 mW. From this extinction characteristic, an extinction ratio is as high as 30 dB, 
that is, the 1×2 Multi-Mode Interference (MMI) coupler evenly splits the optical power between two arms in the RF 
modulator section. Fig. 5 shows the differential DC modulation characteristics of the RF modulation section using 
the quantum-confined Stark effect (QCSE) from the InP-based MQW core layer. The modulator operates at a center 
bias voltage of -8 V, resulting in high modulation efficiency with a Vπ∙L of 0.43 V∙cm. Fig. 6 shows the EO 
response of the RF modulation sections. In Fig. 6(a), the EO characteristics of both LSB and MSB RF modulation 
sections are presented, and both sections exhibit an EO 3-dB bandwidth of approximately 60 GHz, indicating 
consistent performance across four MZ modulators. Fig. 6(b) shows the EO response after the integration of driver 
circuits. The Peaking of responses over 3 dB is observed between 40 and 50 GHz. Finally, the 16QAM operation 
with the symbol rate of 64 GBaud is achieved as the oDAC transmitter, and clear segregation among each symbol is 
successfully observed by an optical modulation analyzer as shown in Fig. 7. 
M4D.2.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 3 -->
0
20
40
60
80
100
−35
−30
−25
−20
−15
−10
−5
0
Normalized transmittance (dB)
Power consumption (mW)
P :20 mW
ER > 30 dB
−4 −3 −2 −1
0
1
2
3
4
−20
−15
−10
−5
0
Normalized transmittance (dB)
Vp_arm - Vn_arm (V)
VL ＝0.43 Vcm
Fig. 4: Measured normalized optical power as a function of 
power consumption in heater for phase adjustment. 
Fig. 5: Optical power versus voltage in the RF modulation 
section under push-pull operation, with center bias at -8 V. 
(An electrode length: 2 mm, FF: 60%). 
0
20
40
60
80
100
−12
−9
−6
−3
0
3
6
EO response (dB)
Frequency (GHz)
 I-LSB
 I-MSB
 Q-LSB
 Q-MSB
    
0
20
40
60
80
100
−12
−9
−6
−3
0
3
6
EO response (dB)
Frequency (GHz)
 I-LSB
 I-MSB
 Q-LSB
 Q-MSB
 
 
Fig. 6: Measurement result of EO response at V of 5.4 V (Electrode length: 1.0 mm, FF: 
80%). EO response of (a) IQ modulator chip and (b) IQ modulator with driver IC. 
Fig. 7 16QAM constellation 
of optical DAC transmitter 
at 64 GBaud. 
3. Conclusion 
We have demonstrated the IQ modulator composed of InP/SOI heterogeneously integrated RF sections using the 
CoW bonding method. The modulator has exhibited phase modulation characteristics with the extinction ratio 
exceeding 30 dB and high phase modulation efficiency of 0.43 Vπ∙cm. A wide EO 3-dB bandwidth of 
approximately 60 GHz has been achieved. After assembling driver ICs, the operation at the symbol rate of 64 
GBaud during 16QAM modulation has been demonstrated. From these results, it has verified the vast potential of 
our IQ modulator using optical DAC towards the realization of energy-efficient and high-speed optical transceivers. 
Acknowledgements: This work is based on results obtained from a project, JPNP16007, commissioned by the 
New Energy and Industrial Technology Development Organization (NEDO). A part of this work was supported by 
"Advanced Research Infrastructure for Materials and Nanotechnology in Japan (ARIM)" of the Ministry of 
Education, Culture, Sports, Science and Technology (MEXT). Proposal Number JPMXP1224IT0013. 
References  
[1] Y. Ogiso, and et al., “High Speed InP Modulator for Beyond 200 Gbaud”, Proc. OFC2024, paper Tu2D.7 (2024).  
[2] D. Dourado, and et al., “Challenges in silicon photonics modulators for data center interconnect applications”, Optics & Laser Technology, 
vol. 144, 107376 (2021). 
[3] H. Sato, and et al., "A 100 Gbaud On-off-Keying Silicon-Polymer Hybrid Modulator Operating at up to 110°C," IEEE Photonics Technol. 
Lett., vol.33, pp.1507-1510, Dec. 2021. 
[4] S. Mookherjea, “Hybrid Integration of Thin-film Lithium Niobate and Silicon Photonics,” Proc. of 2024 IEEE Silicon Photonics 
Conference, Tokyo Bay (Japan), April 2024, DOI: 10.1109/SiPhotonics60897.2024.10544046 
[5] T. Kikuchi, and et al., “Monolithic integration of various-type III-V epitaxial structures on silicon-photonics platform using chip-on-wafer 
hydrophilic bonding process,” Proc. of 2024 IEEE Silicon Photonics Conference, Tokyo Bay (Japan), April 2024, DOI: 
10.1109/SiPhotonics60897.2024.10543539.  
[6] T. Kikuchi, and et al., “III–V gain region/Si waveguide hybrid lasers with InP-based two-storied ridge structure by direct bonding 
technology,” Jpn. J. Appl. Phys., vol. 61, pp. 052002-1 - 052002-10, May 2022.  
[7] T. Hiratani, and et al., “InP/Si Heterogeneously Integrated Tunable Laser by Utilizing Chip-on-Wafer Bonding Method”, J. Light. Technol., 
vol. 43, no. 4, pp. 1708, (2025). 
[8] H. Tanaka, and et al., “InP/Si heterogeneously integrated p-i-n type Mach-Zehnder modulator by utilizing chip-on-wafer bonding method”, 
Proc. OECC2025, paper WE3-4 (2025).  
[9] Y. Sobu, and et al., “High-speed and low-power optical DAC transmitter using all-silicon lumped segmented modulator directly driven by 
CMOS inverter driver” in Proc. Opt. Fiber Commun. Conf. (OFC), Th2A.11, 2024 
M4D.2.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

