---
paper_id: li2026aa
source_url: https://doi.org/10.1038/s41467-025-67902-2
doi: 10.1038/s41467-025-67902-2
license: https://creativecommons.org/licenses/by-nc-nd/4.0
sha256: af047102a2b09024787a5aaa14fe15f45dd5059ace241a499a79539525a52fbd
pages: 9
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Article
https://doi.org/10.1038/s41467-025-67902-2
Ultra-broadband near- to mid-infrared
electro-optic modulator on thin-ﬁlm lithium
niobate
Qiyuan Li1,7, Qiyuan Yi1,7, Aolong Sun2,7, Chenglin Shang1, Yiqi Dai1, Sizhe Xing2,
Jinlai Cui3, Yupeng Zhu3, Jiayi Li3, Jun Zheng3
, Junwen Zhang
2
, Nan Chi
2,
An Pan1,4,7, Cheng Zeng1
, Jinsong Xia
1, Shuang Zheng1,5, Li Shen
1,5,6
&
Minming Zhang1,5
The escalating capacity limitations of conventional near-infrared tele-
communication bands have spurred urgent investigations into wide-band
optical communication systems spanning from the near-infrared to mid-
infrared regimes. This has motivated the development of optical components
combining broadband bandwidth with high-speed operation. The state-of-the-
art modulators face challenges in achieving broad operational bandwidth due
to waveguide dispersion and velocity mismatch. Here we demonstrate a thin-
ﬁlm lithium niobate (TFLN) electro-optic (EO) modulator with an unprece-
dented 800-nm operational bandwidth, covering the full O-U telecom bands
and extending into the 2-μm regime. The TFLN modulator exhibits >67 GHz EO
bandwidth across O-U bands (~100 GHz at O-/S-/C-/L-bands) and >50 GHz at
2-μm band. It enables single-lane exceeding 240 Gbps PAM-4 transmission
across O-U bands and a record 170 Gbps PAM-4 transmission at 2-μm band.
This breakthrough establishes TFLN as a compelling platform for multispectral
photonics, bridging conventional telecom infrastructure with emerging 2-μm
technologies for next-generation optical communications.
The relentless growth of big data, cloud computing, and artiﬁcial
intelligence has escalated global data trafﬁc to unprecedented levels,
challenging the capacity limits of conventional single-mode ﬁber
(SMF)-based optical communication systems1. To address this, two
complementary strategies have emerged: maximizing the utilization of
existing spectral resources and exploring new optical communication
windows beyond traditional O- and C-bands2,3. Recent advances in
hollow-core photonic bandgap ﬁbers (HC-PBGFs) have unveiled ultra-
broad low-loss transmission windows spanning 1240–1940 nm4–6,
covering the O-U bands and the newly proposed 2-μm band. At the
meantime, innovations in ampliﬁer technologies—ranging from rare-
earth-doped ﬁbers to the newly demonstrated integrated optical
parametric ampliﬁers—deliver broadband gain across these extended
spectral regions7–9. Together, these developments enable a transfor-
mative ultra-wide communication spectrum that seamlessly integrates
conventional telecom bands with the emerging 2-μm waveband.
Beyond revolutionizing optical networks, this expanded spectral range
also holds promise for applications such as quantum photonics, pre-
cision metrology, and biomedical imaging10–12. However, the realiza-
tion of end-to-end optical systems spanning these wavelengths
Received: 26 May 2025
Accepted: 11 December 2025
Check for updates
1Wuhan National Laboratory for Optoelectronics and School of Optical and Electronic Information, Huazhong University of Science and Technology,
Wuhan, China. 2Key Laboratory for Information Science of Electromagnetic Waves (MOE), Department of Communication Science and Engineering, Fudan
University, Shanghai, China. 3State Key Laboratory of Optoelectronic Materials and Devices, Institute of Semiconductors, Chinese Academy of Sciences,
Beijing, China. 4Wuhan ANPI Optoelectronics Company Ltd, Wuhan, China. 5Optics Valley Laboratory, Wuhan, China. 6Hubei Optical Fundamental Research
Center, Wuhan, China. 7These authors contributed equally: Qiyuan Li, Qiyuan Yi, Aolong Sun, An Pan.
e-mail: zhengjun@semi.ac.cn;
junwenzhang@fudan.edu.cn; zengchengwuli@hust.edu.cn; lishen@hust.edu.cn
Nature Communications|  (2026) 17:1138 
1
1234567890():,;
1234567890():,;

<!-- page 2 -->
remains hindered by a critical gap: the absence of high-speed broad-
band optical transmitters and photodetectors (PDs).
Integrated PDs for the 2-μm band have achieved bandwidths
exceeding 40 GHz, inherently supporting high-speed optical detection
across the O-U bands13–18. In contrast, the optical bandwidth of their
modulator counterparts remains constrained. Existing integrated
modulation platforms, such as silicon and germanium modulators
relying on free-carrier effects19–23, face inherent challenges at longer
wavelengths due to waveguide dispersion and carrier velocity mis-
match. These limitations conﬁne existing modulators to single-band
operations, hindering their adaptation to emerging multi-band or
ultrabroadband communication systems. Thin-ﬁlm lithium niobate
(TFLN) has emerged as a uniquely promising platform for optical
modulation, combining advantages such as ultra-broad optical trans-
parency (visible to mid-infrared spectrum), large electro-optic (EO)
coefﬁcients enabled by the Pockels effect, low optical loss, and com-
patibility with CMOS voltages enables modulators with unmatched
bandwidth and efﬁciency24–29. Recent milestones include TFLN-based
coherent transmission at 1.96 Tb/s30 and compact packaged modules31,
underscoring its potential for next-generation networks. Although
previous TFLN modulators have operated across discrete wavelengths
from visible to 2-μm bands24,32–35, achieving seamless ultrabroadband
operation for both the O-U telecom bands and the 2-μm bands—while
maintaining the high EO bandwidth—remains an unresolved challenge,
which is due to the intrinsic optical and EO bandwidth limitations of
conventional modulator designs.
Here, we co-design the passive components and modulation
electrodes and demonstrate a high-speed TFLN EO modulator with a
record-breaking continuous operational range of 1260–2060 nm,
seamlessly bridging the O-U bands and the 2-μm band. This broadband
capability is enabled by leveraging adiabatic mode evolution engi-
neering in critical passive optical components—such as optimized
optical splitters and spot-size converters (SSCs). By integrating the
broadband optical splitters and SSCs with high-performance traveling-
wave electrodes, our modulator achieves a 3-dB EO bandwidth
exceeding 67 GHz across the O-U bands and 50 GHz in the 2-μm band—
the highest reported for this spectral region. Furthermore, the mod-
ulator achieves a 3-dB EO bandwidth of ~100 GHz in the O-, S-, C-, and
L-bands (1310 nm, 1485 nm, 1550 nm, 1590 nm). The modulator exhi-
bits exceptional efﬁciency, with Vπ·L values of 1.92, 2.48, 2.61, 2.74, and
3.94 V·cm at 1310, 1485, 1550, 1590, and 2000 nm, respectively. For the
ﬁrst time with a single modulator, we demonstrate transmission in the
O-, E-, S-, C-, L-, U-, and 2-μm bands; data rates reached 170/170/170/
180/180/170/150 Gbps (On-Off Keying, OOK) and 260/260/260/280/
280/240/170 Gbps (4-level Pulse Amplitude Modulation, PAM-4),
respectively, with bit error rate (BER) all below the hard-decision for-
ward error correction (HD-FEC) threshold (3.8 × 10-3). This break-
through paves the way for universal optical transmitters capable of
unifying fragmented spectral resources into a single ultra-broadband
communication infrastructure, with transformative potential for next-
generation
optical
networks,
quantum
systems,
and
sensing
technologies.
Results
Design of the broadband TFLN modulator
As shown in Fig. 1a, the proposed extended full-spectrum optical
communication system leverages TFLN technology to realize broad-
band optical transmitters spanning the O-band to the 2-μm wavelength
range. This approach integrates the TFLN platform (NanoLN) with
existing multi-wavelength laser sources and multi-band optical
ampliﬁers. At the transmitter, a single full-spectrum TFLN EO mod-
ulator converts electrical signals—including OOK and advanced mod-
ulation formats such as PAM-4—into optical signals across the entire
spectral range from the O-band to the 2-μm band. The EOMs are fab-
ricated on a 300-nm-thick x-cut TFLN layer bonded to a 4.7-μm
thermally grown silicon dioxide substrate with a silicon base. As illu-
strated in the inset of Fig. 1a, the waveguide architecture features a 120-
nm slab height and a sidewall angle of θ = 60°. Figure 1b displays the
microscopy images of the fabricated EO modulator, which comprises
two 3-dB power splitters, a 9-mm-long EO Mach-Zehnder inter-
ferometer modulation section, and a thermo-optic phase shifter, and
two SSCs as the edge coupler for light coupling. The SEM images of an
SSC and a power splitter are shown in the inset of Fig. 1b.
To achieve ultra-broadband operating bandwidth for practical
optical packaging, the edge coupler was designed based on broadband
SSC, which is illustrated in Fig. 2a, b. The structure comprises LN bi-
layer tapers and a silicon oxynitride (SiON) ridge waveguide. At the
chip edge, the SiON waveguide interfaces with an ultra-high numerical-
aperture (UHNA4) ﬁber. The fundamental transverse electric (TE)
mode transitions adiabatically from the SiON waveguide to the LN
waveguide via tapered thickness modulation. The refractive index of
SiON (n = 1.54) lies between that of LN (n ≈2.2) and the silicon dioxide
buried oxide (SiO2 BOX, n ≈1.45), enabling optical conﬁnement within
the waveguide while suppressing leakage into the BOX. Achieving low
coupling loss (<1 dB) across the 1260-2060 nm spectral range requires
precise matching between the mode ﬁeld diameter (MFD) of the SiON
waveguide and the coupled ﬁber. The MFD of the UHNA4 ﬁber
increases with wavelength, for example, it is 3.3 µm at 1310 nm, 4.0 µm
at 1550 nm, and 5.3 µm at 2000 nm. For a SiON rib waveguide with a rib
height of 3.2 µm and slab height of 1 µm, the wavelength-dependent
coupling loss with varying the width of the SiON rib waveguide are
presented in Fig. 2c. Narrower rib widths minimize coupling loss at
shorter wavelengths (e.g., 1310 to 1480 nm), while broader widths
improve performance at longer wavelengths (e.g., 1680–2000 nm).
However, an anomalous variation in coupling loss occurs at 1310, 1410,
and 1480 nm when the rib width exceeds 4.8 µm, caused by resonant
ﬂuctuations from higher-order mode excitation. Speciﬁcally, increased
rib width enhances susceptibility to higher-order modes at shorter
wavelengths, inducing resonances that disrupt coupling efﬁciency.
Notably, at 1550 nm, the SSC exhibits consistent low coupling loss
across different rib widths. Below 1550 nm, coupling loss increases
with rib width, while the opposite trend is observed above 1550 nm. To
balance high coupling efﬁciency across O- to U-bands (with prioritized
performance in the C-band) and maintain acceptable loss at longer
wavelengths, the SiON rib width was optimized to 4.4 µm. This design
also demonstrates robust tolerance to dimensional variations in the
SiON waveguide cross-section.
The SSC parameters were optimized through a dual focus: max-
imizing the taper length to ensure adiabatic mode evolution and
minimizing the tip width to enhance coupling efﬁciency, while main-
taining fabrication feasibility. The design details of the SSC including
the inﬂuence of tip widths to the coupling loss are discussed in Sup-
plementary Note 1. The ﬁnalized SSC parameters are summarized in
Table 1. Figure 2d presents the calculated transmission spectrum
across 1200–2100 nm, with transmission efﬁciencies of 91.4%, 92.2%,
and 88.4% at 1310 nm, 1550 nm, and 2000 nm, respectively. Notably,
the simulation framework omitted intrinsic optical absorption losses
in plasma-deposited silicon oxynitride (SiOxNy). Minor spectral ripples
below 1500 nm originate from the reﬂection-induced interference in
the sudden height change of LN waveguide, and the intensity ﬂuc-
tuation of the higher-order modes in the SiON waveguide, as revealed
by mode ﬁeld simulations.
The full spectrum 3-dB power splitter is based on TFLN adiabatic
waveguide, with light is adiabatically coupled from the input wave-
guide to two symmetric output waveguides through three tapers. This
design leverages adiabatic coupling principles, enabling uniform 3-dB
power splitting over an ultrawide wavelength range provided that
adiabatic conditions are satisﬁed. We employ this broadband power-
splitting scheme into the TFLN modulator to construct the light
interference structure. Further design details of the 3-dB power splitter
Article
https://doi.org/10.1038/s41467-025-67902-2
Nature Communications|  (2026) 17:1138 
2

<!-- page 3 -->
are provided in ref.36. For shorter wavelengths (e.g., O-band), wave-
guides thicker than 300 nm impose stricter adiabatic mode evolution
requirements, which would degrade the 3-dB power splitter perfor-
mance. While the reduced electro-optic interaction at this thickness
increases the half-wave voltage, this trade-off was essential to achieve a
functional bandwidth spanning 1260–2050 nm. As shown in Fig. 2e,
the calculated transmission for each splitter is 48.2%, 49.6%, and 44.6%
at 1310, 1550, and 2000 nm, respectively, conﬁrming robust perfor-
mance across the targeted spectrum.
Outside the modulation region, the waveguides are engineered to
support single-mode operation at 1310 nm, ensuring broadband
compatibility across the O-/C- and 2-μm bands. This design suppresses
higher-order mode excitation over the 1260–2060 nm range, thereby
preventing EO performance degradation in the modulation region.
Finite-difference eigenmode (FDE) simulations indicate that single-
mode operation requires a waveguide top width of ~0.8 μm and
therefore the waveguide width was set as W = 0.8 μm in the regions of
SSC, power splitter, and bend waveguide. However, this narrow width
reduces optical mode conﬁnement, increasing propagation losses due
to stronger mode overlap with waveguide sidewalls and cladding,
particularly at longer wavelengths. Additionally, the constrained geo-
metry elevates the half-wave voltage requirement. To balance these
trade-offs, we implement an adiabatic taper proximal to the 3-dB
power splitter, expanding the waveguide width to W = 3.5 μm within
the modulation region. The details of the waveguide width optimiza-
tion are discussed in Supplementary Note 2. A FDE solver was
employed to numerically analyze the coupled electrical-optical modal
properties of the device architecture. The electrode gap (G) is deﬁned
as the horizontal separation between the ground plane and the signal
electrode. As shown in Fig. 2f, we calculated the half-wave voltage-
length product (Vπ·L) and quantiﬁed the metal-induced absorption
loss, both parameterized as functions of G at wavelengths of 1310,
1550, and 2000 nm. The results reveal an inverse relationship between
device modulation efﬁciency (Vπ·L) and optical loss performance,
necessitating tailored selection of G to prioritize either metric for
speciﬁc applications. For this work, an electrode gap of G = 6 μm was
chosen to achieve an optimal trade-off point, simultaneously ensuring
strong modulation performance while limiting absorption loss to
moderate levels.
The traveling-wave electrodes feature a push-pull conﬁguration
based on a coplanar waveguide (CPW) structure. The signal and
ground electrode widths are designed as WS = 25 μm and WG = 150 μm,
respectively, with gold (Au) electrodes of height H = 1 μm and a 2-μm-
thick SiO2 cladding layer. The characteristic impedance (Z0) of the
CPW is approximately 42 Ω, as illustrated in the inset of Fig. 2g. A load
resistor of ZL = 38 Ω is used, which is intentionally lower than Z0 to
mitigate the low-frequency EO S21 roll-off, thereby slightly enhancing
the operational bandwidth37. Design details regarding the traveling-
wave electrode impedance can be found in Supplementary Note 3. The
optical group indices are 2.18, 2.13, and 2.04 at wavelengths (λ) of
1310 nm, 1550 nm, and 2000 nm, respectively (dashed lines in Fig. 2g).
The optical group index at the C-band is aligned with the radio fre-
quency (RF) effective index, resulting in a trade-off in EO bandwidth
performance for the 2-μm band. The RF effective index is designed as
2.13 at 50 GHz. For λ = 2000 nm, the residual index mismatch leads to a
theoretical 3-dB bandwidth of approximately 80 GHz for a 9-mm-long,
TFLN Modulator
Full-Spectrum Optical Modulator
Laser
Modulator
Amplifier
PD
Multi-wavelength Laser Source
Wide-band Optical Fibre
Multi-band Optical Amplifier
E/O Module
O/E Module
Edge Coupler
Phase Shifter
9 mm
Power Splitter
10 μm
50 μm
G
S
G
G
S
G
…
U-band
2-μm band
O-band
E-band
S-band
C-band
L-band
WS
Cladding
BOX
Si Substrate
Au
LiNbO3
W
WG
G
H
Au
OOK/PAM-4
Signal Formats
a
b
5 μm
400
760 1260 1360 1460 1530 1565 1625
1890 2060
1675
(nm)
Visible Light
Near Infrared
Mid Infrared
3000
Wavelength
Fig. 1 | Full-spectrum optical modulator. a Concept of extended full spectrum
optical communication through broadband thin-ﬁlmlithium niobate (TFLN) optical
modulator, multi-wavelength laser sources, multi-band optical ampliﬁers, wide-
band optical ﬁber and PDs. Inset: cross-sectional view of the modulator region.
b Microscopy images of the fabricated LN Mach-Zehnder electro-optic modulator
(EOM). Inset: SEM images of the spot-size converter (SSC) and power splitter.
Article
https://doi.org/10.1038/s41467-025-67902-2
Nature Communications|  (2026) 17:1138 
3

<!-- page 4 -->
impedance-matched, lossless modulator. Design details of the broad
bandwidth traveling-wave electrodes can be found in Supplementary
Note 3. The thermo-optic (TO) phase shifter is employed to establish
the optimal operating bias point. By adopting TO tuning instead of EO
tuning via bias tees on the traveling-wave electrodes, this approach not
only simpliﬁes packaging and testing procedures but also signiﬁcantly
improves the stability of the bias point by avoiding the LN DC bias drift
problem caused by the DC bias voltage under EO bias conditions.
Detailed design parameters and experimental results of the TO phase
shifter used in our modulator can be found in supplementary Note 4.
Measurement and analysis
The fabrication details of TFLN modulator is given in “Methods”. The
coupling losses of the SSC were characterized using a reference
waveguide coupled to a UHNA4 ﬁber. Refractive index-matching oil
(index = 1.46) was applied at the ﬁber-chip interface to minimize
reﬂections. Coupling losses were measured using tunable lasers for the
1260–1680 nm bands, and a custom-built 2-μm ampliﬁed spontaneous
emission (ASE) source paired with an optical spectrum analyzer
(Yokogawa AQ6375E) for the 1920–2060 nm range. Using a polariza-
tion controller (PC), the measured TE mode coupling losses are
0.78 dB/facet at 1310 nm, 1.18 dB/facet at 1450 nm, 1.73 dB/facet at
1485 nm, 0.69 dB/facet at 1550 nm, 0.56 dB/facet at 1590 nm, 0.69 dB/
facet at 1653 nm, 0.82 dB/facet at 1970 nm, and 2.21 dB/facet at
2000 nm, as shown in Fig. 3a. The slightly higher losses at shorter and
longer wavelengths are attributed to minor mode mismatch between
the UHNA4 ﬁber and the SiON waveguide. Notably, three pronounced
loss peaks near 1510 nm and 2000 nm correlate with intrinsic N-H
infrared absorption bands in the plasma-deposited SiON layer, con-
sistent with prior reports of N-H bond absorption38–41. The slight loss
rise of 1370-1410 nm can be attributed to the O-H bond during the
plasma deposition of SiON layer42. The normalized transmission
spectra
of
the
TFLN
modulator
for
the
1260–1680 nm
and
1920–2060 nm spectral region can be found in Supplementary Note 5.
Following characterization of the modulator’s total optical losses, the
on-chip insertion losses (ILs) were deduced as 1.2 dB, 2.8 dB, and 5.8 dB
at wavelengths of 1310 nm, 1550 nm, and 1970 nm, respectively. A test
structure comprising 16 cascaded 3-dB power splitters was used to
measure the ILs of individual splitters, yielding values of 0.23 dB,
0.11 dB, and 0.42 dB at 1310 nm, 1550 nm, and 1970 nm, closely
matching simulations. After accounting for contributions from the
power splitters and waveguide propagation losses, the primary source
Table 1 | Parameters of the SSC
Parameter
w1
w2
w3
w4
w5
w6
s1
s2
s3
h1
h2
Value (μm)
0.2
0.2
2.1
0.8
5
4.4
250
150
100
3.2
1
w1 to w6 are the widths, s1 to s3 are the lengths of the tapered waveguides, h1 to h2 are the heights of SiON layers, as shown in Fig.2a, b.
0
20
40
60
80
100
2.0
2.1
2.2
2.3
2.4
2.5
2.6
2.7
RF nm
Frequency (GHz)
3.8
4
4.2
4.4
4.6
4.8
5
0.3
0.35
0.4
0.45
0.5
0.55
0.6
λ = 1310 nm
λ = 1410 nm
λ = 1480 nm
λ = 1550 nm
Loss (dB/facet)
SiON rib width (μm)
λ = 1610 nm
λ = 1680 nm
λ = 1780 nm
λ = 1890 nm
λ = 2000 nm
0
20
40
60
80
100
40
44
48
52
Z0 (Ω)
Frequency (GHz)
5
6
7
8
1
2
3
4
5
6
λ = 2000 nm
λ = 1550 nm
G (μm)
VπL (V·cm)
λ = 1310 nm
0
10
20
30
40
50
60
Loss (dB/cm)
1200
1400
1600
1800
2000
0.86
0.88
0.9
0.92
0.94
Transmission (a.u.)
Wavelength (nm)
1200
1400
1600
1800
2000
0.4
0.42
0.44
0.46
0.48
0.5
Transmission (a.u.)
Wavelength (nm)
2-μm band
e
f
g
c
d
O
E
S C L U
2-μm band
O
E
S C L
U
ng=2.18@1310nm
ng=2.13@1550nm
ng=2.04@2000nm
w1
b
w2
w3
w4
w5
s3
s2
s1
a
SiON
LN
BOX
w6
h1
h2
Fig. 2 | Design of the thin-ﬁlm lithium niobate (TFLN) modulator. a Schematic
structure of the spot-size converter (SSC). The inset shows the dimension para-
meters of the SiON waveguide. b Dimension parameters of the LN bi-layer tapers.
c Dependence of the coupling loss on SiON rib width. d Simulated transmission
spectrum of the SSC. e Simulated transmission spectrum of the 3-dB power splitter.
f Simulated Vπ·L and metal absorption loss as a function of G. g Simulated
frequency-dependent results of radio frequency (RF) effective index nm. The three
dashed lines correspond to the optical group index ng. The inset shows the simu-
lated characteristic impedance Z0.
Article
https://doi.org/10.1038/s41467-025-67902-2
Nature Communications|  (2026) 17:1138 
4

<!-- page 5 -->
of on-chip loss was identiﬁed as absorption by the metal electrodes,
which becomes more pronounced at longer wavelengths (e.g.,
1970 nm). Detailed loss analysis of the fabricated modulator can be
found in Supplementary Note 6. Experimental on-chip losses for a gap
spacing (G = 6 μm) show strong agreement with simulated values, as
illustrated by the dashed lines in Fig. 2f.
Figure 3b illustrates the half-wave voltage Vπ measurements with a
100 kHz triangular voltage sweep. With a modulator length of 9 mm,
the Vπ·L values are calculated to be 1.92/2.39/2.48/2.61/2.74/2.89/3.90/
3.94 V·cm
for
λ = 1310/1450/1485/1550/1590/1653/1970/2000
nm,
respectively. The simulated and measured Vπ·L parameters across
these eight wavelengths can be found in Supplementary Note 7.
Notably, the experimental Vπ·L and loss are consistent with the simu-
lation results in Fig. 2f for G = 6 μm, validating the accuracy of our
numerical model. This agreement conﬁrms the reliability of our
simulations for predicting performance trends across varying values of
electrode gap. For instance, the simulations suggest that prioritizing
reduced metal absorption loss over minimizing Vπ can be achieved by
increasing G slightly to 7 μm would maintain comparable high-
frequency EO performance. Additionally, static extinction ratios (ERs)
were measured by sweeping the heater voltage, yielding values of ~17,
~15, and ~18 dB at 1310, 1550, and 1970 nm, respectively. The observed
wavelength-dependent ER variations primarily originate from power
imbalance between the output ports of the 3-dB splitter, attributed to
fabrication tolerances.
The EO responses of the fabricated TFLN modulator were char-
acterized using an 110-GHz lightwave component analyzer (LCA,
Ceyear 6433 P) in conjunction with a tunable laser (Santec TSL-570). A
110-GHz high-speed RF probe was employed to deliver RF signals from
the LCA, while a 38-Ω load resistor was attached at the end of the
traveling-wave electrode. As shown in Fig. 3b, the 3-dB bandwidths of
~100 GHz were observed at wavelengths of 1310 nm (O-band), 1485 nm
(S-band), 1550 nm (C-band), and 1590 nm (L-band). Due to the lack of
relevant wavebands detection modules in the LCA, we utilized a 67-
GHz vector network analyzer (VNA, Keysight N5247B) to measure the
EO S21 response. The EO response was captured by feeding electrical
signals from the PDs into the VNA. The measured EO response curves
for the E- (1450 nm) and U-bands (1653 nm) using the commercial PD
revealed that the modulator’s 3-dB bandwidth exceeded 67 GHz, lim-
ited by the VNA’s bandwidth. For the 2-μm band, where commercial
PDs lack sufﬁcient bandwidth, we employed a recently demonstrated
high-speed GeSn PD (bandwidth >40 GHz, ref. 16) to measure the
bandwidth. The details of GeSn PD and the EO frequency responses for
the 2-μm band measurement are shown in Supplementary Note 8.
After system calibration and PD-response de-embedding, character-
ization of the frequency response and high-speed data transmission
capabilities was achieved. The measured EO S21 response at λ = 2 μm
reveals a 3-dB bandwidth of >50 GHz, the highest reported to date for
optical modulators operating in the 2-μm band. As further indicated by
simulations (Fig. 3b, dashed lines), this bandwidth could be extended
to ~80 GHz. The low-frequency peaks observed in both the simulated
and measured responses stem from the deliberate impedance con-
ﬁguration (ZL < Z0), which was implemented to increase bandwidth by
compensating the drop-off at low frequencies37.
To evaluate the high-speed data transmission performance of the
TFLN modulator, the device was optically packaged to UHNA4 ﬁbers
and characterized across all operational wavelength regions: the
O-band (1310 nm), E-band (1450 nm), S-band (1485 nm), C-band
(1550 nm), L-band (1590 nm), U-band (1653 nm), and 2-μm band
(1970/2000 nm). The selected wavelengths in each band were chosen
to align with current and potential communication applications, as
well as the available laser sources and ampliﬁers. Figure 4a illustrates
the experimental setup for measuring eye diagrams and BER. Optical
signals were generated using various laser sources at 1310 nm,
1450 nm, 1485 nm, 1550 nm, 1590 nm, 1653 nm, 1970 nm, and 2000
nm. High-frequency RF signals, synthesized by a 256 GSa/s arbitrary
waveform generator (AWG, Keysight M8199B), were ampliﬁed to
~2.4 Vpp using an RF ampliﬁer and applied to the modulator. A pre-
equalizer was implemented to preprocess signals prior to loading
them into the AWG, thereby compensating for bandwidth limitations
-3 -2 -1
0
1
2
0
1
Transmission (a.u.)
Voltage (V)
0
20
40
60
80
100
-10
-8
-6
-4
-2
0
EO S21 (dB)
Frequency (GHz)
Experiment
Simulated
0
20
40
60
80
100
-10
-8
-6
-4
-2
0
EO S21 (dB)
Frequency (GHz)
Experiment
Simulated
-1
0
1
2
0
1
Transmission (a.u.)
Voltage (V)
0
20
40
60
80
100
-10
-8
-6
-4
-2
0
EO S21 (dB)
Frequency (GHz)
Experiment
Simulated
0
20
40
60
80
100
-10
-8
-6
-4
-2
0
EO S21 (dB)
Frequency (GHz)
Experiment
Simulated
0
20
40
60
80
100
-10
-8
-6
-4
-2
0
EO S21 (dB)
Frequency (GHz)
Experiment
Simulated
0
10 20 30 40 50 60 70 80
-10
-8
-6
-4
-2
0
EO S21 (dB)
Frequency (GHz)
Experiment
Simulated
0
10 20 30 40 50 60 70 80
-10
-8
-6
-4
-2
0
EO S21 (dB)
Frequency (GHz)
Experiment
Simulated
1300
1350
1400
1450
1500
1550
1600
1650
1950
2000
2050
0
1
2
3
4
5
Loss (dB/facet)
Wavelength (nm)
O
E
S
C
L
U
2-μm band
a
b
0
20
40
60
80
100
-10
-8
-6
-4
-2
0
EO S21 (dB)
Frequency (GHz)
Experiment
Simulated
1310 nm
-3 dB
Vπ = 2.13 V
1450 nm
1550 nm
1485 nm
1590 nm
1653 nm
2000 nm
1970 nm
-2
-1
0
1
2
0
1
Transmission (a.u.)
Voltage (V)
-3 dB
-3 dB
-3 dB
-3 dB
-3 dB
-3 dB
-3 dB
PD Limited Region
PD Limited Region
-2
-1
0
1
2
0
1
Transmission (a.u.)
Voltage (V)
-1
0
1
2
3
0
1
Transmission (a.u.)
Voltage (V)
-2
-1
0
1
0
1
Transmission (a.u.)
Voltage (V)
-1 0 1 2 3 4 5
0
1
Transmission (a.u.)
Voltage (V)
-2 -1 0 1 2 3 4
0
1
Transmission (a.u.)
Voltage (V)
Vπ = 2.66 V
Vπ = 2.75 V
Vπ = 2.90 V
Vπ = 3.04 V
Vπ = 3.21 V
Vπ = 4.33 V
Vπ = 4.38 V
Fig. 3 | Measured performance of the thin-ﬁlm lithium niobate (TFLN) mod-
ulator. a Measured coupling loss of the spot-size converter (SSC) for the
1260–1680 nm and 1920–2060 nm range. b Measured electro-optic (EO) S21
responses of the TFLN modulator. The dash lines indicate the simulated band-
widths. The insets show normalized optical transmissions with measured half-wave
voltage Vπ.
Article
https://doi.org/10.1038/s41467-025-67902-2
Nature Communications|  (2026) 17:1138 
5

<!-- page 6 -->
of the system. The bias point of the modulator was stabilized via an
integrated
heater.
The
modulated
light
was
ampliﬁed
using
wavelength-speciﬁc
ampliﬁers:
semiconductor
optical
ampliﬁers
(SOAs) for the O-, E-, S-, and U-bands; two erbium-doped ﬁber ampli-
ﬁers (EDFAs) for the C- and L-bands; and a thulium-doped ﬁber
ampliﬁer (TDFA) for the 2-μm band. The ampliﬁed light was then
coupled into a 100-GHz-bandwidth PD (Finisar XPDV4121R) for
wavelengths up to the U-band, and a GeSn PD (previously character-
ized in prior work) for the 2-μm band. Received signals were captured
by a high-speed real-time oscilloscope (Keysight UXR0594BP), fea-
turing a 59-GHz bandwidth and 256 GSa/s sampling rate. The BER can
be calculated after a series of ofﬂine digital signal processing (DSP). A
Volterra
nonlinear
equalizer
(VNLE)
and
maximum
likelihood
sequence equalization (MLSE) algorithm was jointly employed to
Fig. 4 | Data transmission results. a Experimental setup for measuring the eye
diagrams and the calculated eye diagrams for On-Off Keying (OOK) signals.
b Measured bit error rate (BER) curves versus the received optical power of four-
level pulse amplitude modulation (PAM-4) signals for λ = 1310/1450/1485/1550/
1590/1653/1970/2000 nm. The insets show the calculated eye diagrams. c The BER
curves with increasing data rates of different wavelengths with OOK and PAM-4
signals. d The maximum data rates of OOK/PAM-4 signals with BERs under the hard-
decision forward error correction (HD-FEC) threshold operated in the ultra-wide
wavebands. AWG arbitrary waveform generator, PC polarization controller, RF
Amp. RF ampliﬁer, Opt. Amp. optical ampliﬁer, PD photodetector.
Article
https://doi.org/10.1038/s41467-025-67902-2
Nature Communications|  (2026) 17:1138 
6

<!-- page 7 -->
mitigate both linear and nonlinear signal impairments. Detailed spe-
ciﬁcations of the equipment used in high-speed data transmission
measurements are provided in Supplementary Note 9.
The inset of Fig. 4a displays measured eye diagrams and corre-
sponding BERs for 160 Gbps OOK signals across the O-, E-, S-, C-, L-, and
U-bands, as well as 130 Gbps OOK signals in the 2-μm band. Clear eye
openings were observed for all the wavelengths, respectively. To our
knowledge, this represents the highest OOK transmission rate in the
2-μm band and the ﬁrst TFLN modulator capable of exceeding
170 Gbps OOK operation across all O-U bands. We further evaluated
PAM-4 performance by measuring BER curves as a function of received
optical power (ROP), deﬁned as the optical power incident on the PD. A
variable optical attenuator (VOA) was inserted after the optical
ampliﬁers to adjust ROP. Figure 4b shows back-to-back (B2B) BER
curves for the TFLN modulator. For O-U bands, transmission results
through a 500-m standard SMF were also included and exhibit negli-
gible power penalties compared to the B2B case. To further explain the
data rates of single-lane transmission, we analyze the BER performance
of OOK and PAM-4signals for all eight wavelengths ranging from 120 to
300 Gbps (see Fig. 4c). Under the same HD-FEC threshold (3.8 × 10−3),
we successfully realize the transmission of 260/260/260/280/280/240/
170/150 Gbps PAM-4 signals were obtained at 1310/1450/1484/1550/
1590/1653/1970/2000 nm. Figure 4d shows the maximum data rates
achieved using OOK/PAM-4 signals while maintaining a BER below the
HD-FEC threshold for all eight wavelengths. This represents the ﬁrst
experimental validation of full-spectrum optical communications (O to
U bands) with single-lane data rates exceeding 240 Gbps (PAM-4),
enabled by the modulator’s ultra-wide wavelength operability. Fur-
thermore, the achieved single-lane 170 Gbps transmission in the 2-μm
band signiﬁcantly surpasses the previously reported record for this
spectral region22. Further discussions on data transmission measure-
ments for various wavebands can be found in Supplementary Note 10.
Discussion
Figure 5 compares the 3-dB EO bandwidths of integrated modulators
in the short-wave infrared spectrum regime from 1.2 to 2.1 μm43–47.
Our TFLN modulator operates across an unprecedented 800-nm
optical bandwidth, spanning the O-U bands (1260–1680 nm), and
extending into the 2-μm regime (1920–2060 nm). This broad
operational range enables multi-band compatibility within a single
device, a critical advantage for versatile photonic systems. The TFLN
modulator maintains a ﬂat frequency response across its operational
range. The measured EO bandwidths of the modulator reach
~100 GHz at critical communication wavelengths of 1310 nm (O-
band), 1485 nm (S-band), 1550 nm (C-band), and 1590 nm (L-band),
which
are
comparable
with
state-of-the-art
performance
of
single-waveband-optimized
integrated
modulators
reported
in
literature25,34,35,44,45. The 50-GHz bandwidth at 2000 nm enables, to
our knowledge, demonstration of the highest single-lane 170 Gbps
transmission data transmission in this band. This represents a 2.3-
fold improvement in bandwidth over existing 2-μm modulators,
directly enhancing data capacity and underscoring the potential for
next-generation high-speed communication systems. The detailed
comparisons of the state-of-the-art integrated EO modulators oper-
ating in the 2-μm spectral band are given in “Methods”.
In conclusion, we demonstrate a TFLN modulator featuring
high EO bandwidth and a record-breaking operational range span-
ning over 800 nm, from the O-band to the 2-μm spectral region. The
device achieves 3-dB EO bandwidths of over 67 GHz in the O- to
U-bands and 50 GHz in the 2-μm band. Speciﬁcally, the modulator
achieves a 3-dB EO bandwidth of ~100 GHz at 1310 nm (O-band),
1485 nm (S-band), 1550 nm (C-band), and 1590 nm (L-band). High-
speed data transmissions of over 170 Gbps OOK signals and
240 Gbps PAM-4 signals were experimentally validated at all O-U
bands. Furthermore, 150 Gbps OOK signals and 170 Gbps PAM-4
signals were also demonstrated for the 2-μm band. The BERs are all
below the HD-FEC threshold of 3.8 × 10-3. These results underscore
the transformative potential of TFLN modulators in enabling
ultrabroadband optical communication systems that seamlessly
bridge conventional telecom bands with the emerging 2-μm win-
dow. This advancement directly addresses the escalating band-
width
demands
of
next-generation
data
centers
and
high-
performance computing infrastructures.
Methods
Fabrication of the TFLN modulator
The fabrication process of the TFLN modulator is outlined as follows:
First, electron beam lithography (EBL) was used to deﬁne ridge
waveguide structures on a 700-nm-thick layer of AR-P 6200 resist. The
ridge waveguide was then formed by etching 180 nm of lithium nio-
bate (LN) using Ar⁺-based inductively coupled plasma (ICP) dry etch-
ing. A 2-μm-thick silica cladding layer was deposited over the
waveguide via plasma-enhanced chemical vapor deposition (PECVD)
to encapsulate the modulation section. This silica layer was selectively
etched using ICP dry etching in the electrode and SSC regions. To
create the LN bi-layer tapers, the EBL and ICP etching steps were
repeated to remove the remaining 120 nm of LN. A 200-nm-thick
thermal
titanium
(Ti)
layer
was
deposited
by
electron-beam
1300
1400
1500
1600
1900
2000
20
40
60
80
100
3-dB Bandwidth (GHz)
Wavelength (nm)
O
E
S
C
L
U
2-μm band
[34]
[43]
[44]
[35, 45]
[25, 34]
[46]
[47]
[43]
[19]
[19]
[21]
[24]
This work
This work
Si
TFLN
This Work
Fig. 5 | Comparisons of integrated modulators. Comparison of 3-dB bandwidth of
short-wave infrared high-speed electro-optic (EO) modulators.
Table 2 | Performance comparison of on-chip integrated EO modulators operating at 2-μm band
Ref.
Platform
Structure
Extinction Ratio (dB)
3-dB Bandwidth (GHz)
Max. Baud Rate and Signal Format
Other wavebands capable
19
SOI
MZI
-
10
20 Gbaud OOK
C-band
20
SOI
Michelson
15
-
20 Gbaud OOK 15 Gbaud PAM-4
No
24
LNOI
MZI
20
22
32 Gbaud OOK
No
21
SOI
MZI
22
18
30 Gbaud OOK 40 Gbaud PAM-4
No
22
SOI
MRM
19
18
50 Gbaud OOK
No
23
SOI
Racetrack Ring
21
26
34 Gbaud OOK
No
This work
LNOI
MZI
18
>50
150 Gbaud OOK 85 Gbaud PAM-4
O-U bands
Article
https://doi.org/10.1038/s41467-025-67902-2
Nature Communications|  (2026) 17:1138 
7

<!-- page 8 -->
evaporation (EBE), after which the heater pattern was deﬁned via EBL.
A 1-μm-thick Au traveling-wave electrode was then formed by EBE
deposition and lift-off. Next, a 4.2-μm-thick SiON layer was grown
across the entire chip using PECVD with silane, ammonia, and nitrous
oxide precursors. The SiON layer was patterned and etched twice to
form the SiON ridge waveguide in the SSC region, while the SiON
above the modulation section was removed. A 2-μm-thick protective
SiO2 layer was deposited by PECVD to shield the SSC and minimize
contamination-induced losses in the modulator. Finally, the cladding
thickness in the modulation section was reduced to 2 μm via ICP
etching, and the electrode pads were exposed.
Comparisons of the 2-μm integrated EO modulators
Table 2 summarizes the performance metrics of integrated electro-
optic modulators operating in the 2-μm spectral band. Our TFLN
modulator achieves a > 50 GHz EO 3-dB bandwidth—the highest
reported value in this spectral region—enabling a baud rate of 85
Gbaud (170 Gbps for PAM-4) and surpassing prior state-of-the-art
devices. This advancement establishes a new benchmark for high-
speed data transmission in the 2-μm band.
Data availability
The raw data generated in this study have been deposited in the Fig-
share database under accession code https://doi.org/10.6084/m9.
ﬁgshare.30618413.
References
1.
Mizuno, T. & Miyamoto, Y. High-capacity dense space division
multiplexing transmission. Opt. Fiber Technol. 35, 108–117 (2017).
2.
Soref, R. Enabling 2 -μm communications. Nat. Photonics 9,
358–359 (2015).
3.
Liu, Z. et al. High-capacity directly modulated optical transmitter for
2-μm spectral region. J. Lightwave Technol. 33, 1373–1379 (2015).
4.
Sakr, H. et al. Interband short reach data transmission in ultrawide
bandwidth hollow core ﬁber. J. Lightwave Technol. 38,
159–165 (2020).
5.
Roberts, P. J. et al. Ultimate low loss of hollowcore photonic crystal
ﬁbres. Opt. Express 13, 236–244 (2005).
6.
Petrovich, M. et al. Broadband optical ﬁbre with an attenuation
lower than 0.1 decibel per kilometre. Nat. Photonics 19, 1203 (2025).
7.
Li, Z. et al. Diode-pumped wideband thuliumdoped ﬁber ampliﬁers
for optical communications in the 1800-2050 nm window. Opt.
Express 21, 26450–26455 (2013).
8.
Zhao, P. et al. Ultra-broadband optical ampliﬁcation using nonlinear
integrated waveguides. Nature 640, 918–923 (2025).
9.
Kuznetsov, N. et al. An ultra-broadband photonic-chip-based
parametric ampliﬁer. Nature 639, 928–934 (2025).
10.
Bruns, T. et al. Next-generation in vivo optical imaging with short-
wave infrared quantum dots. Nat. Biomed. Eng. 1, 0056 (2017).
11.
Carlson, D. R. et al. Ultrafast electro-optic light with subcycle con-
trol. Science 361, 1358 (2018).
12.
Karpiński, M., Jachura, M., Wright, L. J. & Smith, B. J. Bandwidth
manipulation of quantum light by an electro-optic time lens. Nat.
Photonics 11, 53 (2017).
13.
Ackert, J. J. et al. High-speed detection at two micrometres with
monolithic silicon photodiodes. Nat. Photonics 9, 393–396 (2015).
14.
Chen, Y., Xie, Z., Huang, J., Deng, Z. & Chen, B. High-speed uni-
traveling carrier photodiode for 2 μm wavelength application.
Optica 6, 884–889 (2019).
15.
Jones, A. H., March, S. D., Bank, S. R. & Campbell, J. C. Low-noise
high-temperature AlInAsSb/GaSb avalanche photodiodes for 2-μm
applications. Nat. Photonics 14, 559–563 (2020).
16.
Cui, J. et al. High-speed GeSn resonance cavity enhanced photo-
detectors for a 50 Gbps Si-based 2 μm band communication sys-
tem. Photon. Res. 12, 767–773 (2024).
17.
Zhu, Y. et al. 112 Gbps CMOS-compatible waveguide germanium
photodetector for the 2 μm wavelength band with a 3.64 A/W
responsivity. Photon. Res. 12, 2373–2378 (2024).
18.
Wang, J. et al, 112 Gb s−1 germanium photodetector at 2 µm enabled
by a 3D integrated waveguide loop. Adv. Mater. Technol.
2401718 (2025).
19.
Cao, W. et al. High-speed silicon modulators for the 2 μm wave-
length band. Optica 5, 1055–1062 (2018).
20. Cao, W. et al. High-speed silicon Michelson interferometer mod-
ulator and streamlined IMDD PAM-4 transmission of Mach-Zehnder
modulators for the 2 μm wavelength band. Opt. Express 29,
14438–14451 (2021).
21.
Wang, X. et al. High-speed silicon photonic Mach-Zehnder mod-
ulator at 2 μm. Photon. Res. 9, 535–540 (2021).
22. Shen, W. et al. High-speed silicon microring modulator at the 2 µm
waveband with analysis and observation of optical bistability.
Photon. Res. 10, A35 (2022).
23. Wang, X. et al. Efﬁcient and high-speed coupling modulation of
silicon racetrack ring resonators at 2 µm waveband. Opt. Lett. 49,
2157–2160 (2024).
24. Pan, B. et al. Demonstration of high-speed thin-ﬁlm lithium-niobate-
on-insulator optical modulators at the 2-μm wavelength. Opt.
Express 29, 17710–17717 (2021).
25. Wang, C. et al. Integrated lithium niobate electro-optic modulators
operating at CMOS-compatible voltages. Nature 562, 101 (2018).
26. Feng, H. et al. Integrated lithium niobate microwave photonic
processing engine. Nature 627, 80–87 (2024).
27.
He, M. et al. High-performance hybrid silicon and lithium niobate
Mach-Zehnder modulators for 100 Gbit s−1 and beyond. Nat. Pho-
tonics 13, 359–364 (2019).
28. Zhang, Y. et al. Systematic investigation of millimeter-wave optic
modulation performance in thin-ﬁlm lithium niobate. Photon. Res.
10, 2380–2387 (2022).
29. Chen, G., Gao, Y., Lin, H.-L. & Danner, A. J. Compact and efﬁcient
thin-ﬁlm lithium niobate modulators. Adv. Photonics Res. 4,
2300229 (2023).
30. Xu, M. et al. Dual-polarization thin-ﬁlm lithium niobate in-phase
quadrature modulators for terabit-per-second transmission. Optica
9, 61 (2022).
31.
Makino, S. et al., In Optical Fiber Communication Conference
(OFC), paper M1D.2 (IEEE, 2022).
32. Renaud, D. et al. Sub-1 Volt and high-bandwidth visible to near-
infrared electro-optic modulators. Nat. Commun. 14, 1496 (2023).
33. Xue, S. et al. Full-spectrum visible electro-optic modulator. Optica
10, 125 (2023).
34. Valdez, F., Mere, V., Wang, X. & Mookherjea, S. Integrated O- and
C-band silicon-lithium niobate Mach-Zehnder modulators with 100
GHz bandwidth, low voltage, and low loss. Opt. Express 31,
5273–5289 (2023).
35. Fang, X., Yang, F., Chen, X., Li, Y. & Zhang, F. Ultrahigh-speed
optical interconnects with thin ﬁlm lithium niobate modulator. J.
Lightwave Technol. 41, 1207–1215 (2023).
36. Yi, Q., Pan, A., Xia, J., Zeng, C. & Shen, L. Ultra-broadband 1 × 2 3 dB
power splitter using a thin-ﬁlm lithium niobate from 1.2 to 2 µm
wave band. Opt. Lett. 48, 5375–5378 (2023).
37. Liu, X. et al. Capacitively-loaded thin-ﬁlm lithium niobate modulator
with ultra-ﬂat frequency response. IEEE Photonics Technol. Lett. 34,
854–857 (2022).
38. Krikorian, S. E. & Mahpour, M. The identiﬁcation and origin of N-H
overtone and combination bands in the near-infrared spectra of
simple primary and secondary amides. Spectrochim. Acta A 29,
1233–1246 (1973).
39. Brown, L. R. & Margolis, J. S. Empirical line parameters of NH3 from
4791 to 5294 cm−1. J. Quant. Spectrosc. Radiat. Transfer 56,
283–294 (1996).
Article
https://doi.org/10.1038/s41467-025-67902-2
Nature Communications|  (2026) 17:1138 
8

<!-- page 9 -->
40. Beale, C. A., Wong, A. & Bernath, P. Infrared transmission spectra of
hot ammonia in the 4800–9000 cm−1 region. J. Quant. Spectrosc.
Radiat. Transfer 246, 106911 (2020).
41.
R. Dadadzhanov, D., A. Vartanyan, T. & Karabchevsky, A. Lattice
Rayleigh anomaly associated enhancement of NH and Ch stretch-
ing modes on gold metasurfaces for overtone detection. Nanoma-
terials 10, 1265 (2020).
42. Hussein, M. G., Wörhoff, K., Sengo, G. & Driessen, A. Optimization of
plasma-enhanced chemical vapor deposition silicon oxynitride
layers for integrated optics applications. Thin Solid Films 515,
7–8 (2007).
43. Alam, M. S. amiul et al. Net 220 Gbps/λ IM/DD transmssion in
O-band and C-band with silicon photonic traveling-wave MZM. J.
Lightwave Technol. 39, 4270 (2021).
44. Han, C. et al. Slow-light silicon modulator with 110-GHz bandwidth.
Sci. Adv. 9, eadi5339 (2023).
45. Shen, J. et al. Highly efﬁcient slow-light Mach-Zehnder modulator
achieving 0.21 V·cm efﬁciency with bandwidth surpassing 110 GHz.
Laser Photonics Rev 18, 2200192 (2024).
46. Li, M., Wang, L., Li, X., Xiao, X. & Yu, S. Silicon intensity Mach-
Zehnder modulator for single lane 100 Gb/s applications. Photon.
Res. 6, 109 (2018).
47. Kharel, P., Reimer, C., Luke, K., He, L. & Zhang, M. Breaking voltage-
bandwidth limits in integrated lithium niobate modulators using
micro-structured electrodes. Optica 8, 357 (2021).
Acknowledgements
This work was supported by the National Major Research and Develop-
ment Program (grant No.2022YFB2802600 to M.Z., 2022YFB2803600
to L.S.), National Natural Science Foundation of China (grant
No.62175080 to L.S., 62435006 to M.Z., 62175079 to J.X., 62205119 to
A.P., 62235005 to J. Zhang, 62090054 to J. Zheng), the open research
fund of Songshan Lake Materials Laboratory (grant No. 2023SLABFK11 to
L.S.), the Open Project Program of Hubei Optical Fundamental Research
Center (grant No. HBO2026C016 to L.S.), Guangdong Provincial Cor-
nerstone Program (grant No. 2025B0303000008 to J. Zhang) and
Natural Science Foundation of Shanghai (grant No. 21ZR1408700 to J.
Zhang). The authors thank the Center of Optoelectronic Micro and Nano
Fabrication and Characterizing Facility, Wuhan National Laboratory for
Optoelectronics of Huazhong University of Science and Technology for
the support in device fabrication.
Author contributions
Q.L., Q.Y., A.P., and L.S. conceived the project. Q.L., Q.Y., and A.P. car-
ried out simulations and designed the TFLN modulator. A.P. and C.S.
fabricated the modulator. Q.L. and A.S. conducted the measurements.
Y.D., S.X., C.S., and S.Z. assisted in the measurements across the O-U
bands. J.C., Y.Z., J.L., and J. Zheng provided the 2-μm photodetector and
assisted in the measurements across 2-μm band. Q.Y., A.S., S.Z., and L.S.
analyzed and discussed the experimental results. Q.L. and L.S. wrote the
manuscript with contributions from all authors. The project was carried
out under the supervision of J. Zheng, J. Zhang, N.C., C.Z., J.X., L.S.,
and M.Z.
Competing interests
The authors declare no competing interests.
Additional information
Supplementary information The online version contains
supplementary material available at
https://doi.org/10.1038/s41467-025-67902-2.
Correspondence and requests for materials should be addressed to
Jun Zheng, Junwen Zhang, Cheng Zeng or Li Shen.
Peer review information Nature Communications thanks Rui Tang, who
co-reviewed with Nuo Chen, and the other, anonymous, reviewer(s) for
their contribution to the peer review of this work. A peer review ﬁle is
available.
Reprints and permissions information is available at
http://www.nature.com/reprints
Publisher’s note Springer Nature remains neutral with regard to
jurisdictional claims in published maps and institutional afﬁliations.
Open Access This article is licensed under a Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International License,
which permits any non-commercial use, sharing, distribution and
reproduction in any medium or format, as long as you give appropriate
credit to the original author(s) and the source, provide a link to the
Creative Commons licence, and indicate if you modiﬁed the licensed
material. You do not have permission under this licence toshare adapted
material derived from this article or parts of it. The images or other third
party material in this article are included in the article’s Creative
Commons licence, unless indicated otherwise in a credit line to the
material. If material is not included in the article’s Creative Commons
licence and your intended use is not permitted by statutory regulation or
exceeds the permitted use, you will need to obtain permission directly
from the copyright holder. To view a copy of this licence, visit http://
creativecommons.org/licenses/by-nc-nd/4.0/.
© The Author(s) 2026
Article
https://doi.org/10.1038/s41467-025-67902-2
Nature Communications|  (2026) 17:1138 
9

