---
paper_id: li2026ba
source_url: https://arxiv.org/abs/2604.14836
doi: 
license: CC-BY-4.0
sha256: 0d19e3fe597669e34b63881cf1caa433d7cfa655a59ace5a4ae30a7249770324
pages: 9
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Low voltage and high-bandwidth thin-film lithium tantalate modulator on a silicon dioxide
substrate
Zihan Li,1, 2, ∗Alexander Kotz,3, ∗Adrian Schwarzenberger,3 Christian Koos,3 and Tobias J. Kippenberg1, 2, †
1Institute of Physics, Swiss Federal Institute of Technology, Lausanne (EPFL), CH-1015 Lausanne, Switzerland
2Institute of Electrical and Micro Engineering (IEM), EPFL, CH-1015 Lausanne, Switzerland
3Institute of Photonics and Quantum Electronics (IPQ),
Karlsruhe Institute of Technology (KIT), 76131 Karlsruhe, Germany
Modern
communication
networks
demand
ever-
increasing transmission bandwidth, placing stringent re-
quirements on low-cost, high-performance electro-optic
modulators.
Substantial advances have been made in
integrated photonics employing lithium niobate on in-
sulator.
In contrast, photonic integrated circuits based
on lithium tantalate—a material already commercially
adopted for wireless filters—have been developed, offer-
ing reduced DC drift, higher optical power handling, and
lower birefringence. These advantages enable more com-
plex and dense photonic integrated circuits, and make
lithium tantalate a promising material platform for next-
generation integrated electro-optic modulators. However,
in contrast to the extensively studied thin-film lithium nio-
bate platform, thin-film lithium tantalate modulators have
only been explored on silicon substrates.
Here, we re-
port the first fabrication and characterization of thin-film
lithium tantalate electro-optic modulators manufactured
on a 4-inch (100 mm) fused-silica substrate for adapt-
ing a low-loss slow-wave microwave electrode to improve
the electro-optic bandwidth. By employing a slow-wave
electrode design to achieve velocity matching between mi-
crowave and optical signals, the demonstrated modulator
achieves a 3-dB electro-optic bandwidth of 64 GHz with
a low half-wave voltage of 1.53 V, with potential to op-
erate at the measured 100 GHz electrical bandwidth, if
the employed spectral biasing is removed. The modulator
moreover exhibits low bias drift, with a constant switch-
ing voltage down to 10 mHz. This performance enables
high-speed data transmission comparable to state-of-the-
art lithium niobate modulators fabricated on quartz sub-
strates. Using the fabricated devices, a net single lane data
rate of 440.6 Gbit s−1 is achieved using PAM8 signaling.
These results establish thin-film lithium tantalate as a vi-
able and scalable alternative to lithium niobate for high-
performance electro-optic links in next-generation com-
munication systems.
INTRODUCTION
The rapid advancement and deployment of artificial intel-
ligence have driven the demand for increasingly large-scale
computing clusters. However, progress in interconnect tech-
nologies has lagged behind the growth of computational ca-
pacity, leading to a widening gap between processing through-
put and network bandwidth [1]. Conventional electrical inter-
connects are fundamentally limited in both transmission dis-
tance and bandwidth, making optical links indispensable for
next-generation data-center and high-performance computing
systems. Traditional optical modules, which rely on directly
modulated semiconductor laser diodes to convert electrical
signals into optical signals, are increasingly unable to meet
the rising requirements for data rate and energy efficiency.
As data centers scale up, power consumption associated with
these modules has become a critical bottleneck.
To address these challenges, broadband external opti-
cal modulators, such as silicon microring modulators and
thin-film lithium niobate (TFLN) Mach–Zehnder modulators
(MZMs), have been extensively investigated. By integrating
with laser sources and digital signal processors, these modu-
lators enable compact and high-speed pluggable optical mod-
ules for dense optical interconnects. Among various material
platforms, lithium niobate (LiNbO3, LN), a well-established
electro-optic crystal, is particularly attractive due to its broad-
band Pockels effect and low operating voltage, offering clear
advantages over silicon photonic integrated circuits (PICs).
Since the first demonstration of TFLN modulators [2], they
have achieved electro-optic (EO) 3 dB bandwidths exceeding
110 GHz with CMOS-compatible drive voltages[2, 3]. Nev-
ertheless, their widespread adoption remains constrained by
high fabrication costs and several practical challenges, includ-
ing DC bias instability [4] and photorefractive effects under
high optical power [5].
Recently, lithium tantalate (LiTaO3, LT) photonic inte-
grated circuits have been developed [6] and high-speed mod-
ulators have been reported [7], establishing LT as a promising
alternative platform for electro-optical photonic integrated cir-
cuits. Already used in high volumes as RF filters, LT ben-
efits from mature crystal growth techniques and an estab-
lished foundry infrastructure, which are expected to support
wafer-scale manufacturing at costs and volumes comparable
to silicon-on-insulator platforms [8].
Compared with LN-
based devices, LT-based electro-optic modulators exhibit im-
proved DC stability and enhanced optical power handling, of-
fering further advantages for long-haul and high-power trans-
mission applications. In addition, LT exhibits a lower bire-
fringence (δn = |no −ne| = 0.004) than LN (δn = 0.074 for
LN at 1550 nm, which simplifies the design of complex PICs,
and enables arrayed waveguide gratings (AWG) [9], which
are challenging to implement in LN. Moreover, the low bire-
arXiv:2604.14836v1  [physics.optics]  16 Apr 2026

<!-- page 2 -->
2
(a)
(b)
 
(c)
(d)
(g)
(h)
100 μm
3 μm
(e)
5 μm
50 μm
0.4
0.2
0
Chip width (mm)
Chip height (mm)
-5
0
5
Facet topography (μm)
0
0.2
0.4
0.6
200 μm
(f)
(i)
18 mm
1. Wafer bonding
2. Carrier removal
3. Optical layer lithography
4. SiO2 cladding
5. Electrode lithography
7. fs laser cutting
6. Metal lift-off and capping
 
LT 
SiO2
Si
FS 
 DLC 
PR
Au
8. Wafer expansion
Activated surfaces
FIG. 1: Low voltage and high bandwidth thin film lithium tantalate modulators on a fused silica substrate. (a) Microscope picture of
a chip comprising three modulators from a 4-inch wafer (b). (c) Fabrication process flow for the fused-silica-based lithium tantalate modu-
lator, including the substrate preparation, device fabrication, and chip singulation via a femtosecond laser. (d)-(e) Colored scanning electron
microscopy (SEM) image of a segmented coplanar waveguide and its cross section. (g)-(h) SEM of the chip facet after laser singulation. The
lithium tantalate rib waveguide is colored blue, gold electrodes are colored yellow, and the silicon dioxide is colored purple. (f) and (i) show
the topography of the whole facet after the laser cutting by SEM and optical profilometer, respectively.
fringence of LT enables wideband operation, as demonstrated
for electro-optical frequency combs [10].
On the lithium
tantalate-on-insulator (LTOI) platform, 3 dB EO modulation
bandwidths exceeding 110 GHz have been demonstrated on
silicon substrates [7].
However, the high dielectric con-
stant of silicon (11.7) fundamentally constrains the design of
traveling-wave coplanar waveguide (CPW) electrodes, partic-
ularly with respect to the capacitance for impedance and ve-
locity matching. Therefore, in our previous work, on-chip mi-
crowave losses still limit the interaction length of LT-based

<!-- page 3 -->
3
modulators to approximately 6 mm, resulting in a relatively
high half-wave voltage (Vπ) of around 4.8 V [7]. To maintain
efficient modulation, the electrode-to-waveguide gap must be
kept within a few micrometers. Meanwhile, the impedance-
matching condition determines the width of the CPW sig-
nal conductor. These competing requirements result in in-
creased conductor loss for the microwave signal, ultimately
limiting the EO bandwidth. To overcome substrate-induced
bandwidth limitations, Kharel et al. and Chen et al. proposed
the use of low-permittivity substrates, such as quartz [11]
or released silicon [12], combined with capacitively loaded
slow-wave CPW electrodes. The introduction of periodic T-
shaped segments effectively separates the current paths in the
CPW and allows for a wider center conductor while preserv-
ing a 50 Ωimpedance match without violating the velocity-
matching condition. These design modifications significantly
reduce microwave loss and increase the EO modulation band-
width. Supplementary Note 2 explains how the microwave
loss benefits from the wider signal conductor and the wider
effective current separation.
Here, we further improve LT-based electro-optic modula-
tors by replacing the silicon substrate with fused silica and
implementing an optimized segmented electrode design. This
approach enables a favorable balance between velocity match-
ing, impedance matching, and microwave loss reduction. As
a result, we achieve a 64 GHz 3 dB EO modulation bandwidth
while maintaining a low Vπ of 1.53 V in the C-band. The
bandwidth is mainly limited by an inadvertent optical path
imbalance at the modulator’s output, which is used for spec-
tral biasing. Optimized design projects a bandwidth expan-
sion to 100 GHz while preserving the same half-wave volt-
age.
To demonstrate the practical application of the pro-
posed devices, we perform high-speed intensity-modulation
and direct-detection (IMDD) signaling experiments, achiev-
ing a net data rate of 440.6 Gbit s−1, which is comparable to
values reported for state-of-the-art lithium niobate modulators
[13].
RESULTS
Design and fabrication
The LT on fused silica (LT-on-FS) electro-optic modula-
tors are fabricated from a commercial X-cut thin-film lithium
tantalate-on-insulator (LTOI) wafer (NANOLN), consisting of
a 600 nm-thick LiTaO3 single crystal thin-film, a 2 µm-thick
silicon dioxide buffer layer, and a 525 µm-thick silicon han-
dle substrate. To enable low-loss microwave propagation, the
LiTaO3 thin-film is transferred to a 4-inch, 500 µm-thick fused
silica wafer via direct wafer bonding, with oxygen plasma
activation employed as the surface pretreatment (EVG 810,
EVG 501, and EVG 301). The original silicon carrier and the
silicon dioxide buffer layer are subsequently removed through
a combination of backside grinding, fluorine-based plasma
dry etching, and buffered hydrofluoric acid wet etching.
With an optimized bonding recipe and carrier removal pro-
cess, the fabricated LT-on-FS wafer achieves a film transfer
yield exceeding ∼95%. In addition, the relatively small ther-
mal expansion coefficient mismatch between silicon (2.6 ×
10−6 K−1) and fused silica (0.5 × 10−6 K−1) allows higher
annealing temperatures during the bonding process com-
pared with direct thin-film transfer from ion-implanted bulk
LT wafers, which exhibit pronounced large and anisotropic
thermal expansion (αz = 2.0 × 10−6 K−1, αx,y = 15.0 ×
10−6 K−1[14]).
Following the wafer preparation, the LiTaO3 photonic inte-
grated circuits (PICs) are fabricated using a direct etching ap-
proach previously reported in [6, 15]. Deep-ultraviolet (DUV)
stepper lithography (ASML PAS 5500/350C) is employed
to define the photonic patterns, which are transferred into a
diamond-like carbon (DLC) hard mask via oxygen plasma
dry etching (SPTS APS). Then, the patterns are etched into
the LiTaO3 layer using argon ion beam etching (Veeco Nexus
IBE350). To remove redeposited LiTaO3 residues resulting
from ion milling, a subsequent wet etching using an aque-
ous solution of hydrogen peroxide and potassium hydroxide is
performed. The final etch depth is 440 nm, leaving a 160 nm-
thick slab to ensure efficient electro-optic modulation. A sec-
ond DUV lithography and ion beam etching step is used to
define the double-layer taper to reduce the edge coupling loss.
The PICs are clad with a 1.5 µm-thick SiO2 layer deposited
by hydrogen-free high-density plasma-enhanced chemical va-
por deposition (HD-PECVD)[16], which also serves as a sac-
rificial layer for metalization process. The microwave elec-
trodes are patterned by DUV stepper lithography and fabri-
cated through a sequence of dielectric etching, metal evap-
oration (15 nm Ti / 800 nm Au), and lift-off [6]. To protect
the soft gold electrodes, a 200 nm-thick SiO2 cap layer is de-
posited by PECVD, and then the pad area is opened by wet
etching. Finally, the wafer is singularized using femtosecond-
laser stealth dicing (General Intelligent Equipment Co., Ltd.).
The key fabrication steps are summarized in Fig. 1 (c).
Figure 1 (a) shows a fabricated 3 mm×20 mm chip con-
taining three unbalanced Mach-Zehnder modulator (MZM),
each composed of a pair of 18 mm-long modulation arms
and two 50:50 multimode interference (MMI) couplers. The
chip is taken from a 4-inch wafer, as shown in Fig. 1 (b).
Colored scanning electron microscopy (SEM) images reveal
well-defined segmented coplanar waveguide (CPW) and LT
waveguides in both top-view (Fig. 1 (d)) and cross-sectional
(Fig. 1 (e)) perspectives.
Due to the amorphous structure of fused silica and the lack
of an efficient deep reactive ion etching process, die singula-
tion is performed using stealth dicing, in which arrays of mod-
ified points are generated inside the fused silica substrate by a
femtosecond laser. These modified regions allow controlled
chip separation during wafer expansion.
Figures 1(f)–(h)
show the chip facets, which are clean and intact. The smooth
region near the top functional layer is suitable for direct fiber
packaging without additional polishing, resulting in a fiber-
to-fiber coupling loss of approximately 12 dB. The facet to-

<!-- page 4 -->
4
pography is quantitatively characterized using an optical pro-
filometer (Sensofar S-Neox), presenting a surface roughness
of ±5 µm induced by laser modification, which indicates the
need for facet polishing before fiber arrays or laser diode cou-
pling.
To achieve broadband electro-optic modulation, segmented
T-shaped CPW are designed.
The segment length LS
(Fig. 3 (a)) is varied to tune the microwave phase velocity,
where increasing the segment length effectively reduces the
microwave phase velocity. The bandwidth is maximized when
the microwave phase velocity matches the optical group ve-
locity, while simultaneously the characteristic impedance is
matched to external circuitry [11]. The electrode parameters,
as defined in Fig. 3 (a), selected for the presented device are
(WT, LT, WS, LS, G, Wsig, P) = (1.5, 32, 2, 5, 5, 100, 35) µm.
Electro-optic Modulation
To evaluate the wafer-scale modulation performance of the
fabricated devices, we select modulators with identical de-
signs from each exposure field and characterize their modula-
tion efficiency, as the half-wave voltage–length product (VπL).
For low-frequency characterization, a sawtooth voltage wave-
form with a peak-to-peak amplitude (Vpp) of 10 V is applied
to the modulator electrodes, while the corresponding optical
output is simultaneously recorded using a photodetector and
an oscilloscope. The half-wave voltage is calculated by fitting
the EO response curve.
Owing to the small difference in the optical group in-
dex between the O-band (e.g.
1300 nm) and the C-band
(e.g. 1550 nm), the modulators are inherently compatible with
both wavelength ranges, with minor sacrifice in insertion loss
and extinction ratio (ER). To quantify the dual-band modula-
tion capability, we measure the modulation efficiency using
laser sources at 1300 nm and 1550 nm separately. Figure 2
summarizes the dual-wavelength measurement results, with
data acquired at 1300 nm shown in orange and at 1550 nm
shown in blue. Figure 2 (a) presents the normalized transmis-
sion over one modulation period for a representative device
(C1 F11 1.02) on a logarithmic scale.
For an 18 mm-long
modulator, half-wave voltages of 1.53 V and 1.21 V are ex-
tracted at wavelengths of 1550 nm and 1300 nm, respectively.
The corresponding extinction ratios are approximately 12 dB
in the C-band and 10 dB in the O-band.
We further analyze the dynamic behavior of our lithium-
tantalate-on-fused-silica (LT-on-FS) MZM. The device is de-
signed as a traveling-wave electro-optic modulator such that
the frequency response is primarily dominated by the RF loss
of the transmission line, the matching between the group ve-
locity of the optical wave and the phase velocity of the RF
wave, as well as the impedance matching of the transmission
line to the source and the termination [18]. The electrical
characterization of the LT-on-FS MZM is performed using a
high-speed vector network analyzer (VNA; ME7838AX, An-
ritsu Corporation) covering a frequency range from 70 kHz to
125 GHz. The measurements are carried out using a pair of
110 GHz ground–signal–ground (GSG) RF probes (T110A-
GSG0100, MPI Corporation). Prior to the measurement, a
two-port short–open–load–through (SOLT) calibration is con-
ducted on a commercial calibration substrate (AC2-2, MPI
Corporation), setting the reference planes at the probe tips
with a characteristic impedance of 50 Ω. The electrical scat-
tering parameters of the 18 mm-long MZM are then obtained
by probing the input and output contact pads.
As shown in Fig. 3 (b), the segmented CPW exhibits a
reflection coefficient (S 11) below −18 dB, and the electrical
transmission coefficient (S 21) drops to −8.2 dB at 120 GHz.
This corresponds to a microwave loss of approximately
4.6 dB cm−1 at 120 GHz, highlighting the low RF attenua-
tion of the T-shape segmented electrodes, see blue curve in
Fig. 3 (c). We also extract the frequency-dependent effective
phase refractive index of the RF signal from the S-parameters
of the CPW, see red curve in Fig. 3 (c). The strong increase
at low frequencies is attributed to the finite conductivity of
the electrodes, which allows magnetic fields to penetrate the
conductors and thereby increases the internal inductance [19].
For comparison, the effective group refractive indices for the
optical wave, obtained from numerical simulations, are given
at a wavelength of 1550 nm (C-band, green) and 1310 nm (O-
band, yellow). The results indicate good velocity matching in
both bands.
To measure the electro-optic frequency response of the LT-
on-FS MZM, an optical carrier is coupled to the modulator,
and the device is driven by the first VNA port, while the sec-
ond port is connected to a photodiode that detects the modu-
lated optical signal. To avoid reflections of the RF wave, the
MZM is terminated by a second probe and a 50 Ωresistor. To
isolate the electro-optic frequency response of the LT-on-FS
MZM, we use appropriate de-embedding techniques that shift
the reference planes of the VNA measurement to the tips of
the input probe and the optical output interface of the mod-
ulator. These de-embedding steps rely on the frequency re-
sponses of the second RF probe and the photodiode as pro-
vided by the manufacturers.
Fig. 3 (d) shows the electro-optic frequency response of
the 18 mm-long LT-on-FS MZM normalized to its value at
1 GHz for optical carrier wavelengths of 1550 nm (C-band,
blue) and 1310 nm (O-band, red). The light-colored dots rep-
resent the raw measurement data, while the solid-colored lines
correspond to the same data after applying a centered 5 GHz-
wide moving-average filter. The smoothed device character-
istics in the C-band indicate a 3 dB bandwidth of 64 GHz and
a 6 dB bandwidth that exceeds 100 GHz. The measurements
in the O-band show similar characteristics, with some uncer-
tainties originating from the O-band calibration of the pho-
todiode. The increased roll-off of the electro-optic response
above 60 GHz is attributed to an inadvertent imbalance in the
optical paths after the modulation and before the combiner,
which introduces an additional optical filtering effect. This
can be easily avoided by an adapted design.
The solid green curve in Fig. 3 (d) shows the electro-

<!-- page 5 -->
5
Vπ = 1.53 V
10-2
10⁰
10²
10⁴
Frequency (Hz)
0
1
2
3
VπL (V∙cm)
(a)
(b)
2.88
3.20
2.90
2.78
2.84
2.83
2.90
2.78
2.82
2.76
2.77
2.29
2.24
2.20
2.34
2.25
3.65
2.59
2.24
2.19
2.36
2.22
Vπ = 1.21 V
(c)
-2
-1
0
1
2
Drive Voltage (V)
-10
-5
0
EO response (dB)
1550 nm
1300 nm
F1
F3
F5
F2
F7
F4
F8
F6
F9
F10
F11
FIG. 2: Modulation efficiency characterization of LT-on-FS Mach-Zehnder modulators in the C- and O-band. Measurements at wave-
length of 1550 nm are colored blue, and those at 1300 nm are shown in orange. (a) Normalized optical transmission in logarithmic scale of an
18 mm-long modulator as a function of the drive voltage at 100 Hz. The half-wave voltages Vπ are 1.53 V and 1.21 V at carrier wavelengths
of 1550 nm and 1300 nm, respectively. (b) Half-wave voltage-length VπL products as a function of drive voltage frequency. The electro-optic
response is stable at both carrier wavelengths 1550 nm and 1300 nm for modulation frequencies from 1 Hz to 10 kHz. (c) Distribution of the
measured VπL for the same device design across a 4-inch wafer.
optic response of the LT-on-FS MZM calculated from the ex-
tracted transmission-line parameters and the simulated effec-
tive group refractive index of the optical wave in the C-band
(see Fig. 3 (c)) according to [18]. The prediction agrees with
the measured electro-optic response of our MZM at modula-
tion frequencies below 60 GHz and provides a reference for
the response for a modulator without the imbalance.
Fur-
ther analysis of the initial drop of the electro-optic response
at low frequencies reveals a mismatch of the characteristic
impedance of the modulator, which amounts to approximately
42 Ωaccording to the electric S-parameters, to the 50 Ωsource
and termination impedance. An 18 mm-long MZM with opti-
mized design, for which the electro-optic response is limited
only by the microwave losses of the electrodes, could achieve
a 3 dB bandwidth of 100 GHz on our LT-on-FS platform, as
indicated by the dashed green curve in Fig. 3 (d)).
High-Speed IMDD Signaling with LT-on-FS MZM
To demonstrate the transmission performance of our LT-
on-FS MZM, we perform a high-speed intensity-modulation
and direct-detection (IMDD) signaling experiment with the
setup illustrated in Fig. 4 (a).
An external-cavity diode
laser (ECDL) provides the optical carrier at a wavelength of
1550 nm with an output power of 17.8 dBm. The polarization
of the light is adjusted via a fiber-based polarization controller
(FPC) to excite the quasi-transverse electric mode of the LT
waveguide, having the dominant component of the electric
field parallel to the substrate plane. Lensed fibers are used for
optical coupling to and from the chip. The electrical drive sig-
nal is generated using a high-speed arbitrary waveform gen-
erator (AWG; M8199B, Keysight Technologies Inc.) with a
sampling rate of 256 GSa/s and transmitted via a 20 cm-long
RF cable, a broadband RF amplifier (Amp.; AH15199B, An-
ritsu Corporation) and a 110 GHz RF probe to the CPW of the
MZM. Pulse-amplitude modulated (PAM) signals with vari-
ous levels are generated from pseudo-random bit sequences
(PRBS) with a pattern length longer than 217 bits using of-
fline digital signal processing (Tx-DSP). The Tx-DSP chain
furthermore consists of a root-raised cosine pulse shaping fil-
ter with a roll-off factor of β = 0.05, predistortion to account
for the frequency response of the transmitter electronics ex-
cluding the modulator, and clipping of the signal to reduce the
peak-to-average power ratio to a value of 9 dB. The predis-
tortion is based on linear minimum-mean-square-error equal-
ization. At the output, the CPW is terminated with a 50 Ωre-
sistor via a second 110 GHz RF probe and a broadband bias-
tee (not shown) to set the quadrature operation point of the
MZM for intensity modulation. The modulated optical signal
has a power of 3.7 dBm in the fiber right after the MZM, ex-
ceeding typical specifications for high-speed optical Ethernet
transceivers [20]. Still, we had to use an erbium-doped fiber
amplifier (EDFA) to slightly boost the optical output power
to a level of approximately 7.7 dBm at the receiver to allow
for detection with a high-speed photodiode (PD; Fraunhofer
HHI) and a directly attached real-time oscilloscope (RTO;
UXR 1004A, Keysight Technologies Inc.).
The latter has
a sampling rate of 256 GSa/s and a bandwidth of 104 GHz.
Note that, in practical systems, a broadband RF amplifier af-
ter the photodiode could be used, thus eliminating the need for
the EDFA. The out-of-band amplified spontaneous-emission
(ASE) noise of the EDFA is suppressed by a tunable bandpass
filter (BPF). After digitization by the RTO, the data is finally
extracted using offline DSP (Rx-DSP), which contains resam-
pling to two samples per symbol, timing recovery, linear Sato
equalization, and additional least-mean-square (LMS) equal-
ization.
In our signaling experiments, we generate and receive PAM
signals with four (PAM4), six (PAM6) and eight (PAM8)

<!-- page 6 -->
6
Wsig
WT
WS
P
G
LS
LT
-9
-6
-3
0
2.2
2.1
Frequency (GHz1/2)
RF attenuation (dB/cm)
Effective index
EO response (dB)
Electrical S-parameters (dB)
2.3
2.4
S11
S21
RF phase index
Optical group index
at 1550 nm
at 1310 nm
C-band (calculated)
C-band (measured)
C-band w/ imp. match. (calculated)
O-band (measured)
FIG. 3: Characterization of the modulator frequency response. (a) Optical micrograph of the slow-wave electrodes of the Mach-Zehnder
modulator (MZM) with an inset showing the T-shaped capacitive loading elements and with a schematic illustrating the design parameters.
(b) Measured electrical S-parameters of the 18 mm-long capacitively loaded coplanar waveguide (CPW) electrodes of the MZM. (c) Extracted
microwave attenuation of the CPW (blue) and effective phase index (red) of the RF signal plotted as a function of the square root of the
frequency. As a reference, the effective group refractive index of the optical signal at a wavelength of 1550 nm (green) and 1310 nm (yellow)
are shown. (d) Frequency-dependent electro-optic (EO) response of the 18 mm-long MZM for carrier wavelengths of 1550 nm (C-band, blue)
and 1310 nm (O-band, red). The light-colored dots correspond to raw data, while the solid-colored lines indicate smoothed data obtained
by applying a centered 5 GHz-wide moving-average filter. The data are obtained from vector network analyzer (VNA) measurements with
a high-speed photodiode, de-embedding, and normalization to the response at 1 GHz. For the O-band measurement, the data points below
10 GHz were obtained with a separate low-speed photodiode using the method described in [17]. We used this approach to avoid uncertainties
of the transfer characteristics of the high-speed photodiode when operated at low modulation frequencies in the O-band. The solid green curve
indicates the EO response of our MZM calculated from the extracted transmission line parameters and the simulated effective group refractive
index of the optical wave in the C-band, see Subfigure (c) according to [18]. The deviation from the measurement above 60 GHz originates
from an inadvertent imbalance in the optical paths before the combiner, which acts as an optical filter reducing the EO response. The dashed
green curve shows the predicted performance of our MZM with an optimized design, where the line impedance is perfectly matched to the
source and the termination, and where the EO response is limited only by the microwave losses of the electrodes.
power levels at symbol rates between 144 GBd and 208 GBd.
Figure 4 (b) shows the bit error ratio (BER) as a function of the
symbol rate, with dashed lines indicating thresholds for soft-
decision forward-error correction (SD-FEC) with 25 % and
15 % overhead [21], as well as for hard-decision forward-error
correction (HD-FEC) with 7 % [22] and 5.8 % (KP4) [20]
overhead. The measured BER remain below the 25 % SD-
FEC limit for PAM8 signals at 184 GBd, PAM6 signals at
192 GBd, and PAM4 signals at 208 GBd. Up to a symbol rate
of 176 GBd, the BER for PAM4 signals stays below the KP4
limit.
To quantify the achievable information rate (AIR), the gen-
eralized mutual information (GMI) of our measurements is
calculated using log-likelihood ratios between the transmit-
ted and received signals while assuming a linear channel
with additive white Gaussian noise (AWGN) as the only im-
pairment [23]. The results are indicated as dashed lines in
Fig. 4 (c), where the highest AIR of 468.1 Gbit/s is achieved
for PAM8 signals at a symbol rate of 180 GBd. To estimate
practically achievable net data rates (NDRs), penalties intro-
duced by typical FEC codes must also be considered. To this
end, the GMI is normalized by the number of bits that can
be encoded into a single symbol and suitable FEC codes with
the nearest lower normalized GMI threshold as given in [24,
Table 2] are selected. The NDR is computed as the product
of line rate and FEC code rate and is indicated by circles and
solid lines in Fig. 4 (c). For each modulation format, we iden-
tify the symbol rates at which the highest NDR are achieved
and extract the corresponding eye diagrams as obtained after
RX-DSP, see Fig. 4 (d-f). The plots also show the correspond-

<!-- page 7 -->
7
176 GBd PAM8 (NDR = 441 Gbit/s)
Time (2 ps/div)
Amplitude (a.u.)
FPC
Amp.
PD
Tx-DSP
AWG
Rx-DSP
LT-on-FS MZM
EDFA
RTO
BPF
Symbol Rate (GBd)
140
160
180
200
PAM4
PAM6
PAM8
BER
10-5
10-4
10-3
10-2
KP4
7% HD
15% SD
25% SD
AIR / NDR (Gbit/s)
Symbol Rate (GBd)
140
300
450
400
350
PAM4
PAM8
PAM6
160
180
200
ECDL
Counts (a.u.)
Counts (a.u.)
Time (2 ps/div)
184 GBd PAM6 (NDR = 409 Gbit/s)
Amplitude (a.u.)
Counts (a.u.)
Amplitude (a.u.)
Time (2 ps/div)
200 GBd PAM4 (NDR = 361 Gbit/s)
A
A
A
C
C
C
B
B
B
(e)
(f)
FIG. 4: Intensity-modulation and direct-detection (IMDD) signaling experiment using our lithium-tantalate-on-fused-silica (LT-on-FS)
Mach-Zehnder modulator (MZM). (a) Experimental setup: An external cavity diode laser (ECDL) provides the optical carrier, and a fiber
polarization controller (FPC) is used to adjust the polarization. Optical coupling at the input and output of the LT-on-FS chip is achieved via a
pair of lensed fibers. The electrical drive signals are synthesized by transmitter digital signal processing (Tx-DSP), generated by an arbitrary
waveform generator (AWG), and amplified by a broadband RF amplifier (Amp.). To enable reception by a rather insensitive photodiode (PD)
without electrical amplifiers, the modulated optical signal is boosted by an erbium-doped fiber amplifier (EDFA), and out-of-band amplified
spontaneous emission (ASE) noise is suppressed by a tunable bandpass filter (BPF). The output signal of the PD is digitized by a high-speed
real-time oscilloscope (RTO) and processed offline using the receiver DSP (Rx-DSP). (b) Measured bit error ratios (BER) as a function of
the symbol rate for pulse-amplitude modulated signals with eight (PAM8, green), six (PAM6, orange), and four (PAM4, blue) levels. The
horizontal black dashed lines indicate the thresholds for 25 % and 15 % soft-decision forward error correction (SD-FEC), as well as 7 %
and 5.8 % (KP4) hard-decision forward error correction (HD-FEC). The labels A , B , and C refer to the corresponding eye diagrams in
Subfigures (d) - (f). (c) Achievable information rates (AIR, dashed lines) and corresponding net data rates (NDR, solid lines) vs. symbol
rate. A maximum NDR of 441 Gbit s−1 is achieved for PAM8 signals at a symbol rate of 176 GBd. (d) - (f) Eye diagrams after Rx-DSP and
corresponding histograms, taken at the center of the respective symbol slot (indicated by the vertical dashed line), for symbol rates offering the
highest NDR for each modulation format. These data points are highlighted in Subfigures (b) and (c)).
ing histograms, evaluated at the center of the symbol period.
The corresponding data points are highlighted in Fig. 4 (b)
and Fig. 4 (c). Clear eye openings and distinct clustering at
the different signal levels are observed. The highest NDR of
440.6 Gbit/s is obtained for PAM8 signals at a symbol rate
of 176 GBd using SD-FEC with 18.7 % overhead. For PAM4
signals at a symbol rate of 176 GBd, the NDR is 332.6 Gbit/s
using KP4 FEC. These results are comparable to those re-
ported for high-bandwidth LNOI MZMs [13]. Note that, at
high symbol rates, the quality of the generated optical signals
is primarily limited by the electrical signal source and not so
much by the LT-on-FS, see Supplementary Note 1 for details.
CONCLUSION
Lithium niobate modulators have represented the state of
the art in terms of electro-optic bandwidth and modulation ef-
ficiency. Whether lithium tantalate, an emerging electro-optic
photonic integrated circuits platform, can serve as a potential
replacement for lithium niobate has remained an open ques-
tion due to concerns regarding microwave loss and its rela-
tively higher dielectric constant. In this work, we adopt a de-
sign similar to that of advanced lithium niobate modulators
and, for the first time, demonstrate thin-film lithium tanta-
late modulators fabricated on a fused silica substrate, which
achieve performance comparable to lithium niobate counter-
parts.
By transferring X-cut lithium tantalate (LT) thin films
onto a fused silica (FS) wafer and implementing opti-
mized segmented traveling-wave electrodes, we simultane-
ously achieve low microwave loss, precise velocity match-
ing, and impedance matching. The fabricated LT-on-FS mod-
ulators exhibit uniform performance across a 4-inch wafer,
with an average modulation efficiency of 2.86 (2.76, 3.20)
V cm in the C-band and 2.42 (2.19, 3.65) V cm in the O-
band. While the measured 3 dB electro-optic bandwidth of
an 18 mm-long Mach–Zehnder modulator is only 64 GHz, the
fundamental bandwidth limitation due to microwave losses of
the traveling-wave electrodes indicates an achievable band-

<!-- page 8 -->
8
width of 100 GHz with Mach-Zehnder modulator (MZM) of
the same length on the lithium-tantalate-on-fused-silica (LT-
on-FS) platform. The observed roll-off in the electro-optic
response is primarily due to an inadvertent imbalance in the
optical path at the MZM output. Beyond device-level char-
acterization, we validate the system-level application of the
LT-on-FS platform through high-speed intensity-modulation
and direct-detection (IMDD) experiments. Using PAM8 sig-
naling in the C-band, a net data rate of up to 440.6 Gbit s−1 is
achieved, comparable to the performance reported for state-
of-the-art thin-film lithium niobate modulators.
Our results demonstrate that LT-on-FS modulators consti-
tute a scalable, high-performance alternative to lithium nio-
bate for broadband and energy-efficient electro-optic links.
Our work provides a strong reference for the adoption of
lithium tantalate in next-generation high-speed optical inter-
connects and its potential to support cost-effective, wafer-
scale manufacturing for future communication systems.
Data and Code Availability Data and code used to produce
the plots within this paper will be available at Zenodo upon
publication of the manuscript.
Funding
This work was supported by the Horizon Europe EIC Tran-
sition programme under grant agreement No. grant agreement
101131069 (ELLIPTIC) and under grant No.
101113260
(HDLN), as well as funding from the Swiss State Secretariat
for Education, Research and Innovation (SERI). This work
has received funding from the European Research Council
(ERC) under the Horizon Europe research and innovation pro-
gramme, grant agreement No. 101167540 (ATHENS), as well
as from the German Research Foundation via the projects
PACE (No. 403188360) and GOSPEL (No. 403187440).
Acknowledgements We acknowledge the EPFL Center of
MicroNano Technology (CMi) and the Institute of Physics
(IPHYS) cleanroom for supporting on sample fabrication.
Competing Interests C.K. and T.J.K. are co-founders and
shareholders of Luxtelligence SA, St. Sulpice, Switzerland,
a company engaged in electro-optic modulators based on fer-
roelectric materials, such as lithium niobate and lithium tan-
talate. The other authors declare no competing interests.
∗Equal contribution for this work
† Electronic address: tobias.kippenberg@epfl.ch
[1] N. Harris,
Lightmatter interconnect launch event at ofc
2025,
URL
https://lightmatter.co/resource/
lightmatter-interconnect-launch-event-at-ofc-2025/
#.
[2] C. Wang, M. Zhang, X. Chen, M. Bertrand, A. Shams-Ansari,
S. Chandrasekhar, P. Winzer, and M. Lonˇcar, Nature 562, 101
(2018), ISSN 1476-4687, URL https://doi.org/10.1038/
s41586-018-0551-y.
[3] M. Xu, M. He, H. Zhang, J. Jian, Y. Pan, X. Liu, L. Chen,
X. Meng, H. Chen, Z. Li, et al., Nat. Commun. 11, 3911
(2020), ISSN 2041-1723, URL https://doi.org/10.1038/
s41467-020-17806-0.
[4] J. Holzgrafe, E. Puma, R. Cheng, H. Warner, A. Shams-Ansari,
R. Shankar, and M. Lonˇcar, Opt. Express 32, 3619 (2024),
URL https://opg.optica.org/oe/abstract.cfm?URI=
oe-32-3-3619.
[5] Y. Xu, M. Shen, J. Lu, J. B. Surya, A. A. Sayem, and H. X.
Tang, Optics Express 29, 5497 (2021).
[6] C. Wang, Z. Li, J. Riemensberger, G. Lihachev, M. Churaev,
W. Kao, X. Ji, J. Zhang, T. Blesin, A. Davydova, et al., Nature
629, 784 (2024), ISSN 1476-4687, URL https://doi.org/
10.1038/s41586-024-07369-1.
[7] C. Wang, D. Fang, J. Zhang, A. Kotz, G. Lihachev, M. Chu-
raev, Z. Li, A. Schwarzenberger, X. Ou, C. Koos, et al., Optica
11, 1614 (2024), URL https://opg.optica.org/optica/
abstract.cfm?URI=optica-11-12-1614.
[8] SOITEC, Capital markets day 2021 (2021), https://www.
soitec.com/en/capital-markets-day-2021.
[9] S. U. Hulyal, J. Hu, C. Wang, J. Cai, G. Lihachev, and T. J. Kip-
penberg, Optica 12, 978 (2025), URL https://opg.optica.
org/optica/abstract.cfm?URI=optica-12-7-978.
[10] J. Zhang, C. Wang, C. Denney, J. Riemensberger, G. Lihachev,
J. Hu, W. Kao, T. Bl´esin, N. Kuznetsov, Z. Li, et al., Nature
637, 1096 (2025).
[11] P. Kharel, C. Reimer, K. Luke, L. He, and M. Zhang, Optica
8, 357 (2021), URL https://opg.optica.org/optica/
abstract.cfm?URI=optica-8-3-357.
[12] G. Chen, K. Chen, R. Gan, Z. Ruan, Z. Wang, P. Huang, C. Lu,
A. P. T. Lau, D. Dai, C. Guo, et al., APL photonics 7 (2022).
[13] E. Berikaa, M. S. Alam, W. Li, S. Bernal, B. Krueger, F. Pittal`a,
and D. V. Plant, IEEE Photonics Technology Letters 35, 850
(2023).
[14] Y. Kim and R. Smith, Journal of Applied Physics 40, 4637
(1969).
[15] Z. Li, R. N. Wang, G. Lihachev, J. Zhang, Z. Tan, M. Churaev,
N. Kuznetsov, A. Siddharth, M. J. Bereyhi, J. Riemensberger,
et al., Nat. Commun. 14, 4856 (2023), ISSN 2041-1723, URL
https://doi.org/10.1038/s41467-023-40502-8.
[16] Z. Qiu, Z. Li, R. N. Wang, X. Ji, M. Divall, A. Siddharth, and
T. J. Kippenberg (2024), 2312.07203, URL https://arxiv.
org/abs/2312.07203.
[17] R. Nagarajan, Technique for measuring the Vpi-AC of a
Mach-Zehnder modulator (Mar. 20,
2001),
U.S. Patent
6,204,954
B1,
URL
https://patents.google.com/
patent/US6204954B1/.
[18] S. H. Lin and S. Y. Wang, 26, 1696 (1987), ISSN 1559-128X.
[19] J.-Y. Ke and C. H. Chen, IEEE Transactions on Microwave The-
ory and Techniques 43, 1128 (1995).
[20] IEEE Std 802.3-2022 (Revision of IEEE Std 802.3-2018) pp.
1–7025 (2022).
[21] A. Graell i Amat and L. Schmalen, Forward Error Correction
for Optical Transponders (Springer International Publishing,
Cham, 2020), pp. 177–257, ISBN 978-3-030-16250-4, URL
https://doi.org/10.1007/978-3-030-16250-4_7.
[22] Recommendation G.975.1, Telecommunication standardization
sector of International Telecommunication Union (2004), URL
https://www.itu.int/rec/T-REC-G.975.1-200402-I/
en.
[23] M. Ivanov, C. H¨ager, F. Br¨annstr¨om, A. Graell i Amat, A. Al-
varado, and E. Agrell, IEEE Trans. Inf. Theory 62, 3011 (2016).

<!-- page 9 -->
9
[24] Q. Hu, R. Borkowski, Y. Lefevre, J. Cho, F. Buchali, R. Bonk,
K. Schuh, E. De Leo, P. Habegger, M. Destraz, et al., J. Light.
Technol. 40, 3338 (2022).

