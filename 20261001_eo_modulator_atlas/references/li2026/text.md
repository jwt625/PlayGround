---
paper_id: li2026
source_url: https://arxiv.org/abs/2607.17436
doi: 
license: CC-BY-4.0
sha256: 87f0860c4af8fbb51420de364038024e8a71bf900e21227e8559bc0f292fc20a
pages: 6
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Broadband suspended lithium tantalate Mach–Zehnder modulator achieving a 460 Gbit/s net data
rate
Zihan Li,1, 2, ∗Alexander Kotz,3, ∗Adrian Schwarzenberger,3 Christian Koos,3 and Tobias J. Kippenberg1, 2, †
1Institute of Physics, Swiss Federal Institute of Technology, Lausanne (EPFL), CH-1015 Lausanne, Switzerland
2Institute of Electrical and Micro Engineering (IEM), EPFL, CH-1015 Lausanne, Switzerland
3Institute of Photonics and Quantum Electronics (IPQ),
Karlsruhe Institute of Technology (KIT), 76131 Karlsruhe, Germany
Thin-film lithium tantalate LiTaO3 photonic integrated circuits have recently been demonstrated as a promis-
ing next-generation electro-optic platform, offering favorable properties including reduced DC drift, higher opti-
cal power handling, and lower birefringence compared to lithium niobate. However, high-speed LiTaO3 modula-
tors reported to date have predominantly relied on silicon substrates, whose large dielectric constant (εr ≈11.7)
compromises microwave velocity matching and imposes RF conductor losses that limit the achievable electro-
optic bandwidth. Here, we implement a silicon substrate undercut technique to suspend the electrode region of
lithium-tantalate-on-insulator (LTOI) Mach-Zehnder modulators (MZMs), effectively decoupling the traveling-
wave electrodes from the high-permittivity silicon handle wafer, thereby reducing microwave losses. In addition,
the undercut removes any susceptibility to parasitic surface conductance (PSC) induced losses of the oxide-
silicon interface. The fabricated MZM achieves a 3 dB electro-optic bandwidth of 110 GHz, with a half-wave
voltage of 5.1 V (VπL = 4 V cm) for an 8 mm-long device. Exploiting the extended bandwidth, we demonstrate
a high single-lane intensity-modulation and direct-detection (IMDD) net data rate of 460 Gbit s−1 using PAM8
signaling. These results establish silicon substrate undercut as an effective and process-compatible pathway to
unlock the full electro-optic potential of lithium tantalate on its native silicon-based wafer platform.
INTRODUCTION
The explosive growth of artificial intelligence workloads
and hyper-scale data-center deployments is driving urgent
demand for optical interconnect at all scales, from short
range (within data centers), to long range (between the data
centers), with high data rates and lower energy consump-
tion [1]. Photonic integrated circuit-based electro-optic mod-
ulators based on thin-film ferroelectric materials, such as
lithium niobate (LiNbO3) and lithium tantalate (LiTaO3), have
attracted considerable attention as they simultaneously offer
large electro-optic coefficients, low drive voltages, and broad
modulation bandwidths well suited for single-lane rates be-
yond 400 Gbit s−1 [2, 3]. This makes this platform in par-
ticular promising for coherent communications, that utilize
complex modulation formats, and is used in long-haul optical
communications. Similarly, it has been considered recently
for linear drive pluggable optics (LPO) [4, 5].
Recently, low loss lithium tantalate photonic integrated cir-
cuits have been demonstrated as an alternative ferroelectric
material platform [6, 7]. Lithium tantalate on insulator (LTOI)
is already today commercially deployed in RF surface acous-
tic wave (SAW) filters for RF front ends and therefore bene-
fits from economies of scales i.e. low costs substrates driven
by volume applications, which is not the case for lithium
niobate on insulator (LNOI). From a materials perspective
compared with LiNbO3, LiTaO3 modulators exhibit improved
DC bias stability [8–10], greater tolerance to high optical
power with modulators up to 1.17 W demonstrated [8], and
markedly lower birefringence (δn = 0.004 vs. δn = 0.074 for
LiNbO3 at 1550 nm), simplifying the design of complex pho-
tonic circuits such as arrayed waveguide gratings (AWG) [11],
which are key building blocks for wavelength division multi-
plexing components - as well as enabling broadband electro-
optic combs [12]. Moreover, the platform has been combined
with the copper Damascene process, enabling direct copper-
to-copper bonding with electronic driver integrated circuits
(IC) [8]. Electro-optic bandwidths exceeding 110 GHz have
been demonstrated for LiNbO3 modulators on silicon sub-
strates [7]; however, the high relative permittivity of sili-
con (εr ≈11.7) introduces an inherent mismatch between
the microwave effective index and the optical group index.
Achieving simultaneous impedance and velocity matching un-
der these conditions forces a narrow signal conductor geome-
try, which in turn increases RF conductor losses [13] and lim-
its the practical electrode length to around 6 mm, yielding rel-
atively large half-wave voltages of approximately 4.8 V [7].
Several strategies have been explored to overcome the di-
electric substrate bottleneck.
Replacing the silicon carrier
with a low-permittivity material such as fused silica [14] re-
moves the velocity-matching constraint and allows a wider
electrode geometry, substantially reducing microwave atten-
uation and extending the interaction length.
An alternative route that avoids the complexity of full wafer
bonding is to selectively remove the silicon beneath the elec-
trode region through a substrate undercut process [15]. This
approach retains the standard LTOI starting wafer and wafer-
scale-compatible processing, while effectively suspending
the traveling-wave electrodes in air, which has the lowest-
permittivity and lowest dielectric loss, between the probe con-
tact pads. The suspended membrane decouples the RF mode
from the silicon handle wafer, enabling velocity matching
with a wider electrode geometry and drastically reduced mi-
crowave loss. Here, we implement a silicon undercut process
for LTOI Mach-Zehnder modulators (MZMs) and demon-
strate that the suspended electrode architecture provides a sig-
arXiv:2607.17436v1  [physics.optics]  19 Jul 2026

<!-- page 2 -->
2
nificant improvement in electro-optic bandwidth for a device
with an 8 mm-long modulation section. We further validate
the high-speed performance through intensity-modulation and
direct-detection (IMDD) signaling experiments, achieving a
net data rate of 460 Gbit s−1 for the LiNbO3 platform. Our
results demonstrate that the silicon undercut offers an accessi-
ble, wafer-compatible route to high-performance LTOI mod-
ulators without the need for a dedicated low-permittivity sub-
strate.
RESULTS
Device Design and Fabrication
The fabricated device, shown in Fig. 1 (a), consists of two
8 mm-long suspended electro-optic phase shifters combined
with 1 × 2 multimode-interference (MMI) couplers to form a
push-pull Mach-Zehnder modulator.
The suspended LTOI MZMs are built on a commercial 4-
inch X-cut LTOI wafer (NANOLN), comprising a 600 nm-
thick LiTaO3 single-crystal thin film, a 4.7 µm-thick SiO2
buried oxide layer, and a 525 µm-thick high-resistivity sili-
con substrate. In contrast to approaches requiring a thick sili-
con dioxide insulating layer [16] or a full silicon dioxide sub-
strate [14], the silicon undercut strategy adopted here selec-
tively removes the silicon only beneath the electrode region,
preserving the mechanical integrity of the chip while elim-
inating the dielectric loading of the traveling-wave coplanar
waveguide (CPW).
The optical waveguides are patterned using deep-ultraviolet
(DUV) stepper lithography followed by argon ion beam etch-
ing with a diamond-like-carbon hard mask, yielding an etch
depth of 400 nm with a 200 nm-thick residual slab, consis-
tent with established LTOI fabrication processes [6, 17]. The
modulator arms are clad with a 1.5 µm-thick silicon dioxide
layer deposited by inductively coupled plasma chemical vapor
deposition (ICP CVD) [18]. Traveling-wave gold electrodes
(15 nm Ti / 800 nm Au) are then defined by DUV lithogra-
phy, dry etching, metal evaporation, and lift-off, following the
same process flow as for our previous LTOI devices [6].
The silicon undercut is carried out as a post-electrode pro-
cessing step, as illustrated in Fig. 1 (b). First, via patterns
are defined by ultraviolet (UV) lithography in-between the
capacitively loaded CPW (Fig. 1 (d)). Then trenches with a
depth (Lc) of 80 µm are anisotropically etched through the sil-
icon dioxide and into the silicon substrate using a Bosch deep
reactive-ion etching process. To optimize the silicon undercut
width for proper velocity matching, the chips are separated
from the wafer by silicon deep etching and backside grind-
ing for individual process[19]. Owing to the chip singulation
process and the inverse taper edge coupler, a typical fiber-to-
fiber coupling loss of −5.2 dB is achieved. Subsequently, an
isotropic silicon etch selectively removes the silicon beneath
the electrode region through the vias, leaving the LiTaO3 pho-
tonic layer surrounded by silicon dioxide and the Au elec-
trodes suspended above an air-filled cavity (Fig. 1 (c)). The
undercut width Wc is experimentally optimized on each chip
to achieve phase matching between the microwave and opti-
cal signals. The suspended membrane extends along the full
8 mm modulation section of the modulator.
To preserve the mechanical robustness of the fragile sus-
pended modulation arms, the T-shaped undercut segments are
applied only to the ground electrodes, while the silicon diox-
ide layer beneath the signal electrode remains continuous af-
ter the silicon undercut. Furthermore, combining anisotropic
and isotropic silicon etching reduces the proportion of the mi-
crowave field in the residual silicon, enabling phase matching
with a smaller lateral undercut extent and thereby improving
mechanical stability. Accounting for fabrication tolerances,
impedance matching, and phase matching, the CPW design
parameters are chosen as (Wsig, P, G, LS , WS , LT, WT) = (70,
35, 6, 20, 1.5, 33, 1.5) µm (Fig. 1 (e)). The electrode cross-
section above the air gap closely resembles that of a modula-
tor on a low-permittivity substrate, substantially relaxing the
trade-off between impedance matching and conductor width
that restricts designs on unsuspended silicon.
Electro-optic Modulation Efficiency
The modulation efficiency of the fabricated suspended
MZM is quantified through the half-wave voltage–length
product (VπL). A low-frequency sawtooth voltage waveform
with a peak-to-peak amplitude of 20 V is applied to the mod-
ulator electrodes, and the normalized optical transmission is
recorded simultaneously using a photodetector (DET08CFC)
and an oscilloscope (Rohde & Schwarz RTA4004). The half-
wave voltage is extracted by fitting the measured electro-optic
response curve.
As summarized in Fig. 2 (a), the 8 mm-long suspended
MZM exhibits a half-wave voltage of 5.1 V at a carrier wave-
length of 1550 nm, corresponding to a VπL product of 4 V cm.
The DC bias stability of the device is verified by measuring
VπL as a function of modulation frequency from 10 mHz to
10 kHz, with no appreciable drift observed across this four-
decade range, consistent with the improved DC stability char-
acteristic of the LiTaO3 platform compared to the LiNbO3
platform [20].
High-Frequency Characterization
The high-frequency electrical and electro-optic behaviour
of the suspended MZM is characterized using a vector net-
work analyzer (VNA; ME7838AX, Anritsu Corporation) op-
erating from 70 kHz to 125 GHz, with 110 GHz ground-
signal-ground (GSG) RF probes (T110A-GSG0100, MPI
Corporation).
A two-port short-open-load-through (SOLT)
calibration on a commercial substrate (AC2-2, MPI Corpo-
ration) is used to set the measurement reference planes at the
probe tips.

<!-- page 3 -->
3
(a)
8 mm
(b)
(d)
(e)
50 μm
50 μm
1. LTOI MZM
Si
Au
SiO2
LT 
SiO2
PR
2. Vias lithography 
3. SiO2 and Si deep etching
4. Si release
Wsig
WT
WS
P
G
LS
LT
Wc
Lc
(c)
FIG. 1: Suspended thin-film lithium tantalate Mach-Zehnder modulator fabricated via silicon substrate undercut. (a) Optical micro-
scope image of a fabricated chip showing the 8 mm-long modulator electrode region. (b) Schematic cross-sectional process flow illustrating the
four key fabrication stages: 1. Starting LTOI MZM on Si with Au electrodes and SiO2 cladding; 2. Via lithography to define the release holes;
3. Deep SiO2 and Si dry etching through the vias; 4. Isotropic Si release to suspend the modulation region, leaving a free-standing membrane
comprising the LiTaO3 waveguide, SiO2 cladding, and Au electrodes. Cross-section (c) and top view (d) scanning electron microscope (SEM)
pictures of the modulation region after silicon release confirms successful suspension of the electrode stack. The undercut cavities beneath the
ground electrodes are clearly visible. (e) Modelled suspended CPW geometry, showing the T-shaped ground-electrode segments and the key
design parameters (Wsig, P, G, LS , WS , LT, WT).
Figure 2 (d) shows the electrical S-parameters of the sus-
pended CPW. The reflection coefficient S 11 remains below
−20 dB across the entire measured frequency range, indicat-
ing good impedance matching to the 50 Ωsource. The trans-
mission coefficient S 21 exhibits a gradual roll-off, reaching
−4.3 dB at 120 GHz.
The microwave attenuation, plotted
against the square root of frequency in Fig. 2 (e), follows a
near-linear trend, confirming that conductor (skin-depth) loss
is the dominant dissipation mechanism [21]. The extracted
effective microwave phase index is in close agreement with
the simulated optical group index of the LiTaO3 waveguide at
1550 nm, confirming velocity matching in the high-frequency
region (Fig. 2 (d)).
To measure the electro-optic frequency response, an optical
carrier at 1550 nm is coupled into the modulator. The device
is driven from the first VNA port, while the second port is
connected to a high-speed photodiode that detects the modu-
lated optical signal. The output port of the CPW is terminated
with a 50 Ωresistor via a second RF probe to suppress back-
reflections. De-embedding is applied to shift the measurement
reference planes to the input probe tip and the optical output
of the MZM, using the frequency responses of the probe and
photodiode supplied by their respective manufacturers.
The resulting electro-optic response, shown in Fig. 2 (f),
is normalized to the value at 1 GHz.
The smoothed curve
yields a 3 dB bandwidth of 110 GHz. Compared to previously
reported LTOI modulators on unsuspended silicon substrates
with comparable active lengths [7], the undercut architecture
provides a substantial bandwidth improvement, demonstrat-
ing that selective substrate removal is an effective strategy to
mitigate the velocity-mismatch and conductor-loss penalties
imposed by the high-permittivity silicon carrier, as well as the
potential parasitic surface-conductance loss in the silicon [22].
High-Speed IMDD Signaling with Suspended Lithium Tantalate
MZM
To evaluate the system-level performance of the suspended
lithium-tantalate-on-insulator MZM, we conduct high-speed
intensity-modulation and direct-detection (IMDD) signaling
experiments with the configuration depicted in Fig. 3 (a). An
external-cavity diode laser (ECDL) operating at 1550 nm pro-
vides the optical carrier with an output power of 17.8 dBm.
The polarization is aligned to the quasi-transverse-electric
mode of the LiTaO3 waveguide using a fiber polarization con-
troller (FPC), and lensed fibers are employed for chip-to-fiber
coupling.
Electrical drive signals are generated by a high-speed arbi-
trary waveform generator (AWG; M8199B, Keysight Tech-

<!-- page 4 -->
4
-5
0
5
Drive Voltage (V)
-20
-15
-10
-5
0
EO response (dB)
0
1
2
3
4
VπL (V·cm)
10-2
10⁰
10²
10⁴
Frequency (Hz)
(a)
(b)
(c)
Vπ = 5.1 V
0
0
20
40
60
80
100
120
Frequency (GHz)
-50
-40
-30
-20
-10
Electrical S parameters (dB)
S11
S21
EO Response (dB)
Frequency (GHz)
6
0
2
4
6
8
10
Square root frequency (GHz1/2)
0
2
4
Microwave loss (dB/cm)
(d)
(e)
-2
-1
0
0
20
40
60
80
100
120
-3
Raw data
Smooth fitting
Effective index
2.1
2.2
2.3
2.4
2.5
2.6
2.0
Microwave loss
Microwave index
Optical group index
FIG. 2: Modulation performance characterization of the suspended LiTaO3 Mach-Zehnder modulator. (a) Normalized optical transmis-
sion as a function of applied voltage at a carrier wavelength of 1550 nm, measured at 100 Hz with a sawtooth waveform. A half-wave voltage
of Vπ = 5.1 V is extracted from the fitted curve. Measured (b) VπL and (c) normalized EO response as a function of modulation frequency
from 10 mHz to 10 kHz, confirming DC bias stability with negligible drift across four decades of frequency. (d) Electrical S-parameters (S 11,
orange; S 21, blue) of the suspended traveling-wave CPW up to 120 GHz. (e) Calculated microwave attenuation (blue) and effective microwave
phase index (orange) of the CPW as a function of the square root of frequency. The microwave effective index closely matches the simulated
optical group index (dashed), confirming velocity matching in the high-frequency region. (f) Measured EO response of the suspended MZM,
normalized to the value at 1 GHz. Raw data (light gray) and a smoothed fit (blue) are shown. A 3 dB bandwidth of 110 GHz is achieved.
nologies Inc.)
operating at a sampling rate of 256 GSa/s,
routed through an RF cable and a broadband amplifier (Amp.;
AH15199B, Anritsu Corporation), and applied to the CPW
via a 110 GHz RF probe. Pulse-amplitude modulated (PAM)
waveforms with four, six, and eight levels are generated from
pseudo-random bit sequences with pattern lengths exceeding
217 bits using offline Tx-DSP, which includes root-raised co-
sine pulse shaping with a roll-off factor of β = 0.05, pre-
distortion based on linear minimum-mean-square-error equal-
ization to compensate for the transmitter frequency response
excluding the LTOI MZM, and signal clipping to a peak-to-
average power ratio of 9 dB. The output CPW is terminated
with a 50 Ωresistor through a second 110 GHz RF probe and a
broadband bias-tee, which simultaneously sets the quadrature
operating point of the modulator. The received optical signal
is digitized by a real-time oscilloscope (RTO; UXR 1004A,
Keysight Technologies Inc.) with a 256 GSa/s sampling rate
and 104 GHz analog bandwidth, after direct detection by a
high-speed photodiode (PD; Fraunhofer HHI). Receiver-side
offline DSP includes resampling to two samples per symbol,
timing recovery, Sato equalization, and additional least mean
square-based adaptive equalization.
Signaling experiments are performed at symbol rates rang-
ing from 144 GBd to 208 GBd for PAM4, PAM6, and PAM8.
The measured BER curves as a function of symbol rate are
presented in Fig. 3 (b), together with reference threshold lines
for common forward error correction (FEC), i.e. KP4, hard-
decision FEC with 7 % overhead (7 % HD) [23], as well as
soft-decision FEC with 15 % (15 % SD) and 25 % (25 % SD)
overhead [24]. The BER remains below the 25 % SD-FEC
limit up to 188 GBd for PAM8, up to 192 GBd for PAM6 and
up to 208 GBd for PAM4 signaling. For the latter, the BER
remain below the KP4 limit up to 176 GBd. The achievable
information rate (AIR) is computed via the generalized mu-
tual information (GMI) assuming an additive white Gaussian
noise channel model [25], and the net data rate (NDR) is de-
rived by selecting appropriate FEC codes from the normal-
ized GMI following [26, Table 2]. The results are shown in
Fig. 3 (c). The highest AIR of 485 Gbit s−1 is obtained for
PAM8 at 180 GBd. The maximum NDR of 460 Gbit s−1 is
achieved for PAM8 at a symbol rate of 180 GBd using SD-
FEC, establishing a high single-lane data rate for the lithium
tantalate modulator platform and demonstrating performance
on par with state-of-the-art thin-film lithium niobate modula-
tors [3].

<!-- page 5 -->
5
ECDL
FPC
PD
Tx-DSP
AWG
Rx-DSP
RTO
Suspending LT MZM
144
160
176
192
208
Symbol Rate (Gbd)
10-5
10-4
10-3
10-2
BER
PAM8
PAM6
PAM4
25 % SD
15 % SD
7 % HD
144
160
176
192
208
Symbol Rate (Gbd)
300
350
400
450
AIR / NDR (Gbit/s)
PAM8
PAM6
PAM4
(a)
(b)
(c)
KP4
FIG. 3: Intensity-modulation and direct-detection (IMDD) signaling experiment using the suspended lithium-tantalate-on-insulator
Mach-Zehnder modulator. (a) Experimental setup. An external-cavity diode laser (ECDL) provides the optical carrier, and a fiber polariza-
tion controller (FPC) adjusts the input polarization. The electrical drive signal is synthesized by transmitter digital signal processing (Tx-DSP)
and generated by an arbitrary waveform generator (AWG). The modulated signal is detected by a high-speed photodiode (PD) connected to a
real-time oscilloscope (RTO), followed by offline receiver DSP (Rx-DSP). The chip under test is the suspended LiTaO3 MZM. (b) Measured
bit error ratios (BER) as a function of symbol rate for PAM4 (blue), PAM6 (orange), and PAM8 (green) signals. Horizontal dashed lines
indicate the 25 % and 15 % soft-decision forward error correction (SD-FEC) thresholds and the 7 % hard-decision FEC (HD-FEC) threshold
as well as the threshold for KP4.. The circled marker indicates the operating point achieving the highest net data rate of 460 Gbit s−1. (c)
Achievable information rates (AIR, dashed lines) and corresponding net data rates (NDR, solid lines) as a function of symbol rate for PAM4
(blue), PAM6 (orange), and PAM8 (green).
CONCLUSION
We have demonstrated a silicon substrate undercut ap-
proach to fabricate suspended thin-film lithium tantalate
Mach-Zehnder modulators. By selectively removing the high-
permittivity silicon beneath the traveling-wave electrodes
through via-defined deep silicon etching followed by isotropic
silicon release, the coplanar waveguide (CPW) is effectively
suspended in air, decoupling the microwave mode from the
silicon handle wafer.
This substantially reduces the mi-
crowave conductor losses with a significantly wider electrode
geometry and relaxes the buried silicon dioxide thickness con-
straint for velocity-matching compared to conventional LTOI
devices on unsuspended silicon.
The fabricated 8 mm-long suspended Mach-Zehnder mod-
ulator (MZM) achieves a 3 dB electro-optic bandwidth of
110 GHz with a half-wave voltage of 5.1 V, representing a
marked improvement over non-suspended LTOI modulators
of comparable length. The DC bias stability characteristic of
the LiTaO3 material is fully preserved after substrate release,
as confirmed by VπL measurements spanning more than four
decades in modulation frequency with no appreciable drift.
Leveraging the enhanced electro-optic bandwidth, we achieve
a single-lane net data rate of 460 Gbit s−1 for the LiTaO3 plat-
form using PAM8 signaling in the C-band, on par with the
best results reported for thin-film lithium niobate modulators.
These results confirm that silicon substrate undercut is
a viable and process-compatible route to high-performance
LiTaO3 electro-optic modulators, without requiring wafer
bonding to an alternative substrate.
The approach is fully
compatible with existing LTOI foundry workflows and can
enable manufacturing of electro-optical modulators for next-
generation coherent communications transceivers or linear
drive pluggable optics within data-centers, as well as modu-
lators for 5G/6G radio-over-fiber [27].
Data and Code Availability. Data and code used to produce
the figures within this paper will be available at Zenodo upon

<!-- page 6 -->
6
publication of the manuscript.
Funding.
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
Acknowledgements. We acknowledge the EPFL Center of
MicroNano Technology (CMi) and the Institute of Physics
(IPHYS) cleanroom for supporting sample fabrication.
Competing Interests. C.K. and T.J.K. are co-founders and
shareholders of Luxtelligence SA, St. Sulpice, Switzerland, a
company engaged in electro-optic modulators based on ferro-
electric materials, such as lithium niobate and lithium tanta-
late. The other authors declare no competing interests.
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
[3] E. Berikaa, M. S. Alam, W. Li, S. Bernal, B. Krueger, F. Pittal`a,
and D. V. Plant, IEEE Photonics Technology Letters 35, 850
(2023).
[4] C. St-Arnault, B. Qiu, D. Kita, K. Anzai, C. R. Cole, R. Dick-
son, B. Beggs, N. Ben-Hamida, C. Reimer, and D. V. Plant, pp.
Th4B–2 (2026).
[5] C. St-Arnault, R. Guti´errez-Castrej´on, S. Bernal, E. Berikaa,
Z. Wei, J. Zhang, M. S. Alam, A. Nikic, B. Qiu, B. Krueger,
et al., Journal of Lightwave Technology 43, 3222 (2024).
[6] C. Wang, Z. Li, J. Riemensberger, G. Lihachev, M. Churaev,
W. Kao, X. Ji, J. Zhang, T. Blesin, A. Davydova, et al., Nature
629, 784 (2024), ISSN 1476-4687, URL https://doi.org/
10.1038/s41586-024-07369-1.
[7] C. Wang, D. Fang, J. Zhang, A. Kotz, G. Lihachev, M. Chu-
raev, Z. Li, A. Schwarzenberger, X. Ou, C. Koos, et al., Optica
11, 1614 (2024), URL https://opg.optica.org/optica/
abstract.cfm?URI=optica-11-12-1614.
[8] M. Lin, Z. Li, A. Kotz, H. Larocque, N. Kuznetsov, J. Sun,
Y. Zhang, S. Zheng, J. Riemensberger, C. Koos, et al., Nature
Communications (2026).
[9] K. Powell, X. Li, D. Assumpcao, L. Magalh˜aes, N. Sinclair, and
M. Lonˇcar, Optics Express 32, 44115 (2024).
[10] A. Sayem, S. Z. Uddin, T.-C. Hu, A. Tate, M. Cappuzzo,
R. Kopf, and M. Earnshaw, arXiv preprint arXiv:2602.00922
(2026).
[11] S. U. Hulyal, J. Hu, C. Wang, J. Cai, G. Lihachev, and T. J. Kip-
penberg, Optica 12, 978 (2025), URL https://opg.optica.
org/optica/abstract.cfm?URI=optica-12-7-978.
[12] J. Zhang, C. Wang, C. Denney, J. Riemensberger, G. Lihachev,
J. Hu, W. Kao, T. Bl´esin, N. Kuznetsov, Z. Li, et al., Nature
637, 1096 (2025).
[13] Z. Li, A. Kotz, A. Schwarzenberger, C. Koos, and T. J. Kippen-
berg, arXiv preprint arXiv:2604.14836 (2026).
[14] P. Kharel, C. Reimer, K. Luke, L. He, and M. Zhang, Optica
8, 357 (2021), URL https://opg.optica.org/optica/
abstract.cfm?URI=optica-8-3-357.
[15] G. Chen, K. Chen, R. Gan, Z. Ruan, Z. Wang, P. Huang, C. Lu,
A. P. T. Lau, D. Dai, C. Guo, et al., APL photonics 7 (2022).
[16] X. Xue, Y. Xu, W. Ding, R. Ye, J. Qiu, G. Li, J. Dai, S. Liu,
H. Li, L. Yuan, et al., Optics Express 34, 13111 (2026).
[17] Z. Li, R. N. Wang, G. Lihachev, J. Zhang, Z. Tan, M. Churaev,
N. Kuznetsov, A. Siddharth, M. J. Bereyhi, J. Riemensberger,
et al., Nat. Commun. 14, 4856 (2023), ISSN 2041-1723, URL
https://doi.org/10.1038/s41467-023-40502-8.
[18] Z. Qiu, Z. Li, R. N. Wang, X. Ji, M. Divall, A. Siddharth,
and T. J. Kippenberg, arXiv preprint arXiv:2312.07203 (2024),
URL https://arxiv.org/abs/2312.07203.
[19] J. Liu, G. Huang, R. N. Wang, J. He, A. S. Raja, T. Liu, N. J. En-
gelsen, and T. J. Kippenberg, Nature communications 12, 2236
(2021).
[20] J. Holzgrafe, E. Puma, R. Cheng, H. Warner, A. Shams-Ansari,
R. Shankar, and M. Lonˇcar, Opt. Express 32, 3619 (2024),
URL https://opg.optica.org/oe/abstract.cfm?URI=
oe-32-3-3619.
[21] R. N. Simons, Coplanar waveguide circuits, components, and
systems (John Wiley & Sons, 2004).
[22] D. Lederer and J.-P. Raskin, Solid-State Electronics 47, 1927
(2003).
[23] Recommendation G.975.1, Telecommunication standardization
sector of International Telecommunication Union (2004), URL
https://www.itu.int/rec/T-REC-G.975.1-200402-I/
en.
[24] A. Graell i Amat and L. Schmalen, Forward Error Correction
for Optical Transponders (Springer International Publishing,
Cham, 2020), pp. 177–257, ISBN 978-3-030-16250-4, URL
https://doi.org/10.1007/978-3-030-16250-4_7.
[25] M. Ivanov, C. H¨ager, F. Br¨annstr¨om, A. Graell i Amat, A. Al-
varado, and E. Agrell, IEEE Trans. Inf. Theory 62, 3011 (2016).
[26] Q. Hu, R. Borkowski, Y. Lefevre, J. Cho, F. Buchali, R. Bonk,
K. Schuh, E. De Leo, P. Habegger, M. Destraz, et al., J. Light.
Technol. 40, 3338 (2022).
[27] T. Ren, M. Zhang, C. Wang, L. Shao, C. Reimer, Y. Zhang,
O. King, R. Esman, T. Cullen, and M. Lonˇcar, IEEE photonics
technology letters 31, 889 (2019).

