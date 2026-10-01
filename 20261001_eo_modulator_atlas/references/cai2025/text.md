---
paper_id: cai2025
source_url: https://doi.org/10.1038/s41467-026-69769-3
doi: 10.1038/s41467-026-69769-3
license: https://creativecommons.org/licenses/by/4.0
sha256: 54d92a9d687820f8bbf2cb1b92ae9f958f52251cf6ebb28e9e61494bb49c8e15
pages: 17
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Heterogeneously integrated lithium tantalate-on-silicon nitride modulators for
high-speed communications
Jiachen Cai,1, 2, ∗Alexander Kotz,3, ∗Hugo Larocque,2, ∗Chengli Wang,1, 2 Xinru Ji,2
Junyin Zhang,2 Daniel Drayss,3 Xin Ou,1, † Christian Koos,3, ‡ and Tobias J. Kippenberg2, 4, §
1State Key Laboratory of Materials for Integrated Circuits,
Shanghai Institute of Microsystem and Information Technology, Chinese Academy of Sciences, Shanghai, China
2Institute of Physics, Swiss Federal Institute of Technology Lausanne (EPFL), CH-1015 Lausanne, Switzerland
3Institute of Photonics and Quantum Electronics (IPQ),
Karlsruhe Institute of Technology (KIT), 76131 Karlsruhe, Germany
4Institute of Electrical and Micro engineering, Swiss Federal Institute of Technology,
Lausanne (EPFL), CH-1015 Lausanne, Switzerland
Driven by the prospects of higher bandwidths for
optical interconnects, integrated modulators in-
volving materials beyond those available in sili-
con manufacturing increasingly rely on the Pock-
els eﬀect.
For instance, wafer-scale bonding of
lithium niobate ﬁlms onto ultralow loss silicon ni-
tride photonic integrated circuits provides het-
erogeneous integrated devices with low modula-
tion voltages operating at higher speeds than sil-
icon photonics.
However, in spite of its excel-
lent electro-optic modulation capabilities, lithium
niobate suﬀers from drawbacks such as birefrin-
gence and long-term bias instability.
Among
other available electro-optic materials, lithium
tantalate can overcome these shortcomings with
its comparable electro-optic coeﬃcient, signiﬁ-
cantly improved photostability, low birefringence,
higher optical damage threshold, and enhanced
DC bias stability. Here, we demonstrate wafer-
scale heterogeneous integration of lithium tanta-
late ﬁlms on low-loss silicon nitride photonic in-
tegrated circuits. With this hybrid platform, we
implement modulators that combine the ultralow
optical loss (∼14.2 dB/m), mature processing
and wide transparency of silicon nitride waveg-
uides with the ultrafast electro-optic response of
thin-ﬁlm lithium tantalate. The resulting devices
achieve a 6 V half-wave voltage, and support mod-
ulation bandwidths of up to 100 GHz. We use sin-
gle intensity modulators and in-phase/quadrature
(IQ) modulators to transmit PAM4 and 16-QAM
signals reaching up to 333 and 581 Gbit/second
net data rates, respectively. Our results demon-
strate that lithium tantalate is a viable approach
to broadband photonics sustaining extended opti-
cal propagation, which can uniquely contribute to
technologies such as RF photonics, interconnects,
and analog signal processors.
∗These authors contributed equally.
† ouxin@mail.sim.ac.cn
‡ christian.koos@kit.edu
§ tobias.kippenberg@epﬂ.ch
I.
INTRODUCTION
Photonic integrated circuits (PICs) provide a unique
approach to scaling standardized optoelectronic technolo-
gies. [1, 2]. Among PIC platforms, silicon nitride-based
devices oﬀer a range of features traditionally leveraged in
optical ﬁbers such as low propagation losses, large power
handling [3], and a wide bandgap enabling transparency
across a wide range of optical wavelengths [4–6]. Thicker
nitride waveguides, e.g. manufactured with a photonic
Damascene process, additionally provide strong optical
conﬁnement and easily-achievable anomalous group ve-
locity dispersion [7–10].
These features become cru-
cial while harnessing this material’s optical nonlineari-
ties [11] and have thus lead to demonstrations pertaining
to Kerr frequency combs [10–13], optical frequency con-
version [14, 15], travelling-wave optical parametric am-
pliﬁer [16, 17], and quantum information science [18–
21]. Further deployment of this platform in applications
such as optical communications and microwave photon-
ics [22, 23] requires electro-optic modulation, which sili-
con nitride cannot directly provide due to its amorphous
nature.
Ferroelectric-based integrated circuits readily
provide such features [24–26], and can thus introduce
electro-optic modulation capabilities into silicon nitride
circuits through the heterogeneous integration of mate-
rials like lithium niboate [27, 28]. Improved metrics in
other ferroelectrics such as lithium tantalate, which of-
fer lower birefringence and photorefractive eﬀects [29–
31], motivate applying similar integration methods to
alternative Pockels materials.
Improved economies of
scale for lithium tantalate substrates due to their role
in 5G/6G RF ﬁlters [32] further encourages the use of
this material in integrated modulators.
Here, we de-
velop a wafer-scale bonding process for lithium tanta-
late on silicon nitride PICs.
With this platform, we
demonstrate modulation deﬁned by a VπL = 4.08 V·cm
and a 3-dB bandwidth close to 100 GHz in a push-
pull Mach-Zehnder modulator (MZM). These results en-
able net data transmission rates of 333 Gbit/s when op-
erating the modulators in pulse amplitude modulation
(PAM) schemes.
We further extend this data trans-
mission demonstration to Si3N4-LiTaO3 IQ modulators,
thereby achieving net data transmission rates of 581
arXiv:2508.06265v2  [physics.optics]  5 Sep 2025

<!-- page 2 -->
2
Gbit/s using quadrature amplitude modulation (QAM)
signals.
With performance metrics that can already
compete with those of monolithic lithium-tantalate-on-
insulator (LTOI) PICs [30, 31], these results show that
Si3N4-LiTaO3 devices can implement systems combin-
ing mature Si3N4 PICs with next-generation ferroelectric
thin ﬁlms, thereby enabling both low-loss strong optical
conﬁnement with GHz-rate modulation.
II.
RESULTS
Hybrid Si3N4-LiTaO3 PIC fabrication. Figure 1(a)
outlines the process ﬂow for the hybrid PICs, which starts
with the fabrication of Si3N4 waveguide structures us-
ing the photonic Damascene process [9]. As detailed in
Supplementary Section 1, the process ﬂow begins with
preform formation on a 100 mm-diameter silicon wafer
with 4 µm thick wet thermal oxide by means of deep ul-
traviolet (DUV) stepper lithography, dry etching and re-
ﬂow under high temperatures. Low-pressure chemical va-
por deposition (LPCVD) then deposits a layer of Si3N4,
which subsequently undergoes chemical-mechanical pol-
ishing (CMP), oxide interlayer deposition, and anneal-
ing. The Si3N4 photonic Damascene process is free of
crack formation in the highly tensile LPCVD Si3N4 ﬁlm
and provides high fabrication yields and ultralow optical
propagation losses of O(dB/m) [8]. The patterned sam-
ple supports strip waveguides with a 1 µm width and a
0.5 µm height. In addition, inverse nanotapers allow for
eﬃcient edge coupling between the chip’s waveguides and
lensed ﬁbers.
Our heterogeneous integration approach additionally
involves 100 mm LTOI wafers fabricated by a hydrogen-
based ion-slicing technique [29, 32]. The resulting wafer
stack consists of a 300 nm x-cut lithium tantalate thin
ﬁlm, a 2 µm buried oxide layer and a 525 µm thick sili-
con substrate. Following surface preparation relying on
cleaning and plasma activation, the Si3N4 and LTOI
wafers undergo hydrophilic pre-bonding at room tem-
perature, where van der Waals interactions establish a
preliminary bond between the two wafers. A subsequent
300◦C thermal annealing process then enhances the bond
strength by accelerating the polymerization of silanol (Si-
OH) groups to covalent bonds (Si-O-Si) at the interface.
After bonding, the process continues by thinning the
LiTaO3-based donor substrate. Backside wafer grinding
followed by a tetramethylammonium hydroxide (TMAH)
wet etch entirely removes the substrate’s silicon.
A
buﬀered hydroﬂuoric acid (BHF) solution then elim-
inates the thermal oxide layer, thereby exposing the
LiTaO3 layer and forming a well-conﬁned hybrid opti-
cal waveguide. This sequence entirely lacks plasma-based
processing traditionally used in LiTaO3 nanofabrication,
which for instance includes oxygen-based diamond-like
carbon hardmask etching [33] and argon dry etching
[34]. With such measures, we avoid introducing potential
plasma-induced charges at the Si3N4 PIC wafer’s Si/SiO2
interface, which can lead to parasitic surface conduction
RF losses [35]. To add high-speed coplanar waveguide
(CPW) and heater electrodes to the sample, lithogra-
phy based on a maskless aligner (Heidelberg Instruments
MLA150) ﬁrst deﬁnes the electrode layout.
Thermal
evaporation of 10 nm of titanium and 800 nm of gold
followed by lift-oﬀthen completes the electrode’s fabri-
cation. Argon-based ion beam etching subsequently pat-
terns the LiTaO3 thin ﬁlm to form adiabatic taper tran-
sitions between Si3N4 waveguides and the hybrid Si3N4-
LiTaO3 waveguides. The process also fully removes the
LiTaO3 near the chip facets to improve edge coupling ef-
ﬁciency to the hybrid Si3N4-LiTaO3 PIC. After cladding
deposition, an additional exposure step followed by hy-
droﬂuoric wet etching selectively expose portions of the
sample for landing probes on the manufactured elec-
trodes. As discussed in Supplementary Section 2, these
cladding openings also match the group velocities of the
CPW RF ﬁeld and optical waveguide mode.
Figure 1(b) shows a photograph of a ﬁnalized 100-mm
hybrid Si3N4-LiTaO3 wafer, where a distinct boundary
line delineates the contour of the bonded LiTaO3 ﬁlm.
The central nine stepper ﬁelds are fully-covered by the
ﬁlm, thereby highlighting the excellent fabrication con-
sistency and large-scale yield of the wafer bonding tech-
nique.
To characterize the optical properties of the wafer’s hy-
brid photonic structures, frequency-comb-assisted spec-
troscopy [36] relying on three external-cavity diode lasers
probed optical waveguide loss in fabricated hybrid Si3N4-
LiTaO3 microring resonators.
Figures 1(c,d) show the
derived external, κex, and intrinsic, κ0, loss rates of the
ring’s TE polarization resonances near the C telecom-
munication band.
Here, κ0 increases with optical fre-
quency, which is consistent with the behavior of simi-
lar structures implemented in LiNbO3-on-Si3N4 hetero-
geneous PICs [27, 28].
In such structures, higher op-
tical frequencies result in a smaller mode ﬁeld deﬁned
by a stronger overlap with the higher-index ferroelectric
slab, thus inducing additional bending losses in the rings.
Over the frequency range considered in Fig. 1(d), ﬁnite
element method simulations suggest the lithium tanta-
late ﬁlm holds roughly 48% of the mode’s energy. Fig-
ure 1(e) shows a typical normalized resonance in the
C-band, which indicates a ﬁtted intrinsic linewidth of
72 MHz and a corresponding propagation optical loss of
α ≈14.2 dB/m.
Electro-optic performance. To demonstrate electro-
optic capabilities in our hybrid Si3N4-LiTaO3 platform,
we rely on electro-optic modulators consisting of 6.8 mm-
long MZM pairs operating in a push-pull conﬁguration
with high-speed CPW electrodes. Figure 2(a) shows a
micrograph of such a modulator.
Supplementary Sec-
tion 3 provides simulated data regarding the transmis-
sion of some of its underlying components. The CPW
design features a signal-to-ground spacing that accounts
for trade-oﬀs between metal-induced optical losses and
modulation.
From the cross-sectional SEM images in

<!-- page 3 -->
3
Figure 1. Wafer-scale manufacturing of Si3N4-LiTaO3 photonic integrated circuits. (a) Schematic diagram of the
main steps in the fabrication of heterogeneously integrated Si3N4-LiTaO3 photonic circuits.
(b) Photograph showing the
100 mm Si3N4-LiTaO3 wafer with high fabrication yield. (c) Fitted value of external linewidth, κex, and intrinsic linewidth,
κ0, for a ring resonator with a 2 µm waveguide width and 112 GHz free spectral range. (d) Corresponding ﬁtted κ0 histogram.
(e) Representative normalized resonance with an intrinsic linewidth of 72 MHz.
Figs. 2(b) and the inset diagram in Figs. 2(c), our CPW
design balances out these two factors with a 23 µm signal
electrode width and a 6 µm gap separating the ground
and signal electrodes. Simulations presented in Supple-
mentary Section 2 suggest this conﬁguration yields metal
spacing-induced propagation losses of 0.1 dB/cm and a
voltage length product of VπL ∼5.5 V·cm. As further
discussed in the Supplementary Information, the cho-
sen signal width also aﬀects the modulator’s underlying
bandwidth by altering RF propagation losses and veloc-
ity mismatching between the modulator’s co-propagating
RF and optical ﬁelds.
We ﬁrst characterize the modulator’s electro-optic per-
formance under quasi-DC operation. Coupling 1550 nm
continuous-wave light into the Si3N4-LiTaO3 MZM ex-
hibited a ﬁber-to-chip coupling loss of approximately
5 dB. Figure 2(c) shows the measured output power while
applying a 100 Hz triangle voltage wave via an RF probe.
These results indicate a Vπ = 6 V half-wave voltage for a
6.8 mm long push-pull conﬁguration, which corresponds
to a VπL = 4.08 V·cm voltage-length product.
Addi-
tional data provided in Supplementary Section 4 suggest
this response is consitent down to modulation frequen-
cies of at least 1 Hz and over optical wavelengths ranging

<!-- page 4 -->
4
Y-splitter
a
500 nm
b
c
d
-6
-3
0
3
6
Voltage (V)
0
0.4
0.8
Norm. Trans.
Vπ = 6 V
20
40
60
80
100
Frequency (GHz)
-30
-20
-10
0
S parameter (dB)
Electrical S11
Electro-optic S21
3dB EO bandwidth ~ 100 GHz
500 μm
6.8 mm long LTOD modulator
3 μm
Au
SiO2
G
S
G
50 μm
50 μm
50 μm
Mode Transition Taper
Si3N4
Waveguide
LTOD
Waveguide
- 3dB
- 6dB
0
6 μm 23 μm 6 μm
LiTaO
Si N4
3
3
Figure 2. Hybrid Si3N4-LiTaO3 electro-optic modulators. (a) Optical micrograph of a fabricated 6.8 mm-long hybrid
modulator.
Insets: optical micrographs of the modulator’s underlying components, which include Y-splitters, a coplanar
waveguide electrode, and tapered transitions between Si3N4 waveguides and Si3N4-LiTaO3 waveguides with a simulated 0.28 dB
insertion loss. (b) False-colored scanning electron microscopy (SEM) image of the cross-section of a manufactured Si3N4-LiTaO3
modulator. (c) Normalized transmission of a 6.8 mm-long push-pull MZM versus applied voltage. The inset diagram shows
the high-speed electrode geometry with a 23 µm signal width and a 6 µm electrode gap. (d) Measured electro-optic response
(electro-optic S21) and microwave return loss (electrical reﬂection S11), revealing a high 3-dB bandwidth near 100 GHz.
from 1500 nm to 1630 nm. As highlighted in Fig. 2(d),
probing the modulator’s response with a vector network
analyzer (VNA) provided the modulator’s bandwidth.
The resulting S21 electro-optic response exhibits excellent
ﬂatness within 3 dB across frequencies ranging from 25
MHz to 110 GHz while keeping S11 microwave reﬂections
below −15 dB. These metrics reach levels comparable
to state-of-the-art ﬁgures achieved in other electro-optic
PIC platforms [30, 37, 38], which underscores the signiﬁ-
cant potential of Si3N4-LiTaO3 for applications requiring
high-speed electro-optic modulation.
Data transmission experiments. To demonstrate the

<!-- page 5 -->
5
e
FPC
ECL
PD
EDFA
90:10
BPF
LTOD modulator
VOA
Rx-DSP
Power
Meter
mW
RTO
b
c
d
Symbol rate in GBd
AIR / NDR (Gbit/s)
AIR & NDR
160GBd (320 Gbit/s)
200 GBd (400 Gbit/s)
Counts (a.u.)
Amplitude (a.u.)
Amplitude (a.u.)
Time axis
Counts (a.u.)
Bit error ratio (BER)
 BER 
10
140
160
180
200
140
250
400
350
300
160
180
200
-1
7% HD-FEC
A
15% SD-FEC
25% SD-FEC
10-2
10-3
10-4
Symbol rate in GBd
Net: 333 Gbit/s
NDR
AIR
Tx-DSP
AWG
Time axis
a
B
B
A
Figure 3.
Transmission experiments using intensity-modulation and direct detection (IMDD) (a) Experimen-
tal setup.
ECL: External cavity laser; FPC: Fiber polarization controller; AWG: Arbitrary-waveform generator; Tx-DSP:
Transmitter-digital signal processing (oﬄine); EDFA: Erbium-doped ﬁber ampliﬁer; BPF: Bandpass ﬁlter; VOA: Variable opti-
cal attenuator; PD: Photodiode; RTO: High-speed real-time oscilloscope (256 GSa/s, 105 GHz); Rx-DSP: Receiver-digital signal
processing (oﬄine). (b) Measured bit error ratio (BER) versus symbol rate for PAM4 transmission, with horizontal dashed lines
indicating the thresholds for soft-decision forward error correction with 15 % and 25 % coding overhead (15 % SD-FEC, 25 %
SD-FEC) and for hard-decision FEC with 7 % coding overhead (7 % HD-FEC). The eye diagrams of two selected data points
A
○and B
○are shown in panels (d) and (e). (c) Achievable information rates (AIR, orange dashed line) and corresponding net
data rates (NDR, solid brown line) of the IMDD measurements, showing that the highest achieved NDR is 333 Gbit/s using
a PAM4 signals at a symbol rate of 192 GBd. (d), (e) Eye diagrams (left) at selected symbol rate of 160 GBd PAM4 (d) and
200 GBd PAM4 (e), labeled by A
○and B
○in panel (b), along with the associated histograms (right) of the reconstructed signal
amplitudes in the center of the corresponding symbol slot.
viability and performance of our Si3N4-LiTaO3 platform,
we use our devices in high-speed optical data communi-
cation experiments, covering both intensity-modulation
and direct-detection (IMDD) and coherent modulation
schemes. Figure 3(a) shows the setup for the IMDD ex-
periment. Herein, a tunable external-cavity laser (ECL)
operating in the C-band provides the optical carrier. A
ﬁber-based polarization controller (FPC) then adjusts
the output polarization before coupling to the quasi-TE
mode, having a dominant electric ﬁeld parallel to the
substrate plane, of the Si3N4-LiTaO3 MZM via a pair of
lensed ﬁbers. We drive the MZM with electrical PAM4
signals, that are synthesized by oﬄine digital signal pro-
cessing (Tx-DSP) based on pseudo-random bit sequences
(PRBS) and root-raised cosine (RRC) pulse-shaping ﬁl-
ters [30], and that are converted to the analogue domain
using a high-speed arbitrary waveform generator (AWG,
M8199B, Keysight Technologies Inc.). 20 cm-long RF ca-
bles send the drive signals to the coplanar transmission
line of the MZM via a ﬁrst impedance-matched probe in
a ground-signal-ground conﬁguration.
A second probe
terminates the transmission line with a 50 Ωcoaxial ter-
mination. Tx-DSP compensates for frequency-dependent
RF loss up to the input of the feeding probe by imple-
menting a linear minimum-mean-square-error predistor-
tion.
An erbium-doped ﬁber ampliﬁer (EDFA) brings
the rather weak output signal of -10 dBm from the MZM
to a power level near 10 dBm compatible with the re-
ceiver.
This receiver comprises a high-speed photodi-
ode (PD, Finisar Corp.), which is directly connected to
a high-speed real-time oscilloscope (RTO, UXR 1004A,
Keysight Technologies Inc.) with a sampling rate of 256
GSa/s and an analogue bandwidth of 105 GHz.
The
EDFA is followed by an optical bandpass ﬁlter (BPF)
to suppress out-of-band ampliﬁed spontaneous-emission
(ASE) noise and by a variable optical attenuator (VOA,
LTB-1, EXFO Inc.) that adjusts the optical power to 10
dBm, the maximum input power accepted by the photo-
diode. At the receiver, we use oﬄine digital signal pro-
cessing (Rx-DSP) to extract and demodulate the PAM4
data. As further elaborated in Supplementary Section 5,
the Rx-DSP suite comprises standard algorithms such as
timing recovery, linear Sato equalization, and an addi-
tional decision-directed least-mean-square equalizer.
In our experiment, we used this setup to operate the
device at diﬀerent PAM4 symbol rates between 144 GBd
and 200 GBd.
Figure 3(b) provides the resulting bit
error ratios (BER) along with the BER thresholds for

<!-- page 6 -->
6
diﬀerent forward error correction (FEC) schemes indi-
cated by dashed horizontal lines.
For symbol rates of
160 GBd (line rate 320 Gbit/s) and lower, the BER
is below the threshold for hard-decision FEC with 7 %
coding overhead (7 % HD-FEC) [39], and it stays be-
low the threshold for soft-decision FEC (SD-FEC) with
15 % coding overhead [40, Table 7.5] up to symbol rates
of 192 GBd (line rate 384 Gbit/s). For 200 GBd (line
rate 400 Gbit/s), we ﬁnd a BER value that is still com-
patible with soft-decision forward-error correction (SD-
FEC) with 25 % coding overhead (25 % SD-FEC) [40,
Table 7.5]. Figure 3(d) and Fig. 3(e) depict the recon-
structed eye diagrams of the 160 GBd and the 200 GBd
signals, respectively, along with the histograms taken at
the center of the symbol slot.
The data points corre-
sponding to these eye diagrams are marked A
○and B
○
in Fig. 3(b). While the signal quality clearly leaves room
for improvement, the demonstrated symbol rates and line
rates can already compete with those achieved by stan-
dalone lithium-tantalate-on-insulator (LTOI) MZM [30].
To quantify the information transfer eﬃciency of our
transmission system, we derive the generalized mutual in-
formation (GMI) of the transmitted signals from the log-
likelihood ratios (LLRs) of the received symbols based
on an additive white Gaussian noise (AWGN) channel
model [42]. Figure 3(c) shows the resulting achievable
information rates (AIR), which correspond to the prod-
uct of the symbol rate and the GMI of each symbol and
provide an upper bound for the transmission capacity
of the system.
To estimate practically achievable net
data rates (NDR), we additionally have to include penal-
ties introduced by typical FEC codes. By comparing the
normalized GMI (NGMI) estimated in our experiments
to the NGMI thresholds provided in [43], we select suit-
able FEC codes and evaluate the NDR by multiplying the
line rates and the associated FEC code rates. The high-
est NDR of 333 Gbit/s is obtained for a symbol rate of
192 GBd (line rate 384 Gbit/s) in combination with 15 %
SD-FEC, while the AIR at this symbol rate amounts to
349 Gbit/s.
Besides MZM and IMDD signaling, our Si3N4-LiTaO3
platform also supports more advanced systems, such as
IQ modulators (IQM) for coherent communications. Fig-
ure 4(a) shows a micrograph of such an IQM. The device
consists of an input waveguide followed by a Y-splitter
leading to two 13.5 mm-long MZMs, such as the one
shown in Fig. 2(a). The individual MZMs are DC-biased
at their minimum transmission point using two 110 GHz
bias-Tees (BT110R-C, SHF Communication Technolo-
gies AG). A heater-based phase shifter, which can pro-
duce a drift-free thermo-optic phase shift [44], then sets
the relative phase between the signals generated in the
in-phase (I) and the quadrature (Q) arms of the IQM
to 90◦.
Thereafter, a power combiner, like the one in
Fig. 2(a), synthesizes the output signal with two modu-
lated quadratures. To demonstrate the viability of the
device, we use the setup shown in Fig. 4(b) to trans-
mit quadrature phase-shift keying (QPSK) and 16-state
quadrature-amplitude modulation (16QAM) signals at
symbol rates between 144 GBd and 204 GBd. The setup
shares several similarities with the one used for IMDD
transmission shown Fig. 3(a). Here, the two AWG chan-
nels however individually drive the I and Q arms of the
IQM. After modulation, ampliﬁcation and bandpass ﬁl-
tering, the modulated signal enters a 90◦optical hybrid
module, where it interferes with a continuous-wave local
oscillator provided by a second ECL to down-convert the
optical signal to four electrical baseband signals. Sub-
sequent balanced photodiodes (BPD, Fraunhofer HHI)
then detect the resulting waveforms. Finally, as clariﬁed
in Supplementary Section 5, oﬄine receiver DSP recovers
and evaluates the signals.
Figure 4(c) shows the measured BER along with the
BER thresholds for diﬀerent FEC schemes indicated by
dashed horizontal lines.
Figure 4(d) provides the cor-
responding AIR and NDR. For QPSK signals, we ob-
tained reliable BER values only for our measurements at
symbol rates of 176 GBd and higher, whereas the mea-
surements at 144 GBd and 160 GBd led to BER val-
ues below 1.24 × 10−6, which corresponds to the detec-
tion limit for the length of the recorded waveforms. As
shown in Fig. 4(e), the constellation diagram at 160 GBd
features a constellation signal-to-noise ratio (CSNR) of
15.8 dB, which would correspond to an estimated BER
of 3.5×10−10 [45, Eq. 2.18]. At the highest QPSK symbol
rate of 204 GBd, we measure a BER of 7 × 10−3, which
is still below the BER threshold for SD-FEC schemes
with 25 % overhead. In this case, the CSNR amounts
to 8.5 dB as displayed in the corresponding constella-
tion diagram from Fig. 4(f). Here, the 75 GHz electrical
3 dB-bandwidth of the AWG [46] mainly limits the per-
formance of this experiment. As indicated in Fig. 4(c),
the highest symbol rate achieved for 16-QAM signaling
amounts to 176 GBd and leads to a measured BER of
3.08 × 10−2 – still below the BER threshold for SD-FEC
codes with 25 % overhead. This corresponds to a line
rate of 704 Gbit/s.
The corresponding NDR of up to
581 Gbit/s validates the feasibility of telecommunication
systems based on hybrid Si3N4-LiTaO3 circuits.
Fig-
ure 4(g) shows the corresponding constellation diagram,
from which we measure a CNSR of 12.0 dB.
To the best of our knowledge, these experiments rep-
resent the ﬁrst demonstration of LiTaO3-on-Si3N4-based
high symbol-rate communication experiments and repre-
sents an important milestone that has not been reached
for other similar heterogeneous photonic platforms to
date [47–49] and exceeds the previously reported symbol
rates below 100 GBd in hybrid silicon lithium niobate
coherent modulators [50].
III.
DISCUSSION
Alternative hybrid Si3N4-LiTaO3 waveguide geome-
tries can further improve some of their performance ﬁg-
ures. However, such improvements might come at the ex-

<!-- page 7 -->
7
a
b
13.5 mm 
Heater
Heater
Y-combiner
Y-splitter
In-phase Arm
Quadrature Arm
500 μm
e
f
g
FPC
BPD
EDFA
LTOD IQ modulator
TO-PS
Q-arm
I-arm
BPF
BPF
ECL
EDFA
ECL
90°
OH
RTO
Amp.
Bias
AWG
Tx-DSP
Rx-DSP
c
10
140
160
180
200
140
160
180
200
-6
7% HD-FEC
Detection limit
15% SD-FEC
25% SD-FEC
Symbol rate in GBd
-2
10
10
-3
10-4
10-5
10-1
Bit error ratio (BER)
250
Symbol rate in GBd
d
AIR / NDR (Gbit/s)
650
550
450
350
NDR
AIR
16QAM
QPSK
QPSK
16QAM
CSNR: 12 dB
176 GBd 16QAM
204 GBd QPSK
CSNR: 8.5 dB
160 GBd QPSK
CSNR: 15.8 dB
A
A
B
B
C
C
Figure 4. Hybrid Si3N4-LiTaO3 IQ modulator for coherent data transmission. (a) Optical micrograph of a 13.5 mm-
long Si3N4-LiTaO3 IQ modulator. The waveguides are highlighted in red for better visibility. (b) Schematics of the setup used
for the coherent communications experiment. AWG: Arbitrary-waveform generator; Amp.: RF ampliﬁer; Tx-DSP: Transmitter-
digital signal processing (oﬄine); ECL: External cavity laser; FPC: Fiber polarization controller; EDFA: Erbium-doped ﬁber
ampliﬁer; BPF: Bandpass ﬁlter; TO-PS: Thermo-optic phase shifter; ECL: External-cavity laser; 90◦OH: 90◦optical hybrid;
BPD: Balanced photodiode; RTO: High-speed real-time oscilloscope (256 GSa/s, 105 GHz); Rx-DSP: Receiver-digital signal
processing (oﬄine). (c) Measured bit error ratio (BER) versus symbol rate for QPSK (brown line) and 16-QAM (dark blue
line) signals with horizontal dashed lines indicating the thresholds for soft-decision forward error correction having 15 % and
25 % coding overhead (15 % SD-FEC, 25 % SD-FEC) and for hard-decision FEC with 7 % coding overhead (7 % HD-FEC). The
gray dashed line at the bottom indicates the detection limit, below which our experiment cannot reliably measure BER due to
the limited length of the recorded signals (223 samples). This limit corresponds to 13 detected bit errors at a symbol rate of
160 GBd with QPSK signals during the recording period. The resulting BER falls within the 99 % conﬁdence interval, ranging
from half to twice the measured BER [41]. The constellation diagrams of three selected data points A
○, B
○and C
○are shown
in panels (e), (f) and (g). (d) Achievable information rate (AIR, dashed lines) and corresponding net data rate (NDR, solid
lines) versus symbol rate for QPSK (brown lines) and 16-QAM (dark blue lines) signals. (e)-(g) Constellation diagrams for
QPSK signals at symbol rates of 160 GBd (e) and 204 GBd (f), and for the 16-QAM signals at a symbol rate of 176 GBd (g).
The corresponding data points are marked by red dashed circles and labeled with A
○, B
○and C
○, respectively, in panel (c).
pense of other metrics. For instance, relying on a thinner
bonded LiTaO3 ﬁlm can reduce slab losses in the result-
ing hybrid waveguide, thereby leading to lower optical
propagation losses and ring resonators with higher qual-
ity factors. However, this narrower ﬁlm will also reduce
the overlap of the propagating mode with the electro-
optic material, thus resulting in lower modulation eﬃ-
ciency. Besides such underlying design tradeoﬀs, some

<!-- page 8 -->
8
features might require speciﬁc waveguide dimensions that
will constrain other ﬁgures of merit to ﬁxed ranges. For
example, a waveguide hosting optical nonlinearities must
often verify speciﬁc dispersive relations determined by
its cross-section [8, 10, 11]. As a result, distinct features
required of the PIC by speciﬁc end-uses will ultimately
dictate the geometry and hence the performance of its
waveguides.
Our Si3N4-LiTaO3 platform could potentially beneﬁt
from additional functionalities brought by platform ex-
tensions.
Namely, adapting its bonding process could
lead to the integration of III-V active optical materi-
als [51–54] and potentially to a multilayer platform fea-
turing both III-V components [55] and Si3N4-LiTaO3
hybrid waveguides.
Incorporating distinct Si3N4 and
LiTaO3 waveguide layers would also introduce bene-
ﬁts, such as eliminating slab losses in Si3N4 waveguides
and increasing mode overlap with LiTaO3 for enhanced
electro-optic coupling strengths. However, these beneﬁts
will come at the cost of those attributed to the minimal
processing of the LiTaO3 ﬁlm in our fabrication ﬂow.
To summarize, we introduced a photonic platform re-
lying on wafer-scale bonding of LiTaO3 ﬁlms on foundry-
compatible Si3N4 PICs. Its fabrication process reaches
a 100% bonding yield across the wafer’s nine central de-
vice ﬁelds. Compared to other monolithic electro-optic
PIC platforms [30, 37, 38], Si3N4-LiTaO3 circumvents
specialized LiTaO3 etching by relying on a standard-
ized Si3N4 high-volume process ﬂow. Resulting circuits
preserve low optical losses at telecommunication wave-
lengths while also accommodating broadband modula-
tors featuring high eﬃciencies and a ﬂat electro-optic re-
sponse over modulation rates extending up to 100 GHz.
Data transmission experiments conﬁrm the practicality
of these features by demonstrating net data rates exceed-
ing 500 Gbit/s. Ease of access to high-volume production
of Si3N4-LiTaO3 PICs reinforces its prospects in ﬁeld-
deployable applications not only limited to high-speed
optical communications, but that also include microwave-
to-optical transducers [56–58] and fast tunable LiDAR
[59, 60]. Built-in integration with thick Si3N4waveguides
also enables a new generation of on-chip systems leverag-
ing both electro-optic and ultralow loss waveguides rang-
ing from microwave oscillators [22, 23], to potential in-
terfaces with Kerr frequency combs [8, 9].
Note:
During the preparation of this manuscript, two similar
manuscripts [48, 49] were posted on the arXiv pre-print repository,
but with distinct fabrication processes and experimental results.
Author Contributions:
J.C., C.W., and X.J. fabricated the
Si3N4-LiTaO3 PICs.
J.C., C.W, and J.Z. measured the losses
and EO responses of the PICs. A.K. measured the frequency re-
sponse. A.K. and D.D. conceived and performed the data trans-
mission experiments together with C.K. and jointly discussed the
results.
J.C., A.K., D.D., and H.L. analyzed the data.
T.J.K.,
C.K., X.O. and H.L. supervised all aspects of the project.
H.L.
and J.C. prepared the content of the manuscript in assistance with
contributions and discussions provided by all the authors.
Funding Information and Disclaimer:
This work was sup-
ported by funding from the Swiss National Science Foundation un-
der grant agreement No.
216493 (HEROIC)), by funding from
the German Research Foundation via the projects PACE (#
403188360) and GOSPEL (# 403187440), and by the European In-
novation Council (EIC) via the project ELLIPTIC (# 101187515).
Acknowledgments: The PICs were fabricated in the EPFL Cen-
ter of MicroNanoTechnology (CMi). The LTOI wafers were fabri-
cated in Shanghai Novel Si Integration Technology (NSIT) and the
SIMIT-CAS.
Competing Interests:
C.K. and T.J.K are co-founders and
shareholders of Luxtelligence SA, St. Sulpice, Switzerland, a com-
pany engaged in electro-optic modulators based on ferroelectric ma-
terials.
Data and Code Availability Statement: The code and data
used to produce the plots within this work will be released on the
repository Zenodo upon publication of this preprint.
[1] B. Li, W. Jin, L. Wu, L. Chang, H. Wang, B. Shen,
Z. Yuan, A. Feshali, M. Paniccia, K. J. Vahala, and J. E.
Bowers, Reaching ﬁber-laser coherence in integrated pho-
tonics, Opt. Lett. 46, 5201 (2021).
[2] Y. Liu, Z. Qiu, X. Ji, A. Lukashchuk, J. He, J. Riemens-
berger, M. Hafermann, R. N. Wang, J. Liu, C. Ronning,
and T. J. Kippenberg, A photonic integrated circuit-
based erbium-doped ampliﬁer, Science 376, 1309 (2022).
[3] K. Liu, N. Jin, H. Cheng, N. Chauhan, M. W. Puckett,
K. D. Nelson, R. O. Behunin, P. T. Rakich, and D. J. Blu-
menthal, Ultralow 0.034 db/m loss wafer-scale integrated
photonics realizing 720 million q and 380 µw threshold
brillouin lasing, Opt. Lett. 47, 1855 (2022).
[4] X. Lu, G. Moille, A. Singh, Q. Li, D. A. Westly, A. Rao,
S.-P. Yu, T. C. Briles, S. B. Papp, and K. Srinivasan,
Milliwatt-threshold visible–telecom optical parametric
oscillation using silicon nanophotonics, Optica 6, 1535
(2019).
[5] X. Lu, G. Moille, A. Rao, D. A. Westly, and K. Srini-
vasan, On-chip optical parametric oscillation into the vis-
ible: generating red, orange, yellow, and green from a
near-infrared pump, Optica 7, 1417 (2020).
[6] D. J. Blumenthal, Photonic integration for UV to IR ap-
plications, APL Photonics 5, 020903 (2020).
[7] M. H. P. Pfeiﬀer, J. Liu, A. S. Raja, T. Morais, B. Gha-
diani, and T. J. Kippenberg, Ultra-smooth silicon nitride
waveguides based on the damascene reﬂow process: fab-
rication and loss origins, Optica 5, 884 (2018).
[8] J. Liu, G. Huang, R. N. Wang, J. He, A. S. Raja, T. Liu,
N. J. Engelsen, and T. J. Kippenberg, High-yield, wafer-
scale fabrication of ultralow-loss, dispersion-engineered
silicon nitride photonic circuits, Nat. Commun. 12, 2236
(2021).
[9] X. Ji, J. Liu, J. He, R. N. Wang, Z. Qiu, J. Riemens-
berger, and T. J. Kippenberg, Compact, spatial-mode-
interaction-free, ultralow-loss, nonlinear photonic inte-
grated circuits, Commun. Phys. 5, 84 (2022).
[10] X.
Ji,
X.
Li,
Z.
Qiu,
R.
N.
Wang,
M.
Divall,
A.
Gelash,
G.
Lihachev,
and
T.
J.
Kippenberg,
Copper-impurity-free photonic integrated circuits en-
able
deterministic
soliton
microcombs,
Preprint
at
https://arxiv.org/abs/2504.18195 (2025).

<!-- page 9 -->
9
[11] M. H. P. Pfeiﬀer, A. Kordts, V. Brasch, M. Zervas,
M. Geiselmann, J. D. Jost, and T. J. Kippenberg, Pho-
tonic damascene process for integrated high-q microres-
onator based nonlinear photonics, Optica 3, 20 (2016).
[12] G. Moille, M. Leonhardt, D. Paligora, N. Englebert,
F. Leo, J. Fatome, K. Srinivasan, and M. Erkintalo, Para-
metrically driven pure-Kerr temporal solitons in a chip-
integrated microcavity, Nat. Photonics 18, 617 (2024).
[13] G. Moille, P. Shandilya, A. Niang, C. Menyuk, G. Carter,
and K. Srinivasan, Versatile optical frequency division
with Kerr-induced synchronization at tunable microcomb
synthetic dispersive waves, Nat. Photonics 19, 36 (2025).
[14] X. Lu, G. Moille, Q. Li, D. A. Westly, A. Singh, A. Rao,
S.-P. Yu, T. C. Briles, S. B. Papp, and K. Srinivasan,
Eﬃcient telecom-to-visible spectral translation through
ultralow power nonlinear nanophotonics, Nat. Photonics
13, 593 (2019).
[15] X. Lu, G. Moille, A. Rao, D. A. Westly, and K. Srini-
vasan, Eﬃcient photoinduced second-harmonic genera-
tion in silicon nitride photonics, Nat. Photonics 15, 131
(2021).
[16] J. Riemensberger, N. Kuznetsov, J. Liu, J. He, R. N.
Wang, and T. J. Kippenberg, A photonic integrated
continuous-travelling-wave parametric ampliﬁer, Nature
612, 56 (2022).
[17] P. Zhao, V. Shekhawat, M. Girardi, Z. He, V. Torres-
Company, and P. A. Andrekson, Ultra-broadband opti-
cal ampliﬁcation using nonlinear integrated waveguides,
Nature 640, 918 (2025).
[18] X. Lu, Q. Li, D. A. Westly, G. Moille, A. Singh, V. Anant,
and K. Srinivasan, Chip-integrated visible-telecom en-
tangled photon pair source for quantum communication,
Nat. Phys. 15, 373 (2019).
[19] A. Singh, Q. Li, S. Liu, Y. Yu, X. Lu, C. Schneider,
S. Höﬂing, J. Lawall, V. Verma, R. Mirin, S. W. Nam,
J. Liu, and K. Srinivasan, Quantum frequency conversion
of a quantum dot single-photon source on a nanophotonic
chip, Optica 6, 563 (2019).
[20] V. D. Vaidya, B. Morrison, L. G. Helt, R. Shahrokshahi,
D. H. Mahler, M. J. Collins, K. Tan, J. Lavoie, A. Re-
pingon, M. Menotti, N. Quesada, R. C. Pooser, A. E.
Lita, T. Gerrits, S. W. Nam, and Z. Vernon, Broad-
band quadrature-squeezed vacuum and nonclassical pho-
ton number correlations from a nanophotonic device, Sci.
Adv. 6, eaba9186 (2020).
[21] H. Aghaee Rad, T. Ainsworth, R. N. Alexander, B. Al-
tieri,
M. F. Askarani,
R. Baby,
L. Banchi,
B. Q.
Baragiola, J. E. Bourassa, R. S. Chadwick, I. Chara-
nia, H. Chen, M. J. Collins, P. Contu, N. DâĂŹArcy,
G. Dauphinais, R. De Prins, D. Deschenes, I. Di Luch,
S. Duque, P. Edke, S. E. Fayer, S. Ferracin, H. Fer-
retti, J. Gefaell, S. Glancy, C. GonzÃąlez-Arciniegas,
T. Grainge, Z. Han, J. Hastrup, L. G. Helt, T. Hill-
mann,
J. Hundal,
S. Izumi,
T. Jaeken,
M. Jonas,
S. Kocsis, I. Krasnokutska, M. V. Larsen, P. Laskowski,
F. Laudenbach, J. Lavoie, M. Li, E. Lomonte, C. E.
Lopetegui, B. Luey, A. P. Lund, C. Ma, L. S. Mad-
sen, D. H. Mahler, L. Mantilla CalderÃşn, M. Menotti,
F. M. Miatto, B. Morrison, P. J. Nadkarni, T. Naka-
mura, L. Neuhaus, Z. Niu, R. Noro, K. Papirov, A. Pe-
sah, D. S. Phillips, W. N. Plick, T. Rogalsky, F. Ror-
tais, J. Sabines-Chesterking, S. Safavi-Bayat, E. Sazhaev,
M. Seymour, K. Rezaei Shad, M. Silverman, S. A. Srini-
vasan, M. Stephan, Q. Y. Tang, J. F. Tasker, Y. S. Teo,
R. B. Then, J. E. Tremblay, I. Tzitrin, V. D. Vaidya,
M. Vasmer, Z. Vernon, L. F. S. S. M. Villalobos, B. W.
Walshe, R. Weil, X. Xin, X. Yan, Y. Yao, M. Zamani Ab-
nili, and Y. Zhang, Scaling and networking a modular
photonic quantum computer, Nature 638, 912 (2025).
[22] I. Kudelin, W. Groman, Q.-X. Ji, J. Guo, M. L. Kelleher,
D. Lee, T. Nakamura, C. A. McLemore, P. Shirmoham-
madi, S. Haniﬁ, H. Cheng, N. Jin, L. Wu, S. Halladay,
Y. Luo, Z. Dai, W. Jin, J. Bai, Y. Liu, W. Zhang, C. Xi-
ang, L. Chang, V. Iltchenko, O. Miller, A. Matsko, S. M.
Bowers, P. T. Rakich, J. C. Campbell, J. E. Bowers, K. J.
Vahala, F. Quinlan, and S. A. Diddams, Photonic chip-
based low-noise microwave oscillator, Nature 627, 534
(2024).
[23] Y. He, L. Cheng, H. Wang, Y. Zhang, R. Meade, K. Va-
hala, M. Zhang, and J. Li, Chip-scale high-performance
photonic microwave oscillator, Sci. Adv. 10, eado9570
(2024).
[24] C. Wang, M. Zhang, X. Chen, M. Bertrand, A. Shams-
Ansari, S. Chandrasekhar, P. Winzer, and M. Lončar,
Integrated lithium niobate electro-optic modulators op-
erating at CMOS-compatible voltages, Nature 562, 101
(2018).
[25] M. Li, J. Ling, Y. He, U. A. Javid, S. Xue, and Q. Lin,
Lithium niobate photonic-crystal electro-optic modula-
tor, Nat. Commun. 11, 4123 (2020).
[26] H. Larocque, D. L. P. Vitullo, A. Sludds, H. Sattari,
I. Christen, G. Choong, I. Prieto, J. Leo, H. Zarebidaki,
S. Lohani, B. T. Kirby, Ö. Soykal, M. Soltani, A. H.
Ghadimi, D. R. Englund, and M. Heuck, Photonic crys-
tal cavity iq modulators in thin-ﬁlm lithium niobate, ACS
Photonics 11, 3860 (2024).
[27] M. Churaev, R. N. Wang, A. Riedhauser, V. Snigirev,
T. Blésin, C. Möhl, M. H. Anderson, A. Siddharth,
Y. Popoﬀ, U. Drechsler, D. Caimi, S. Hönl, J. Riemens-
berger, J. Liu, P. Seidler, and T. J. Kippenberg, A het-
erogeneously integrated lithium niobate-on-silicon nitride
photonic platform, Nat. Commun. 14, 3499 (2023).
[28] V. Snigirev, A. Riedhauser, G. Lihachev, M. Churaev,
J. Riemensberger, R. N. Wang, A. Siddharth, G. Huang,
C. Möhl, Y. Popoﬀ, U. Drechsler, D. Caimi, S. Hönl,
J. Liu, P. Seidler, and T. J. Kippenberg, Ultrafast tunable
lasers using lithium niobate integrated photonics, Nature
615, 411 (2023).
[29] C. Wang, Z. Li, J. Riemensberger, G. Lihachev, M. Chu-
raev, W. Kao, X. Ji, J. Zhang, T. Blesin, A. Davydova,
Y. Chen, K. Huang, X. Wang, X. Ou, and T. J. Kippen-
berg, Lithium tantalate photonic integrated circuits for
volume manufacturing, Nature 629, 784 (2024).
[30] C. Wang, D. Fang, J. Zhang, A. Kotz, G. Lihachev,
M. Churaev, Z. Li, A. Schwarzenberger, X. Ou, C. Koos,
and T. J. Kippenberg, Ultrabroadband thin-ﬁlm lithium
tantalate modulator for high-speed communications, Op-
tica 11, 1614 (2024).
[31] M. Lin, Z. Li, A. Kotz, H. Larocque, J. Riemens-
berger,
C.
Koos,
and
T.
J.
Kippenberg,
Copper
damascene
process-based
high-performance
thin
ﬁlm
lithium
tantalate
modulators,
Preprint
at
https://arxiv.org/abs/2505.04755 (2025).
[32] Y. Yan, K. Huang, H. Zhou, X. Zhao, W. Li, Z. Li, A. Yi,
H. Huang, J. Lin, S. Zhang, M. Zhou, J. Xie, X. Zeng,
R. Liu, W. Yu, T. You, and X. Ou, Wafer-Scale Fabrica-
tion of 42o Rotated Y-Cut LiTaO3-on-Insulator (LTOI)
Substrate for a SAW Resonator, ACS Appl. Electron.

<!-- page 10 -->
10
Mater. 1, 1660 (2019).
[33] Z. Li, R. N. Wang, G. Lihachev, J. Zhang, Z. Tan,
M. Churaev, N. Kuznetsov, A. Siddharth, M. J. Bereyhi,
J. Riemensberger, and T. J. Kippenberg, High density
lithium niobate photonic integrated circuits, Nat. Com-
mun. 14, 4856 (2023).
[34] J. Kassabov, E. Atanassova, D. Dimitrov, and E. Gora-
nova, Argon plasma treatment eﬀects on Si-SiO2 struc-
tures, Solid-State Electron. 31, 147 (1988).
[35] M. Shen, L. Yang, Y. Xu, and H. X. Tang, Parasitic
conduction loss of lithium niobate on insulator platform,
Appl. Phys. Lett. 124, 101107 (2024).
[36] P. Del’Haye, O. Arcizet, M. L. Gorodetsky, R. Holzwarth,
and T. J. Kippenberg, Frequency comb assisted diode
laser spectroscopy for measurement of microcavity dis-
persion, Nat. Photonics 3, 529 (2009).
[37] M. Xu, M. He, H. Zhang, J. Jian, Y. Pan, X. Liu,
L. Chen, X. Meng, H. Chen, Z. Li, X. Xiao, S. Yu, S. Yu,
and X. Cai, High-performance coherent optical modula-
tors based on thin-ﬁlm lithium niobate platform, Nat.
Commun. 11, 3911 (2020).
[38] C. Han, Z. Zheng, H. Shu, M. Jin, J. Qin, R. Chen,
Y. Tao, B. Shen, B. Bai, F. Yang, Y. Wang, H. Wang,
F. Wang, Z. Zhang, S. Yu, C. Peng, and X. Wang,
Slow-light silicon modulator with 110-ghz bandwidth,
Sci. Adv. 9, eadi5339 (2023).
[39] Forward error correction for high bit-rate DWDM sub-
marine systems, Recommendation G.975.1 (Telecommu-
nication standardization sector of International Telecom-
munication Union, 2004).
[40] A. Graell i Amat and L. Schmalen, Forward error cor-
rection for optical transponders, in Springer Handbook of
Optical Networks, edited by B. Mukherjee, I. Tomkos,
M. Tornatore, P. Winzer, and Y. Zhao (Springer Inter-
national Publishing, Cham, 2020) pp. 177–257.
[41] M. Jeruchim, Techniques for estimating the bit error rate
in the simulation of digital communication systems, IEEE
J. Sel. Areas Commun. 2, 153 (1984).
[42] M. Ivanov, C. Häger, F. Brännström, A. Graell i Amat,
A. Alvarado, and E. Agrell, On the information loss of
the max-log approximation in bicm systems, IEEE Trans.
Inf. Theory 62, 3011 (2016).
[43] Q. Hu, R. Borkowski, Y. Lefevre, J. Cho, F. Buchali,
R. Bonk, K. Schuh, E. De Leo, P. Habegger, M. De-
straz, N. Del Medico, H. Duran, V. Tedaldi, C. Funck,
Y. Fedoryshyn, J. Leuthold, W. Heni, B. Baeuerle, and
C. Hoessbacher, Ultrahigh-net-bitrate 363 Gbit/s PAM-
8 and 279 Gbit/s polybinary optical transmission using
plasmonic mach-zehnder modulator, J. Light. Technol.
40, 3338 (2022).
[44] S. Sun, M. He, M. Xu, S. Gao, Z. Chen, X. Zhang,
Z. Ruan, X. Wu, L. Zhou, L. Liu, C. Lu, C. Guo, L. Liu,
S. Yu, and X. Cai, Bias-drift-free mach-zehnder modula-
tors based on a heterogeneous silicon and lithium niobate
platform, Photon. Res. 8, 1958 (2020).
[45] D. A. De Arruda Mello and F. A. Barbosa, Digital Coher-
ent Optical Systems: Architecture and Algorithms, Opti-
cal Networks (Springer International Publishing, 2021).
[46] M8199B 256 GSa/s Arbitrary Waveform Generator,
Keysight Technologies, Inc. (2024), version 1.2.
[47] Z. Li, Y. Chen, S. Wang, F. Xu, Q. Xu, J. Zhang, Q. Zhu,
W. Yue, X. Ou, Y. Cai, and M. Yu, Lithium niobate
electro-optical modulator based on ion-cut wafer scale
heterogeneous bonding on patterned SOI wafers, Photon.
Res. 13, 106 (2025).
[48] M. Niels, T. Vanackere, E. Vissers, T. Zhai, P. Nenezic,
J. Declercq, C. Bruynsteen, S. Niu, A. Moerman, O. Cay-
tan, N. Singh, S. Lemey, X. Yin, S. Janssen, P. Ver-
heyen, N. Singh, D. Bode, M. Davi, F. Ferraro, P. Ab-
sil, S. Balakrishnan, J. Van Campenhout, G. Roelkens,
B. Kuyken, and M. Billet, A high-speed heterogeneous
lithium tantalate silicon photonics platform, Preprint at
https://arxiv.org/abs/2503.10557 (2025).
[49] M. A. Rahman, F. Valdez, V. Mere, C. O. de Beeck,
P.
Wuytens,
and
S.
Mookherjea,
High-performance
hybrid
lithium
niobate
electro-optic
modulators
in-
tegrated with low-loss silicon nitride waveguides on
a wafer-scale silicon photonics platform, Preprint at
https://arxiv.org/abs/2504.00311 (2025).
[50] Z. Wang, G. Chen, Z. Ruan, R. Gan, P. Huang, Z. Zheng,
L. Lu, J. Li, C. Guo, K. Chen, and L. Liu, Silicon-Lithium
Niobate Hybrid Intensity and Coherent Modulators Us-
ing a Periodic Capacitively Loaded Traveling-Wave Elec-
trode, ACS Photonics 9, 2668 (2022).
[51] C. Xiang, J. Guo, W. Jin, L. Wu, J. Peters, W. Xie,
L. Chang, B. Shen, H. Wang, Q.-F. Yang, D. Kinghorn,
M. Paniccia, K. J. Vahala, P. A. Morton, and J. E. Bow-
ers, High-performance lasers for fully integrated silicon
nitride photonics, Nat. Commun. 12, 6650 (2021).
[52] C. Xiang, J. Liu, J. Guo, L. Chang, R. N. Wang,
W. Weng, J. Peters, W. Xie, Z. Zhang, J. Riemensberger,
J. Selvidge, T. J. Kippenberg, and J. E. Bowers, Laser
soliton microcombs heterogeneously integrated on silicon,
Science 373, 99 (2021).
[53] J. Sun, J. Lin, M. Zhou, J. Zhang, H. Liu, T. You, and
X. Ou, High-power, electrically-driven continuous-wave
1.55-µm Si-based multi-quantum well lasers with a wide
operating temperature range grown on wafer-scale InP-
on-Si (100) heterogeneous substrate, Light Sci. Appl. 13,
71 (2024).
[54] X. Xie, C. Wei, X. He, Y. Chen, C. Wang, J. Sun,
L. Jiang, J. Ye, X. Zou, W. Pan, and L. Yan, A 3.584
Tbps coherent receiver chip on InP-LiNbO3 wafer-level
integration platform, Light Sci. Appl. 14, 172 (2025).
[55] C. Xiang, W. Jin, O. Terra, B. Dong, H. Wang, L. Wu,
J. Guo, T. J. Morin, E. Hughes, J. Peters, Q.-X. Ji, A. Fe-
shali, M. Paniccia, K. J. Vahala, and J. E. Bowers, 3D
integration enables ultralow-noise isolator-free lasers in
silicon photonics, Nature 620, 78 (2023).
[56] J. Holzgrafe, N. Sinclair, D. Zhu, A. Shams-Ansari,
M. Colangelo, Y. Hu, M. Zhang, K. K. Berggren, and
M. Lončar, Cavity electro-optics in thin-ﬁlm lithium nio-
bate for eﬃcient microwave-to-optical transduction, Op-
tica 7, 1714 (2020).
[57] M. Shen, J. Xie, Y. Xu, S. Wang, R. Cheng, W. Fu,
Y. Zhou, and H. X. Tang, Photonic link from single-ﬂux-
quantum circuits to room temperature, Nat. Photonics
18, 371 (2024).
[58] H. K. Warner, J. Holzgrafe, B. Yankelevich, D. Barton,
S. Poletto, C. J. Xin, N. Sinclair, D. Zhu, E. Sete, B. Lan-
gley, E. Batson, M. Colangelo, A. Shams-Ansari, G. Joe,
K. K. Berggren, L. Jiang, M. J. Reagor, and M. Lončar,
Coherent control of a superconducting qubit using light,
Nat. Phys. 21, 831 (2025).
[59] B. Li, Q. Lin, and M. Li, Frequency–angular resolving li-
dar using chip-scale acousto-optic beam steering, Nature
620, 316 (2023).
[60] A. Siddharth, S. Bianconi, R. N. Wang, Z. Qiu, A. S.

<!-- page 11 -->
11
Voloshin, M. J. Bereyhi, J. Riemensberger, and T. J. Kip-
penberg, Ultrafast tunable photonic-integrated extended-
dbr pockels laser, Nat. Photonics 19, 709 (2025).

<!-- page 12 -->
Supplementary Information for: Heterogeneously integrated lithium
tantalate-on-silicon nitride modulators for high-speed communications
Jiachen Cai,1, 2, ∗Alexander Kotz,3, ∗Hugo Larocque,2, ∗Chengli Wang,1, 2 Xinru Ji,2
Junyin Zhang,2 Daniel Drayss,3 Xin Ou,1, † Christian Koos,3, ‡ and Tobias J. Kippenberg2, 4, §
1State Key Laboratory of Materials for Integrated Circuits,
Shanghai Institute of Microsystem and Information Technology, Chinese Academy of Sciences, Shanghai, China
2Institute of Physics, Swiss Federal Institute of Technology Lausanne (EPFL), CH-1015 Lausanne, Switzerland
3Institute of Photonics and Quantum Electronics (IPQ),
Karlsruhe Institute of Technology (KIT), 76131 Karlsruhe, Germany
4Institute of Electrical and Micro engineering, Swiss Federal Institute of Technology,
Lausanne (EPFL), CH-1015 Lausanne, Switzerland
Contents
1. Photonic Damascene process for integrated silicon nitride waveguides
2
2. Simulations for silicon nitride-lithium tantalate Mach-Zehnder modulators
2
A. Trade-oﬀbetween Vπ · L and optical loss
2
B. Wave velocity matching
3
3. Simulations for tapered mode transitions
4
4. Microwave transmission and modulation eﬃciency
5
5. Digital signal processing at the receiver for silicon nitride-lithium tantalate modulator-based communication
experiments
6
∗These authors contributed equally.
† ouxin@mail.sim.ac.cn
‡ christian.koos@kit.edu
§ tobias.kippenberg@epﬂ.ch

<!-- page 13 -->
2
1.
Photonic Damascene process for integrated silicon nitride waveguides
Si3N4
SiO2
Si
Resist
Planarization
DUV Lithography
Dry Etch
Si3N4 LPCVD
Preform Reflow
Cladding Deposition
Figure S1. Wafer-scale photonic Damascene process for low-loss Si3N4 photonic integrated circuits. The process
ﬂow includes deep ultraviolet (DUV) stepper lithography, ﬂuoride dry etching, preform reﬂow, high-density Si3N4 deposition,
chemical mechanical polishing (CMP), and cladding deposition.
The process begins with a standard wet oxide substrate (4 µm SiO2
/525 µm Silicon). The waveguide and ﬁller
patterns are deﬁned on the top silica using an ASML PAS 5500/350C DUV stepper with a 248 nm wavelength light
source. The micron-order ﬁller patterns are designed as quasi-intersecting line structures to mitigate the tensile stress
associated with high density Si3N4 deposition. These patterns are then transferred into the underlying wet oxide via
ﬂuorine-based dry etching to a depth of 500 nm. A subsequent high-temperature annealing step at 1250◦C is performed
to promote oxide reﬂow and improve sidewall smoothness. After annealing, a 700 nm-thick Si3N4 layer is deposited
into the trenches using LPCVD. A dedicated-adjusted CMP process is implemented to remove the redundant material
outside the trench, as well as enabling ultralow surface roughness on the top surface of Si3N4 waveguides. A small
amount of oxide interlayer is then deposited by an inductively-coupled-plasma CVD (ICPCVD) with SiCl4 gas as
the precursor. An additional 1200◦C annealing is conducted to eliminate absorption losses associated with hydrogen
impurities. Finally, a second CMP step is adopted to ensure sub-nanometer surface roughness for subsequent wafer
bonding of lithium tantalate on Damascene silicon nitride wafers.
2.
Simulations for silicon nitride-lithium tantalate Mach-Zehnder modulators
A.
Trade-oﬀbetween Vπ · L and optical loss
Optical simulations (COMSOL multiphysics) were ﬁrst employed to calculate the mode proﬁles in Si3N4 -LiTaO3
hybrid waveguides. Given the fabrication process from the main text and Section 1, the thickness of LiTaO3 and
Si3N4 is chosen to be 300 nm and 500 nm, respectively. Figure S2(a) represents the electric ﬁeld distribution of the
hybrid optical mode. The conﬁned mode is divided roughly equally between the Si3N4 ridge waveguide and the lithium
tantalate thin ﬁlm. As stated in the Discussion part of the main text, alternative mode distributions are feasible by
altering the ﬁlm thickness, but it involves changes in the preparation of the lithium tantalate wafer, such as adjusting
the ion-implantation dose and annealing temperature during the corresponding smart-cut process [1, 2]. Furthermore,
modifying the ﬁlm thickness also introduces a performance trade-oﬀbetween actuation voltage and optical insertion
loss.
Eﬃcient electro-optic (EO) modulation can be achieved by adopting a close enough electrode-to-waveguide
spacing at the expense of larger optical absorption. Here, the device geometry is well optimized using ﬁnite element
simulations. Assuming that the propagation mode is transverse electric (TE), the estimated loss related to various
electrode gaps can be extracted from the imaginary part of the eﬀective mode index (neff),
α = 0.1 × | log10(e−4π
λ ·Im(neff ))| (dB/cm),
(1)

<!-- page 14 -->
3
a
5 V/m per div
Air cladding
Au
Wet Oxide
5.5
6
6.5
7
7.5
Electrode Gap (μm)
4
5
6
7
8
9
VπL(V·cm)
0
0.1
0.4
Optical Loss (dB/cm)
Width = 1.00 μm
Width = 1.25 μm
Width = 1.50 μm
Width = 1.75 μm
Width = 2.00 μm
5.5
6
6.5
7
7.5
Electrode Gap (μm)
LTOD Mode field
b
c
neff = 1.8202
0.2
0.3
Figure S2. Static ﬁeld simulation of silicon nitride-lithium tantalate Mach-Zehnder modulators. (a) Simulated
optical mode ﬁeld in the cross-section of the hybrid LiTaO3 -Si3N4 waveguide.
(b) Calculated optical loss and (c) Vπ · L
versus diﬀerent electrode gap values and Si3N4 waveguide widths. The hollow circles mark the estimated performance for the
fabricated Si3N4 -LiTaO3 modulators reported in this work, with a waveguide width of 1 µm and an electrode gap of 6 µm.
where λ represents the wavelength of the selected mode. Figure S2(b) shows how the electrode gap aﬀects transmission
loss for various Si3N4 waveguide widths. The resulting voltage-length products (Vπ ·L) for the push-pull conﬁguration
can be expressed as [3]:
Vπ · L = 1
2
λ
n3r33Γ,
(2)
where n is the refractive index of lithium tantalate and r33 is the EO coeﬃcient in the crystallographic c-axis. The
mode overlap Γ can be given by
Γ =
R R Ez(x,z)
V
· |ez(x, z)|2dxdz
R R
|ez(x, z)|2dxdz
,
(3)
where ez(x, z) and Ez(x, z) represent the horizontal electric ﬁeld components of the optical TE mode and actuation
ﬁeld from the electrodes, respectively. As plotted in Fig. S2(c), the 1 µm -wide waveguide shows higher modulation
eﬃciency because of a larger optical mode overlap with the lithium tantalate thin ﬁlm. Based on the above analysis,
we opt for a device geometry featuring a signal-ground spacing of 6 µm and a Si3N4 waveguide width of 1 µm to obtain
a high modulation eﬃciency with negligible insertion loss.
B.
Wave velocity matching
To sustain interactions between propagating microwaves and optical modes over long distances, the velocities of
these two ﬁelds must be similar [4]. For a ﬁxed optical waveguide geometry, such a requirement can be satisﬁed by a
suitable traveling wave electrode design. We simulate the eﬀective refractive index (neff,MW ) of a propagating RF
ﬁeld in high-speed electrodes using Ansys HFSS. The neff,MW can be extracted by phase unwrapping the transmission
S21 of coplanar waveguide (CPW) electrodes:
neff = c0
unwrap(ang(S21))
2πωMW Lele
,
(4)
where c0 is the speed of light in vacuum, ωRF is the RF modulation frequency and Lele is the length of the CPW
electrodes.
The group refractive index (ng,optical) of the waveguide’s optical TE mode is obtained using Ansys
Lumerical Mode. Figure S3 implies a near-perfect alignment between the optical ng and the microwave neff above
a modulation frequency of 40 GHz, thereby verifying the required phase-matching behavior for an Si3N4 -LiTaO3
modulator with a high EO bandwidth.

<!-- page 15 -->
4
0
10
20
30
40
50
60
70
Frequency (GHz)
2
2.2
2.4
2.6
2.8
3
3.2
Effective Index neff
 neff, MW = 2.19
 ng, optical = 2.185
Sim. microwave effective index
Sim. optical group index
Figure S3. Velocity mismatch between microwave and lightwave. Simulated optical group index and microwave eﬀective
index, showing a negligible disparity in the high modulation frequency range.
3.
Simulations for tapered mode transitions
Si3N4
LiTaO3
SiO2
1500
1520
1540
1560
1580
1600
1620
1640
0.8
0.9
1
Nrom. Trans.
-100
0
100
x (μm)
-10
0
10
y (μm)
0.1
0.3
0.5
0.7
0.9
Wavelength (nm)
E(a.u.)
i
ii
iii
Si3N4
Taper
Hybrid LTOD waveguide
i
ii
iii
Si3N4 waveguide
Mode transition
LTOD waveguide
a
b
c
Figure S4. Simulated transmission of silicon nitride-lithium tantalate waveguide transitions. (a) Schematic dia-
gram and corresponding FDTD simulations of the adiabatic coupling from the Si3N4 waveguide to the hybrid Si3N4 -LiTaO3
waveguide. Simulated (b) transmission spectrum and (c) electric ﬁeld distribution of the adiabatic taper.
Low-loss optical mode coupling is achieved using an adiabatic taper transition [5], enabling eﬃcient single-mode
transition from Si3N4 waveguides to hybrid Si3N4 -LiTaO3 waveguides (Fig. S4(a)). Based on the fabricated Si3N4 -
LiTaO3 structure, the coupling design exhibits a 100 µm-long inverse taper and a 500 nm-wide tip, which are con-
structed in the etched LiTaO3 ﬁlm. As illustrated in Figs.
S4(b,c), this tapered design facilitates high coupling

<!-- page 16 -->
5
eﬃciency across a broad wavelength range (1500 nm to 1640 nm) for Si3N4 -LiTaO3
modulators, achieving a low
insertion loss of near 0.28 dB per facet.
4.
Microwave transmission and modulation eﬃciency
0
1
2
3
4
5
6
7
RF Loss (dB/cm)
Vπ
101
102
103
104
105
Frequency (Hz)
5.5
6.5
7.5
  (V)
-10
-5
0
5
10
Modulation Voltage (V)
0
0.2
0.4
0.6
0.8
1
Norm. Trans.
1 Hz
100 Hz
1 kHz
10 kHz
100 kHz
b
a
1500
1520
1540
1560
1580
1600
1620
1640
Wavelength (nm)
  (V)
Vπ
5.5
6.5
7.5
Sim. data
Exp. data
c
d
0
1
2
3
4
5
6
7
8
Square root Frequency (GHz1/2)
6 μm 23 μm 6 μm
Au
800 nm
Exp. data
Figure S5. Modulator microwave transmission and modulation eﬃciency. (a) Measured co-planar waveguide mi-
crowave losses on a square-root frequency axis.
(b) Measured transmission of the Si3N4 -LiTaO3
modulators driven by
modulation frequencies from 1 Hz to 100 kHz. Extracted modulator Vπ value at (c) diﬀerent modulation frequencies at a ﬁxed
optical 1550 nm optical wavelength and (d) diﬀerent optical wavelengths at a ﬁxed 100 Hz modulation frequency.
We ﬁrst characterized the RF attenuation properties of the fabricated CPW-type modulators. Figure S5(a) presents
the extracted microwave loss of the long gold electrode on the Si3N4 -LiTaO3
platform, using a 67 GHz vector
network analyzer (VNA). The loss proﬁle demonstrates performance comparable to state-of-the-art ultrabroadband
EO modulators reported in literature [6, 7]. Additionally, the square-root frequency dependence of RF loss in ﬁgure
S5(a) indicates the dominant ohmic loss mechanism (α ∝√fMW ), conﬁrming that our fabrication techniques can
eﬀectively prevent the devices from parasitic-capacitance-induced loss [8, 9].
To further demonstrate the dynamic performance of Si3N4 -LiTaO3 Mach-Zehnder modulators, we then investigated
their EO performance under low-frequency RF modulation ranging from 1 Hz to 100 kHz (Fig. S5(b)). Despite the
inﬂuence of ferroelectric hysteresis, we extracted a stable half-wave voltage (Vπ) with an average Vπ value of 6.1 V
(Fig. S5(c)) from the experimental data. Similarly, we studied the wavelength response of the modulator performance
driven by a 100 Hz triangle wave signal (Fig. S5(d)), which features excellent agreement with our simulation results
(red dashed line). This wavelength-dependent behavior is due to the increased optical mode expansion in the lithium
tantalate thin ﬁlm at longer wavelengths, leading to an enhanced microwave-optical ﬁeld interaction.

<!-- page 17 -->
6
5.
Digital signal processing at the receiver for silicon nitride-lithium tantalate modulator-
based communication experiments
In the data transmission experiments discussed in the main manuscript, the optical signals were detected by
photodiodes and the resulting photocurrents were digitized by a high-speed real-time oscilloscope (UXR 1004A,
Keysight Technologies Inc.) operating at a sampling rate of 256 GSa/s with an analog bandwidth of 105 GHz. A
total of 223 samples corresponding to a time interval of approximately 33 µs were recorded. Oﬄine, non-data-aided
signal processing was employed to extract the transmitted data.
For both IMDD and coherent transmission schemes, the digitized signal was initially resampled to two samples
per symbol. Timing recovery was then performed using a feedforward timing recovery algorithm, as described in [10]
and [11], to estimate and correct the timing oﬀset of the received signal. An adaptive receive ﬁlter, implemented as
a time-domain linear equalizer, was subsequently applied.
In the IMDD case, linear Sato equalization [12] was utilized, followed by a linear post-equalizer based on the
decision-directed least-mean-squares (DD-LMS) algorithm [13] to recover the transmitted data.
For coherent communications, a linear equalizer based on the constant modulus algorithm [14] was applied. This
was followed by frequency oﬀset compensation using a phase increment estimation algorithm [15], which corrects
for the frequency mismatch between the optical carrier at the transmitter and the local oscillator at the receiver.
Residual phase errors, originating from the phase noise of the transmitter and receiver lasers, was mitigated using the
blind phase search algorithm [16]. Finally, a DD-LMS-based linear post-equalizer [13] was employed to recover the
complex-valued transmitted symbols.
Supplementary References
[1] G. Besnard, B.-Y. Nguyen, and C. Maleville, Smart cut™technology: from substrate enginnering to advanced 3d integra-
tion, in 2022 International Conference on IC Design and Technology (ICICDT) (2022) pp. 81–83.
[2] X.-Q. Feng and Y. Huang, Mechanics of smart-cut® technology, Int. J. Solids Struct. 41, 4299 (2004).
[3] Y. Liu, H. Li, J. Liu, S. Tan, Q. Lu, and W. Guo, Low v&#x03c0; thin-ﬁlm lithium niobate modulator fabricated with
photolithography, Opt. Express 29, 6320 (2021).
[4] Y. Hu, D. Zhu, S. Lu, X. Zhu, Y. Song, D. Renaud, D. Assumpcao, R. Cheng, C. J. Xin, M. Yeh, H. Warner, X. Guo,
A. Shams-Ansari, D. Barton, N. Sinclair, and M. Loncar, Integrated electro-optics on thin-ﬁlm lithium niobate, Nat. Rev.
Phys. 7, 237 (2025).
[5] M. Churaev, R. N. Wang, A. Riedhauser, V. Snigirev, T. Bl´esin, C. M¨ohl, M. H. Anderson, A. Siddharth, Y. Popoﬀ,
U. Drechsler, D. Caimi, S. H¨onl, J. Riemensberger, J. Liu, P. Seidler, and T. J. Kippenberg, A heterogeneously integrated
lithium niobate-on-silicon nitride photonic platform, Nat. Commun. 14, 3499 (2023).
[6] C. Wang, D. Fang, J. Zhang, A. Kotz, G. Lihachev, M. Churaev, Z. Li, A. Schwarzenberger, X. Ou, C. Koos, and T. J.
Kippenberg, Ultrabroadband thin-ﬁlm lithium tantalate modulator for high-speed communications, Optica 11, 1614 (2024).
[7] M. Xu, M. He, H. Zhang, J. Jian, Y. Pan, X. Liu, L. Chen, X. Meng, H. Chen, Z. Li, X. Xiao, S. Yu, S. Yu, and X. Cai,
High-performance coherent optical modulators based on thin-ﬁlm lithium niobate platform, Nat. Commun. 11, 3911 (2020).
[8] M. Shen, L. Yang, Y. Xu, and H. X. Tang, Parasitic conduction loss of lithium niobate on insulator platform, Appl. Phys.
Lett. 124, 101107 (2024).
[9] P. Yang, S. Sun, Y. Zhang, R. Cao, H. He, H. Xue, and F. Liu, High-bandwidth lumped mach-zehnder modulators based
on thin-ﬁlm lithium niobate, Photonics 11, 10.3390/photonics11050399 (2024).
[10] S. Barton and Y. Al-Jalili, A symbol timing recovery scheme based on spectral redundancy, in IEE Colloquium on Advanced
Modulation and Coding Techniques for Satellite Communications (1992) pp. 3/1–3/6.
[11] P. Matalla, M. S. Mahmud, C. F¨ullner, C. Koos, W. Freude, and S. Randel, Hardware comparison of feed-forward clock
recovery algorithms for optical communications, in OFC 2021, Th1A.10 (2021).
[12] Y. Sato, A method of self-recovering equalization for multilevel amplitude-modulation systems, IEEE Trans. Commun. 23,
679 (1975).
[13] S. Randel, D. Pilori, S. Corteselli, G. Raybon, A. Adamiecki, A. Gnauck, S. Chandrasekhar, P. Winzer, L. Altenhain,
A. Bielik, and R. Schmid, All-electronic ﬂexibly programmable 864-Gb/s single-carrier PDM-64-QAM, in OFC 2014,
Th5C.8 (2014).
[14] S. Moshirian, S. Ghadami, and M. Havaei, Blind channel equalization, arXiv:1208.2205.
[15] A. Leven, N. Kaneda, U.-V. Koc, and Y.-K. Chen, Frequency estimation in intradyne reception, IEEE Photon. Technol.
Lett. 19, 366 (2007).
[16] T. Pfau, S. Hoﬀmann, and R. Noe, Hardware-eﬃcient coherent digital receiver concept with feedforward carrier recovery
for m -QAM constellations, J. Light. Technol. 27, 989 (2009).

