---
paper_id: wang2024a
source_url: https://doi.org/10.1364/optica.537730
doi: 10.1364/optica.537730
license: CC-BY-4.0
sha256: 47b20a56d7f95cdd99caf9ee3db0a63440a80e74205895de79aea3d7d37654fe
pages: 7
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Ultrabroadband thin-film lithium tantalate modulator for high-speed communications
Chengli Wang,1, 2, ∗Dengyang Fang,3, ∗Junyin Zhang,1, 2, ∗Alexander Kotz,3 Grigory Lihachev,1, 2 Mikhail
Churaev,1, 2 Zihan Li,1, 2 Adrian Schwarzenberger,3 Xin Ou,4, † Christian Koos,3, ‡ and Tobias Kippenberg1, 2, §
1Institute of Physics, Swiss Federal Institute of Technology Lausanne (EPFL), CH-1015 Lausanne, Switzerland
2Center of Quantum Science and Engineering, EPFL, CH-1015 Lausanne, Switzerland
3Institute of Photonics and Quantum Electronics (IPQ),
Karlsruhe Institute of Technology (KIT), 76131 Karlsruhe, Germany
4National Key Laboratory of Materials for Integrated Circuits,
Shanghai Institute of Microsystem and Information Technology, Chinese Academy of Sciences, Shanghai, China
The continuous growth of global data traffic over the
past three decades, along with advances in disaggregated
computing architectures, presents significant challenges
for optical transceivers in communication networks and
high-performance computing systems. Specifically, there
is a growing need to significantly increase data rates
while reducing energy consumption and cost.
High-
performance optical modulators based on materials such
as InP, thin-film lithium niobate (LiNbO3), or plasmonics
have been developed, with LiNbO3 excelling in high-speed
and low-voltage modulation. Nonetheless, the widespread
industrial adoption of thin film LiNbO3 remains com-
pounded by the rather high cost of the underlying ’on
insulator’ substrates – in sharp contrast to silicon pho-
tonics, which can benefit from strong synergies with
high-volume
applications
in
conventional
microelec-
tronics.
Here, we demonstrate an integrated 110 GHz
modulator using thin-film lithium tantalate (LiTaO3)
— a material platform that is already commercially
used for millimeter-wave filters and that can hence build
upon technological and economic synergies with existing
high-volume applications to offer scalable low-cost manu-
facturing. We show that the LiTaO3 photonic integrated
circuit based modulator can support 176 GBd PAM8
transmission at net data rates exceeding 400 Gbit/s,
while exhibiting a lower bias drift compared to LiNbO3.
Moreover, we show that using silver electrodes can reduce
microwave losses compared to previously employed gold
electrodes. Our demonstration positions LiTaO3 modula-
tor as a novel and highly promising integration platform
for next-generation high-speed,
energy-efficient,
and
cost-effective transceivers.
The relentless increase in global data traffic, driven by the
widespread use of novel technologies such as 5G and artifi-
cial intelligence (AI), has created significant challenges for
transceivers at all levels of optical networks [1, 2]. These
challenges include increased transmission rates along with
reduced energy consumption and costs. Over the previous
years, silicon photonics has been widely deployed in the op-
∗These authors contributed equally.
† ouxin@mail.sim.ac.cn
‡ christian.koos@kit.edu
§ tobias.kippenberg@epfl.ch
tical transceiver market, mainly driven by the cost-efficiency
of the underlying silicon-on-insulator (SOI) substrates and the
amenability to high-volume production of photonic integrated
circuits (PIC) using technically mature CMOS processes [3–
5]. However, on a technical level, silicon photonic electro-
optic modulators have to rely on free-carrier dispersion to
overcome the intrinsic lack of Pockels-type nonlinearities on
bulk silicon. This approach is currently reaching its physi-
cal limits [5, 6] in terms of bandwidth, power consumption,
and impairments such as free-carrier absorption and modu-
lation nonlinearity, especially given the future demand for
highly efficient transceivers that offer line rates of 1.6 Tbit/s
or more [1].
Apart from silicon, substantial efforts have been made
towards developing high-performance optical modulators
across various material platforms, such as indium phosphide
(InP) [7], thin-film lithium niobate [8–10], plasmonic and
silicon-organic hybrid (SOH) [11–13] and other platforms
[14–16]. Ferroelectric thin-film lithium niobate platform of-
fers low optical and microwave losses, high optical power
handling, as well as high Pockels coefficient, while enabling
linear high-speed modulation at low voltage levels without
performance degradation over time [8, 10, 17].
However,
despite tremendous research progress in device design and
demonstrations, it is still an open question whether thin-film
LiNbO3 modulators can achieve market penetration on the
same scale as silicon photonics does today. One reason is the
high cost of LiNbO3-on-insulator (LNOI) wafers, which is a
major obstacle towards adoption of the technology in cost-
sensitive transceiver markets.
Specifically, LNOI substrate
technology cannot rely on any high-volume applications out-
side photonics — unlike silicon photonics and the underlying
silicon-on-insulator (SOI) substrates, which were driven by
significant investments into CMOS technology over the previ-
ous three decades.
In contrast to that, another ferroelectric material, lithium
tantalate, has achieved mass production due to its wide appli-
cation in RF filters for 5G [18]. This already existing sub-
stantial fabrication volume has allowed to mature the technol-
ogy and to address cost challenges when adopting LiTaO3-
on-insulator (LTOI) as a platform for optical modulators. Re-
cently, the first low-loss LiTaO3 PICs have been demon-
strated using diamond-like carbon (DLC) as a hardmask [19],
and equivalent or even superior performance of LiTaO3 com-
pared to LiNbO3 has been demonstrated. Specifically, the
optical birefringence LiTaO3 (∆n = ne −no = 0.004) is
arXiv:2407.16324v2  [physics.optics]  28 Oct 2024

<!-- page 2 -->
2
G
S
G
Y-splitter (combiner)
2 μm
5 μm
(a)
0
10
20
30
40
50
60
Frequency (GHz)
2
2.2
2.4
2.6
2.8
3.0
neff
Ag
LT
SiO2
Si
0
2
4
6
8
0
2
4
6
8
RF loss (dB/cm)
(b)
RF Refractive index
Square root Frequency (GHz-¹/²)
0.7 dB cm-1 GHz-1/2
0.58 dB cm-1 GHz-1/2
Silver
Gold
ng
(c)
3 μm
600 nm
6 mm
z
y
RF
Optical
6 μm
4.7 μm
LiTaO3
= 2.25 
=  2.22 @ 50GHz
Silver electrode
FIG. 1. Thin-film Lithium Tantalate electro-optic Mach-Zender modulator. (a) Microscope image of a fabricated LiTaO3 modulator.
Inset scanning electron microphotography (SEM)images show the key components of the LiTaO3 modulator, including the cross-sectional
LiTaO3 waveguide (blue), the silver electrode (yellow), the ground-single-ground (GSG) configuration and the Y-spliter. (b) Phase matching
between the optical and microwave waves. The simulated group index ng = 2.25 of the optical LiTaO3 waveguide is marked in a dotted
line. The measured RF phase index is neff = 2.22 at 50 GHz. Inset: Cross-section of MZM covered with SiO2 cladding. Parameters are:
electrode thickness: 800 nm, cladding thickness: 1.4 µm, LiTaO3 thickness (half-etched): 600 nm, BOX layer: 4.7 µm, electrode gap: 6 µm,
waveguide width: 1.2 µm, signal electrode width: 19 µm. (c) Measured radiofrequency (RF) losses on a square-root frequency axis.
×17 lower compared to LiNbO3 (∆n = −0.07), which fa-
cilitates the design and development of compact devices with
tight bends. Moreover, LiTaO3 has a significantly higher op-
tical damage threshold [20] and weaker photorefractive ef-
fect [19], which helps mitigate DC bias drift problems that
are commonly observed in LiNbO3 modulators [8, 21, 22].
Given these advantages, LiTaO3 is expected to match or even
surpass LiNbO3 in performance for various photonic devices
— at reduced cost. However, to date, the core component
of optical communications, the ultra-high speed electro-optic
modulator, has not yet been demonstrated in LTOI.
Here, we report on a high-speed LiTaO3 Mach-Zehnder
modulator (MZM) with a measured electro-optic 3 dB band-
width of approximately 110 GHz and a half-wave voltage-
length product of 2.8 V·cm. We achieve these performance
parameters by engineering the microwave and photonic cir-
cuits and by applying high-conductivity electrodes to simul-
taneously achieve low microwave losses and group-velocity-
matching. We further conducted data communication exper-
iments using the LTOI modulator and demonstrated PAM8
transmission at a symbol rate of 176 GBd, achieving a single-
carrier net data rate of more than 400 Gbit/s with a bit-error ra-
tio (BER) below the threshold for soft-decision forward-error
correction (SD-FEC) with 25% coding overhead. Addition-
ally, we show that the LTOI device exhibits a lower bias drift
compared to an LiNbO3 integrated modulator.
Fig. 1(a) shows a fabricated LiTaO3 MZM composed of
two 50:50 adiabatic Y-splitters at either end and a push-
pull optical waveguide phase shifter pair with a length of
6 mm.
In our work we use an x-cut single crystalline
LiTaO3 thin film, which was fabricated from a optical-grade
bulk LiTaO3 wafer by using ion-cutting and wafer bonding
methods. The fabricated LTOI wafer stack consists of a 600
nm thin-film LiTaO3, a 4.7 µm thick thermal silicon diox-
ide, and a 525 µm thick high-resistivity silicon carrier wafer.
The device fabrication process chosen in this work was based
on die level processing and involved three main steps: 1)
electron beam patterning; 2) ion-beam dry-etching and KOH
wet-etching; 3) lift-off of the coplanar waveguide (CPW)
electrodes. The performance of a high-speed traveling-wave
electro-optic modulators based on ferroelectric thin films de-
pends on the impedance of the RF waveguide, the loss of the

<!-- page 3 -->
3
-4
-2
0
2
4
Voltage (V)
0
0.2
0.4
0.6
0.8
1
Transmission
0
10
20
30
40
50
60
Time (min)
Power shift (dB)
LTOI Air claded (Au)
LTOI SiO2 claded (Ag)
LNOI Air claded (Au) Ref.*
Quadrature
Vπ = 4.8 V
20
40
60
80
100
Frequency (GHz)
-40
-30
-20
-10
0
Electro-electro S11 (dB)
20
40
60
80
100
Frequency (GHz)
-10
-8
-6
-4
-2
0
2
Electro-optic S21 (dB)
- 3 dB 
- 6 dB 
- 18 dB 
(c)
(d)
(a)
(b)
-10
-8
-6
-4
-2
0
2
Simulated S21
Measured S21
FIG. 2. Electro-optic performance of the LiTaO3 Mach-Zender modulator. (a) Measured DC and extinction for a 6 mm long modulator.
(b) Dynamics of relative output intensity when bias set to quadrature at quadrature as a function of the operating time in the LiTaO3 MZM
and comparison with thin-film LiNbO3 modulators. The bias drift data of LiNbO3 in yellow curve is from Ref [25].(c) EO bandwidth (S21)
and (d) electrical reflection (S11) of the LiTaO3 modulator, showing a high 3-dB bandwidth at around 110 GHz. The simulated EO response
is calculated from the electro-electro measurement.
RF signal, and the matching of the propagation speeds of the
optical and RF signals. The relevant theory is well-developed
in the integrated modulator field [8, 17, 23, 24]. To make a
trade-off between decreasing the RF loss and the modulation
efficiency, the LiTaO3 waveguide width is chosen to 1.2 µm
and the gap between the waveguide sidewalls and the elec-
trode is 2.4 µm on each side. By designing the thickness of
each layers of the LTOI wafer, we can achieve good veloc-
ity matching between the RF and optical waves. Fig. 1(b)
presents the measured RF refractive index of a SiO2 cladded
600 mm long MZM devices, showing well-matched velocity
with neff = 2.22 at 50 GHz and ng = 2.25. The detailed cross-
section structure and the layer thicknesses are shown in the
inset of Fig. 1(b).
For the RF attenuation, instead of the commonly used
gold CPW in LiNbO3 modulator [9, 17, 25], silver is de-
posited with an 800 nm thickness as the CPW electrodes
in this work.
Although gold is widely employed due to
its low resistivity (ρAu = 2.2 × 10−6 Ωcm) and excel-
lent process stability, silver is showing an even lower resis-
tivity (ρAg = 1.55 × 10−6 Ωcm) [26].
To evaluate the
electrical losses of gold and silver electrodes, we fabricated
CPWs using the same device parameters with both silver and
gold. Fig. 1(c) shows the microwave loss measurement re-
sult of the silver and gold CPWs using a 67 GHz vector net-
work analyzer (VNA). The results indicate that silver CPWs
exhibit better linear loss due to their lower resistivity, with
αRF,Ag = 0.58 dB cm−1GHz−1/2 compared to gold, which
has αRF,Au = 0.77 dB cm−1GHz−1/2 (Figure 1(c)). The
square root dependence of the loss in Figure 2(b) also con-
firms the assumption that ohmic loss is the dominant source of
RF attenuation. Additionally, silver demonstrated good stabil-
ity over a one-month exposure test, showing no performance
degradation. This indicates that silver is a promising CPW
electrode material with low ohmic loss, good stability, and
greater economic efficiency.
The static Vπ characterization is performed first. Light at
1550 nm is coupled in and out using two lensed fibers with
a coupling loss of about 7 dB per facet. The majority of the
coupling loss results the large mismatch between the mode
sizes and mode indices of the optical fiber and the partially-
etched LiTaO3 rib waveguide. This occurs because tapering

<!-- page 4 -->
4
the rib waveguide pushes the optical mode into the slab re-
gion, which can be optimized by using a mode size converter
[27]. The low propagation loss performance of LiTaO3 has
been demonstrated in our previous work [19]. We fabricated
additional optical LiTaO3 waveguides without CPW to inves-
tigate the metal absorption loss in the current devices. The
measured results showed no measurable optical loss differ-
ence between the reference waveguides and the real device
waveguides, indicating that the absorption loss of the elec-
trodes in our current design length is negligible, consistent
with the simulation, which showed an electrode absorption
loss of 0.03 dB/cm. For a 6 mm MZM with 100 Hz triangular
voltage sweeps, the measured Vπ is 4.8 V at 1550 nm, corre-
sponding to VπL of 2.8 V · cm. The VπL increase relative to
the previous result [19], can be attributed to the enlarged gap
between the ground and signal electrode. By refining the de-
sign parameters, specifically through extending the length of
the modulator and narrowing the electrode gap, or by using a
substrate with lower RF absorption losses like quartz [28], it
is feasible to achieve high-bandwidth and low-voltage opera-
tions on LiTaO3.
Next, we study the bias drift of the modulator. Although
thin film LiNbO3 modulator has been well developed, the is-
sue of EO relaxation in LiNbO3 poses a considerable chal-
lenge for long-timescale applications.
The causes of EO
relaxation in LiNbO3 are investigated to be related to fac-
tors such as interface defects and the photorefractive effect
[21, 29]. LiTaO3 has been demonstrated a weaker photore-
fractive (PR) effect and lower drift phenomenon compared to
LiNbO3 in optical microresonators [19, 22]. In the fabricated
LiTaO3 MZM, we set a quadrature bias and measured the DC
drift over 60 minutes. We use two types of LiTaO3 MZMs
devices: one with air cladding and gold as the CPW material,
and another with SiO2 cladded and silver as the CPW mate-
rial. We observed that the DC drift behavior differed between
these two devices. The MZM with SiO2-cladding and silver
as electrode a drift of about 3 dB, while the device with air
cladding and gold as the CPW material had a drift of only 1
dB. The difference in drift between these two devices might be
due to the presence of more traps at the SiO2 cladding inter-
face, enhancing the PR effect [29]. The different CPW materi-
als might also contribute to the variation in DC drift behavior,
which requires further investigation. However, in both cases,
the DC drift in the LTOI is much smaller than that observed
8 dB drift in LiNbO3 [25], indicating better DC stability in
LiTaO3 MZM.
We then characterized the small-signal electro-optic (EO)
bandwidth of the fabricated devices. The measured EO re-
sponse of the LiTaO3 MZM in Fig. 2(c) shows a flat response
over the frequency range with a roll-off around 3 dB from 10
MHz to 110 GHz. Fig. 2(d) shows the measured electrical
reflection of the silver CPW below -18 dB at up to 110 GHz
revealing that the good impedance matching was achieved and
the reflection levels are sufficiently low for practical applica-
tions.
To demonstrate the outstanding performance of the de-
signed and fabricated thin-film LiTaO3 MZM, a high-speed
data communication experiment was conducted. The under-
lying setup is depicted in Fig. 3(a).
The optical carrier is
provided by an external-cavity laser (ECL), which, after a
fiber polarization controller (PC), is coupled into the MZM
through a lensed fiber. The electrical drive signal is gener-
ated by a high-speed arbitrary waveform generator (AWG,
M8199B, Keysight Technologies Inc., CA, USA) and fed to
the MZM via a 20 cm-long RF cable and a 110 GHz RF probe.
At the transmitter, we use digital signal processing (Tx-DSP)
to generate various pulse-amplitude modulation (PAM) sig-
nals based on pseudo-random bit sequences (PRBS) and root-
raised cosine (RRC) pulse-shaping filters with a roll-off of
ρ = 0.05.
We account for the frequency-dependent RF
loss in the cable by applying a linear minimum-mean-square-
error (MMSE) predistortion. A second 110 GHz RF probe
with an attached bias-T followed by a 50 Ohm resistor is
used to terminate the device and to set the MZM bias point
to quadrature point for intensity modulation. The intensity-
modulated optical signal generated by the MZM is coupled
out through a second lensed fiber and amplified by an erbium-
doped fiber amplifier (EDFA). Unwanted out-of-band ampli-
fied spontaneous-emission (ASE) noise of the EDFA is sup-
pressed by a tunable bandpass filter (BPF, Koshin Kogaku Co.
Ltd., Kanagawa, Japan). A tap separates 1% of the light af-
ter the BPF to monitor the optical spectrum of the signal by
an optical spectrum analyzer (OSA, Apex Technologies, FR).
The remaining 99% of the light is passed through a variable
optical attenuator (VOA, LTB-1, EXFO Inc., CA) to adjust
for an optical output power of 10 dBm, and the signal is then
sent to the subsequent high-speed 90 GHz photodiode (PD,
Finisar Corp., CA, USA) for direct detection. The received
electrical signal is digitized by a real-time oscilloscope (RTO,
UXR 1004A, Keysight Technologies Inc., CA, USA) with an
analogue bandwidth of 100 GHz and a sampling rate of 256
GSa/s. The data is finally extracted by offline receiver DSP
(Rx-DSP), which contains resampling to 2 Sa/sym, timing re-
covery, linear Sato equalization, and an additional decision-
directed least-mean-square (DD-LMS) equalizer.
We use the setup shown in Fig. 3 to generate and receive
PAM2, PAM4 and PAM8 data signals with symbol rates rang-
ing from 144 GBd to 208 GBd in an optical back-to-back
configuration. Inset in Fig. 3 (a) shows an exemplary optical
spectrum of 176 GBd PAM8 measured by the OSA. After the
Rx-DSP, the bit error ratios (BER) of the various PAM sig-
nals are calculated and plotted as a function of symbol rate,
see Fig. 3 (b).The limits for soft-decision forward-error cor-
rection (SD-FEC) with 25% and 15% overhead and for hard-
decision forward error correction (HD-FEC) with 7% over-
head are indicated by horizontal black dashed lines. The re-
sults indicate that we can transmit PAM8 signal with a sym-
bol rate of 176 GBd, while the measured BER of 3.8 × 10−2
is still below the threshold of 25% SD-FEC limit. For 200
GBd PAM4 and 208 GBd PAM2, the achieved BERs are well
below the threshold of 15% SD-FEC and 7% HD-FEC lim-
its, respectively. The eye diagrams and histograms of the cir-
cled data points are depicted in Fig. 3 (d). The histograms
show the display of the signal amplitudes sampled at the cen-
ter of the eye diagrams indicated by vertical dashed lines. We
then calculate achievable information rates (AIR) by multiply-

<!-- page 5 -->
5
10-4
10-3
10-2
Bit error ratio (BER)
140
160
180
200
Symbol rate (GBd)
100
200
300
400
500
AIR / NDR (Gbit/s)
140
160
180
200
Symbol rate (GBd)
176 GBd PAM8 (528 Gbit/s)
200 GBd PAM4 (400 Gbit/s)
FPC
ECDL 
EDFA
AWG
Tx-DSP
VOA
RTO
Rx-DSP
256 GSa/s
OSA
1549
1549.5
1550
1550.5
1551
(20 dB/div)
Optical power
RBW:100 MHz
176 GBd PAM8
(a)
(b)
(c)
(d)
1%
LiTaO3 MZM (110 GHz)
BPF
Wavelength (nm)
PD
256 GSa/s
PAM8
7% FEC
15% FEC
25% FEC
PAM4
PAM2
PAM8
PAM4
PAM2
Net: 405 Gbit/s
208 GBd PAM2 (208 Gbit/s)
(e)
(f)
10-1
<10-7
(f)
(e)
(d)
Time (2 ps/div)
Time (2 ps/div)
Time (2 ps/div)
Counts (arb.u.)
Counts (arb.u.)
Counts (arb.u.)
Amplitude (arb.u.)
Amplitude (arb.u.)
Amplitude (arb.u.)
FIG. 3. Schematic and results of the intensity-modulation direct detection (IMDD) experiment employing the thin-film LiTaO3 Mach-
Zender modulator. (a) Experimental setup: an external cavity laser (ECL) serves as a light source, and a polarization controller (PC) is used
to tune the polarization of the light. The optical input/output coupling to the 110 GHz LiTaO3 MZM relies on a pair of lensed fibers. The
drive signals are synthesized by transmitter digital signal processing (Tx-DSP) and generated by an arbitrary waveform generator (AWG). The
modulated light is amplified by an erbium-doped fiber amplifier (EDFA), and the out-of-band amplified spontaneous emission (ASE) noise is
suppressed by a tunable bandpass filter (BPF). 99% of the amplified light enters a variable optical attenuator (VOA) before being detected by
a photodiode (PD). A high-speed real-time oscilloscope (RTO) samples the resulting signal, which is processed offline by receiver DSP (Rx-
DSP). 1% of the amplified light after the EDFA is sent to an optical spectrum analyzer (OSA) for monitoring. The inset 1 shows an exemplary
optical spectrum for a 176 GBd PAM8 signal. (b) Measured bit error ratios (BER) as a function of symbol rates for PAM8 (red), PAM4
(green), and PAM2 (blue) signals. Horizontal black dashed lines indicate the thresholds for 25%, 15% soft-decision and for 7% hard-decision
forward error correction (FEC). (c) Extracted available information rates (AIR, dashed lines) and associated net data rates (NDR, solid lines)
for measurements with BER values below 25% SD-FEC limit. The yellow star marks the highest NDR of 405 Gbit/s achieved by using a
PAM8 signal at a symbol rate of 176 GBd. (d)-(f) Eye diagrams and associated histograms taken in the center of the symbol slot (vertical
dashed line) for the circled data points marked in Subfigure (b).

<!-- page 6 -->
6
ing the symbol rates with the normalized mutual information
(NGMI), where the NGMI values are calculated for each mea-
surement based on log-likelihood ratios (LLR) by using an
additive white Gaussian noise (AWGN) channel model. The
corresponding results are displayed by curves in dashed lines
in Fig. 3 (b). In the same graph, we further plot the associ-
ated net data rates (NDR) for measurements with BER values
below the 25% SD-FEC limit in solid lines. The NDRs are
obtained by using the respective code rate associated with the
NGMI threshold, as measured in the publication [30], multi-
plied with the symbol rate. As a result, the highest AIR of 432
Gbit/s is achieved by using PAM8 signal at a symbol rate of
176 GBd. Our first proof-of-concept LTOI MZM can hence
achieve a single-carrier net data rate of 405 Gbit/s, which is
among the highest values so far achieved for MZM [10, 11],
and which underlines the outstanding potential of the technol-
ogy. Note that in this IMDD demonstration, only the most
computing-efficient linear equalizations are employed, which
leaves great room for further improvement in terms of net data
rate by using more advanced probabilistic constellation shap-
ing (PCS) [31] or fast-than-Nyquist (FTN) coding [32] and by
employing non-linear equalization techniques [33].
In conclusion, we have demonstrated the first high-speed
LTOI based modulator, offering an electro-optic 3 dB band-
width of 110 GHz. We prove the viability of the device by
using it in IMDD transmission experiment, reaching a single-
carrier net data rate of 405 Gbit/s, which is already on par with
best-in-class LNOI and plasmonic platform devices [10, 11].
We further show that using silver as an electrode material al-
lows to reduce the microwave losses, and we find that LTOI
devices offer a series of technical advantages with respect to
their LNOI counterparts, such as increased DC bias stability.
Using longer devices can further reduce the half-wave volt-
age to well below 1 V. By combining the excellent signal
fidelity with advanced in-phase/quadrature (I/Q) modulator
design or more advanced signaling techniques and by incor-
porating polarization-division multiplexing, LiTaO3 devices
could offer line rates of 2 Tbit/s or more. With the mass pro-
duction and availability of LTOI wafers, our results position
LTOI as a highly promising integration platform for future
electro-optic modulators that might outperform LNOI devices
and that are key to future high-speed optical communication
networks with increased throughput and efficiency.
ACKNOWLEDGMENTS
The samples were fabricated in the EPFL Center of Micro-
NanoTechnology (CMi) and the Institute of Physics (IPHYS)
cleanroom. T.J.K. acknowledges support from the Swiss Na-
tional Science Foundation under grant agreement No. 216493
(HEROIC). This work was further supported by the BMBF
project Open6GHub (no.
16KISK010), by the ERC Con-
solidator Grant TeraSHAPE (773248), by the DFG projects
PACE (403188360) and GOSPEL (403187440) within the Pri-
ority Programme Electronic-Photonic Integrated Systems for
Ultrafast Signal Processing (SPP 2111), and by the DFG Col-
laborative Research Centers (CRC) HyPERION (SFB 1527).
AUTHOR CONTRIBUTIONS
C.W. and X.O. fabricated the wafers. C.W. and J.Z. de-
signed the devices. C.W. and J.Z. fabricated the devices. D.F.
and C.W. carried out the measurements. D.F., C.W., and J.Z.
analyzed the data. C.W. and D.F. prepared the figures and
wrote the manuscripts with contributions from all authors.
X.O.,C.K. and T.J.K. supervised the project.
COMPETING INTERESTS
The authors declare no competing financial interests.
DATA AVAILABILITY STATEMENT
The code and data used to produce the plots within this
work will be released on the repository Zenodo upon publi-
cation of this preprint.
[1] Tauber, D. et al. Role of coherent systems in the next dci gener-
ation. Journal of Lightwave Technology 41, 1139–1151 (2023).
[2] Winzer, P. J., Neilson, D. T. & Chraplyvy, A. R. Fiber-optic
transmission and networking: the previous 20 and the next 20
years. Optics express 26, 24190–24239 (2018).
[3] Xu, Q., Schmidt, B., Pradhan, S. & Lipson, M. Micrometre-
scale silicon electro-optic modulator.
nature 435, 325–327
(2005).
[4] Sun, C. et al. Single-chip microprocessor that communicates
directly using light. Nature 528, 534–538 (2015).
[5] Thomson, D. et al. Roadmap on silicon photonics. Journal of
Optics 18, 073003 (2016).
[6] Zhang, X., Shi, G., Leveillee, J. A., Giustino, F. & Kioupakis,
E. Ab initio theory of free-carrier absorption in semiconductors.
Physical Review B 106, 205203 (2022).
[7] Ogiso, Y. et al.
80-ghz bandwidth and 1.5-vv π inp-based
iq modulator. Journal of Lightwave Technology 38, 249–255
(2019).
[8] Zhang, M., Wang, C., Kharel, P., Zhu, D. & Lonˇcar, M. Inte-
grated lithium niobate electro-optic modulators: when perfor-
mance meets scalability. Optica 8, 652–667 (2021).
[9] He, M. et al. High-performance hybrid silicon and lithium nio-
bate mach–zehnder modulators for 100 gbit s- 1 and beyond.
Nature Photonics 13, 359–364 (2019).
[10] Berikaa, E. et al. Tfln mzms and next-gen dacs: Enabling be-
yond 400 gbps imdd o-band and c-band transmission. IEEE
Photonics Technology Letters (2023).
[11] Kulmer, L. et al. Single carrier net 400 gbit/s im/dd over 400
m fiber enabled by plasmonic mach-zehnder modulator. In Op-
tical Fiber Communication Conference, W4H–5 (Optica Pub-
lishing Group, 2024).

<!-- page 7 -->
7
[12] Kieninger, C. et al.
Silicon-organic hybrid (SOH) Mach-
Zehnder modulators for 100 GBd PAM4 signaling with sub
-1 dB phase-shifter loss.
Optics Express 28, 24693–24707
(2020).
[13] Freude, W. et al.
High-Performance Modulators Employing
Organic Electro-Optic Materials on the Silicon Platform. IEEE
Journal of Selected Topics in Quantum Electronics 30, 1–22
(2024).
[14] Abel, S. et al. Large pockels effect in micro-and nanostructured
barium titanate integrated on silicon. Nature materials 18, 42–
47 (2019).
[15] Phare, C. T., Daniel Lee, Y.-H., Cardenas, J. & Lipson, M.
Graphene electro-optic modulator with 30 ghz bandwidth. Na-
ture photonics 9, 511–514 (2015).
[16] Han, J.-H. et al. Efficient low-loss ingaasp/si hybrid mos optical
modulator. Nature Photonics 11, 486–490 (2017).
[17] Wang, C. et al. Integrated lithium niobate electro-optic modu-
lators operating at cmos-compatible voltages. Nature 562, 101–
104 (2018).
[18] Ballandras, S. et al. New generation of saw devices on advanced
engineered substrates combining piezoelectric single crystals
and silicon.
In 2019 Joint Conference of the IEEE Interna-
tional Frequency Control Symposium and European Frequency
and Time Forum (EFTF/IFC), 1–6 (IEEE, 2019).
[19] Wang, C. et al. Lithium tantalate photonic integrated circuits
for volume manufacturing. Nature 1–7 (2024).
[20] Yan, X. et al. High optical damage threshold on-chip lithium
tantalate microdisk resonator.
Optics Letters 45, 4100–4103
(2020).
[21] Holzgrafe, J. et al. Relaxation of the electro-optic response in
thin-film lithium niobate modulators. Optics Express 32, 3619–
3631 (2024).
[22] Yu, J. et al. Tunable and stable micro-ring resonator based on
thin-film lithium tantalate. APL Photonics 9 (2024).
[23] Alferness,
R.,
Korotky,
S. & Marcatili,
E.
Velocity-
matching techniques for integrated optic traveling wave
switch/modulators. IEEE journal of quantum electronics 20,
301–309 (1984).
[24] Zhu, D. et al. Integrated photonics on thin-film lithium niobate.
Advances in Optics and Photonics 13, 242–352 (2021).
[25] Xu, M. et al. High-performance coherent optical modulators
based on thin-film lithium niobate platform. Nature communi-
cations 11, 3911 (2020).
[26] Matula, R. A. Electrical resistivity of copper, gold, palladium,
and silver. Journal of Physical and Chemical Reference Data
8, 1147–1298 (1979).
[27] Ying, P. et al. Low-loss edge-coupling thin-film lithium nio-
bate modulator with an efficient phase shifter. Optics letters 46,
1478–1481 (2021).
[28] Xu, M. et al. Dual-polarization thin-film lithium niobate in-
phase quadrature modulators for terabit-per-second transmis-
sion. Optica 9, 61–62 (2022).
[29] Xu, Y. et al.
Mitigating photorefractive effect in thin-film
lithium niobate microring resonators. Optics Express 29, 5497–
5504 (2021).
[30] Hu, Q. et al. Ultrahigh-net-bitrate 363 gbit/s pam-8 and 279
gbit/s polybinary optical transmission using plasmonic mach-
zehnder modulator. Journal of Lightwave Technology 40, 3338–
3346 (2022).
[31] Yamazaki, H. et al. Net-400-gbps ps-pam transmission using
integrated amux-mzm. Optics express 27, 25544–25550 (2019).
[32] Che, D. & Chen, X. Higher-order modulation vs faster-than-
nyquist pam-4 for datacenter im-dd optics: an air comparison
under practical bandwidth limits. Journal of Lightwave Tech-
nology 40, 3347–3357 (2022).
[33] Liu, L. et al. Intrachannel nonlinearity compensation by inverse
volterra series transfer function. Journal of Lightwave Technol-
ogy 30, 310–316 (2011).

