---
paper_id: rahman2025
source_url: https://arxiv.org/abs/2504.00311
doi: 
license: CC-BY-NC-ND-4.0
sha256: b9713ce9a115b30fd06ef20bfdd0425c481ecdffa19bc85ea637034baee25cb7
pages: 14
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
High-performance Hybrid Lithium Niobate Electro-optic Modulators Integrated with
Low-loss Silicon Nitride Waveguides on a Wafer-scale Silicon Photonics Platform
Arif Rahman,1 Forrest Valdez,2 Viphretuo Mere,2 Camiel Op
de Beeck,2, ∗Pieter Wuytens,2 and Shayan Mookherjea1, †
1University of California, San Diego, Department of Electrical
and Computer Engineering, La Jolla, California 92093-0407, USA
2LIGENTEC SA, EPFL Innovation Park, Batiment L,
Ch.
de la Dent d’Oche 1B, 1024 Ecublens VD, Switzerland
(Dated: April 3, 2025)
Heterogeneously-integrated electro-optic modulators (EOM) are demonstrated using the hybrid-
mode concept, incorporating thin-film lithium niobate (LN) by bonding with silicon nitride (SiN)
passive photonics. At wavelengths near 1550 nm, these EOMs demonstrated greater than 30 dB ex-
tinction ratio, 3.8 dB on-chip insertion loss, a low-frequency half-wave voltage-length product (VπL)
of 3.8 V.cm, and a 3-dB EO modulation bandwidth exceeding 110 GHz. This work demonstrates
the combination of multi-layer low-loss SiN waveguides with high-performance LN EOMs made in
a scalable fabrication process using conventional low-resistivity silicon (Si) wafers.
I.
INTRODUCTION
The
integration
of
high-performance
electro-optic
modulator (EOM) devices with a low-loss photonic in-
tegrated circuit (PIC) platform can address important
challenges in modern photonics, such as developing high-
throughput interconnects and switches, signal process-
ing applications including wideband waveform genera-
tion and low-noise signal acquisition, and advanced com-
puting.
Traditionally, the most widely-used EOM de-
vices are made using lithium niobate (LN) which sup-
ports a wide range of optical wavelengths and radio fre-
quencies (RF) [1, 2]. However, traditional LN EOM de-
vices require specialized fabrication methods and materi-
als which are different from those used in the global sili-
con (Si) microelectronics industry. On the other hand, Si
photonic EOM devices, which are made in Si foundries,
have shown limited performance with lower modulation
amplitude at higher speeds, and limited linearity and op-
tical power handling due to free-carrier absorption and
two-photon absorption [3–6].
Recent interest in thin-
film lithium niobate (TFLN) or lithium niobate on in-
sulator (LNOI) photonics [7] has led to very high band-
width modulators using a variety of designs and fabrica-
tion approaches [8–11]. An important challenge now is to
make high-performance LN EOM devices using industry-
standard Si wafer fabrication methods, which could lead
to greater scalability and lower cost, and the ability to
make more complex designs with higher yield.
One approach to developing such modulators is to use
the hybrid-mode design, in which the light is distributed
among two (or more) waveguide core materials, one of
which is a thin-film slab of LN, and the other can be a
dielectric or semiconductor (e.g., silicon nitride, SiN, or
Si) which is patterned into strip waveguides [9, 12–14].
∗cob@ligentec.com
† smookherjea@ucsd.edu
This design can be contrasted to one in which the LN
slab was loaded by a dielectric material to form a rib
waveguide [15–17]. In the hybrid-mode devices such as
those discussed here, the waveguide core materials were
separated by a thin cladding layer, such as silicon diox-
ide, and the strip waveguides in the dielectric material
were used for passive photonic components without LN.
The thin LN layer was transferred by bonding to a pla-
narized wafer containing the strip waveguides [18, 19].
After bonding, the LN slab can be left unetched because
the mode confinement in the EO region can be controlled
over a wide range by changing only the width of the strip
waveguide [13]. Wide waveguides, which confine most of
the light in the passive material, are used in the feeder
section and for waveguide bends, splitters and combin-
ers.
Narrow waveguides, which push light mostly into
the LN region, are used in the phase-shift section of the
EOM device [9, 13]. As design and fabrication technol-
ogy have improved, the mode confinement factor in LN
in the modulation section of hybrid LN EOM devices has
increased from 11% [12] to over 80% [9].
The hybrid-mode design does not require precise align-
ment during film transfer, in contrast to micro-transfer
printing [20, 21], and does not require a smooth and
damage-free LN waveguide etching or milling process, in
contrast to EOM devices developed on an LNOI plat-
form [8, 11, 22–24]. Furthermore, heterogeneous integra-
tion of thin-film EO crystals is a powerful tool for incor-
porating high-performance EO microsystems by building
upon existing Si and SiN photonics technology without
having to start from scratch, since many components of a
mature process design kit (PDK) can be reused in the de-
sign of large-scale PICs [25]. Because of these attractive
characteristics, here we use the hybrid-mode concept to
integrate high-performance EOM devices with a low-loss
SiN photonics platform.
arXiv:2504.00311v2  [physics.optics]  2 Apr 2025

<!-- page 2 -->
2
(c)
Edge 
coupler
2x2 
coupler
Co-Planar
SiN waveguides
TFLN
Al
(b)
(e)
(f)
(d)
TFLN 
edge
Air trench into Si substrate
~80 µm
8 µm
SiN1
SiN2
x-TFLN
SiO2
Al
Si
100 nm
300 nm
~750 µm
G
G
S
(not to scale)
(a)
Slow-Wave
traveling-wave RF 
electrodes
1 mm
neff = 1.755; Aeff = 0.92 μm2  
ΓLN = 0.0
(e)
(f)
(d)
neff = 1.819; Aeff = 1.43 μm2  
ΓLN = 0.53
neff = 1.830; Aeff = 1.78 μm2  
ΓLN = 0.08
Feeder Mode
Hybrid Mode
Transition Mode
SiN1
Air
SiO2 
1 µm
SiN1
TFLN
SiO2 
Air
SiN2
SiO2 
1 µm
1 µm
TFLN
SiN2
Figure 1. (a) Photograph of the fabricated wafer (100 mm diameter). (b) Magnified photograph of the electro-optic modulator
(EOM) devices discussed in this report. (c) Schematic drawing of the wafer cross-section. Abbreviations: S and G: signal
and ground electrodes, respectively; TFLN: thin-film lithium niobate; SiN: silicon nitride; Si: silicon; Al: aluminum. (d)-(f)
Simulations of the optical mode in the feeder, transition, and hybrid sections, which are indicated in panel (b). The optical
effective index (neff), effective area (Aeff) and mode confinement fraction in LN (ΓLN) are tabulated.
II.
DESCRIPTION
A photograph of the completed wafer is shown in
Fig. 1(a) and the three EOM devices discussed in this re-
port are shown in Fig. 1(b). This image also shows some
passive structures (without electrodes) that were in-
cluded for measurement calibration. These EOM devices
are push-pull Mach-Zehnder interferometers (MZI) in
which both the optical and microwave fields co-propagate
from left to right. A portion of the bonded LN film (x-
cut orientation, 300 nm thickness) which covers the SiN
waveguides is visible in the figure with a slightly differ-
ent color. Figure 1(c) shows a schematic drawing of the
wafer cross-section with two silicon nitride layers (SiN1
and SiN2) with strip waveguides. The SiN1 layer con-
tains the feeder waveguides and 2 × 2 multi-mode inter-
ference (MMI) couplers (length about 100 µm) and an
optical path length difference segment in the passive re-
gion outside the LN (thus creating an asymmetric MZI
structure). The feeder waveguides continue a short dis-
tance into the LN-covered region.
An adiabatic taper
was formed between the SiN1-SiN2 layers by decreasing
the width of the SiN1 waveguide in the lower layer while
simultaneously increasing the width of the SiN2 waveg-
uide in the upper layer. The SiN2 waveguide tapers from
the transition mode to the hybrid mode formed between
the SiN2 strip waveguide and the LN slab, which is used
in the EO phase-shift segments.
The optical and microwave structures reported here
were designed using software (Ansys Lumerical; and
PathWave EM Design, Keysight Technologies) guided by
insights from earlier work [26–29]. As shown in Fig. 1(d)-
(f), the SiN1 waveguide has the smallest optical effective
area (Aeff = 0.92 µm2), whereas a representative mode in
the transition section and the hybrid mode have a larger
effective area of 1.78 µm2 and 1.43 µm2, respectively.
The confinement factor Γ stated in Figs. 1(d)-(f) is de-
fined by the spatial distribution of the magnitude of the
Poynting vector, S = E × H∗where E and H are the
vectorial electric and magnetic field distributions of the
mode. The hybrid mode has ΓLN ≈53% of the optical
power in LN.
The three EOM designs shown in Fig. 1(b) from top
to bottom are labeled Designs 1, 2, and 3, respectively.
Design 1 uses coplanar waveguide (CPW) electrodes with
a signal-ground gap distance of G = 6 µm. According to
our simulations, Design 1 can achieve a good match be-
tween the RF phase and optical group indices of refrac-
tion up to high RF modulation bandwidths. However,
the SiN waveguides were fabricated on a low-resistivity
Si substrate, for which the high-frequency RF propaga-
tion loss is too large to support high modulation band-
widths (as will be shown by the data presented in this
report). Therefore, in Designs 2 and 3, portions of the
substrate were removed by etching air trenches as de-
scribed below (also see the Supplementary Information,

<!-- page 3 -->
3
Section S1) whereas no trenches were included with De-
sign 1, in order to provide a comparison. Removing the
low-resistivity Si substrate reduces the propagation loss
of the RF coplanar waveguide mode and increases the
modulation bandwidth [23, 30–32]. These trenches also
have the effect of lowering the RF index. To re-establish
index matching in Designs 2 and 3, we used periodically-
patterned slow-wave electrodes (SWE) [33]. Specifically,
Designs 2 and 3 used G = 4 µm and 5 µm, respectively,
with a capacitively-loaded T-rail electrode design [28, 34].
As the distance between the inner electrode edge and the
inner T-rail edge increases, the RF wave becomes slower
which results in a larger RF effective index. Furthermore,
the T-rail stem width also affects the RF wave velocity,
with narrower width corresponding to a slower RF field.
The measurements reported below show that both De-
signs 2 and 3 can support high modulation bandwidths
beyond 100 GHz.
III.
FABRICATION PROCESS
Figure 2 describes the fabrication process, which starts
with a 200 mm diameter silicon (Si) handle wafer, with
a thickness of about 700 µm and resistivity of about
10 Ω.cm.
The two SiN layers were fabricated using
LIGENTEC’s proprietary technology with high-quality
silicon nitride deposited using a low-pressure chemical-
vapor deposition (LPCVD) process.
The strip waveg-
uides in the SiN1 layer have dimensions of 1 µm×0.8 µm
(width × thickness), and those in the SiN2 layer have di-
mensions of 1.4 µm × 0.35 µm. The buffer oxide below
SiN1 has a nominal thickness of 4 µm, and the top ox-
ide layer above SiN2 has a nominal thickness of 0.1 µm.
A planar bondable surface was achieved using chemical-
mechanical polishing (CMP). After processing, the wafer
was cored down to a diameter of 100 mm to match the
lithium-niobate-on-insulator (LNOI) wafers used here.
Lithium-Niobate-On-Insulator
(LNOI)
wafers
(100
mm
diameter)
with
300
nm
thickness
x-cut
LN on 2 µm thick buried oxide, and an approximately
500 µm thick Si handle were commercially procured
(NanoLN, Jinan Jingzheng Electronics Co., Ltd.). After
cleaning and surface activation [Fig. 2(b)], the Si and
LNOI wafers were bonded using a hydrophilic process
[Fig. 2(c)]. The initial contact bonding was mediated by
weak van der Waals forces between the exposed surfaces.
After room temperature bonding, the bonded sample
was annealed using temperature cycles up to 300 ◦C
under applied pressure. The annealing process converts
the hydrogen bonds into covalent bonds at the interface,
thereby improving the bond strength [35]. No adhesives
or polymers were used as the bonding layer. The handle
and oxide layer of the LNOI wafer were removed using
a combination of dry and wet etching steps [Fig. 2(d)].
The Aluminum (Al) electrodes were formed on top of the
LN layer using deposition, lithography, and a dry-etch
process [Fig. 2(e)].
After bonding, LN etching was performed across the
wafer to remove the LN over the passive sections where
it is not needed, such as over the SiN1 components. The
SiN1 waveguides continue outside of the bonded region,
as shown in Fig. 1(b), and can be used to connect seam-
lessly to other components on the same chip. Vertical
trenches were formed from the LN surface through the
oxide into the Si substrate as access holes for a dry-
etch process which formed an undercut. This is shown
schematically in Fig. 2(f), and the Supplementary Infor-
mation, Section S1, shows scanning electron microscope
(SEM) images of the cross-section. The trenches have lat-
eral dimensions of about 30 µm×8 µm (length × width)
and were about 5 µm in height. They were placed about
8 µm away from the SiN2 waveguide edge and do not
affect the optical mode. These trenches were used to ac-
cess the Si substrate for an isotropic dry etch process
which undercut the phase-shift segments in the Designs
2 and 3 EOM devices. Visual inspection showed a lateral
undercut width of about 80 µm.
IV.
RESULTS
Individual dies were diced from the wafer and were
tested using an optoelectronic probe station at room
temperature. Continuous-wave light was coupled to the
waveguides using a tunable semiconductor laser source
(81640B; Keysight Technologies) and lensed tapered
fibers with a spot size of about 3.5 µm (Oz Optics). The
on-chip optical power was about -3 dBm (0.5 mW). An
optical transmission scan over wavelengths from 1520 nm
to 1620 nm is reported in the Supplementary Informa-
tion, Section S2, and shows a high on-off contrast of
about 30 dB over more than 100 nm of operational band-
width.
A.
Optical insertion loss
The on-chip insertion loss (IL) of the EOM was mea-
sured by comparing the optical transmission through the
EOM to that through a passive waveguide formed in
the SiN1 layer. Both devices use similar waveguide-fiber
edge couplers, but the passive waveguide does not in-
clude transitions to and from the SiN2 layer, or the sub-
components of the EOM, such as the 3-dB input and
output couplers and the phase-shift segments. Since the
EOM is based on a MZI which includes a path-length
difference segment, the values at the peak of the spectra
near 1550 nm were taken as representative of its trans-
mission. The measured average on-chip IL (and one stan-
dard deviation) for EOM devices across four chips in the
three different designs were: 3.1 dB (1.4 dB) for Design
1, 6.6 dB (2.7 dB) for Design 2, and 4.0 dB (0.8 dB). De-
sign 2 has the smallest signal-ground electrode gap (G)
and therefore, has the highest IL. The observed variation
in IL from one chip to another is a consequence of the

<!-- page 4 -->
4
Si Substrate
SiO2
100 nm
SiN
350 nm
200 nm
2000 nm
300 nm
200 mm SiNX
100 
mm 
LNOI
X-cut LN
SiO2
Si Substrate
Bonding and annealing  
Plasma 
surface 
activation
LNOI handle removal
Metal deposition and Lift-off 
Al
LN
SiN
SiO2
(a)
(b)
(c)
(d)
(e)
(f)
800 nm
4000 nm
Air trench into substrate
~80 µm
8 µm
300 nm
1000 nm
Figure 2. Process flow diagram showing the main steps, (a) - (f), in the fabrication of the hybrid TFLN-SiN modulators.
(Schematic diagrams, not to scale.)
varying layer thicknesses and fidelity-to-design of devices
fabricated on the edges of the wafer compared to those in
the center. As this was the first fabrication run for these
hybrid modulators, we believe that uniformity issues can
be iteratively improved in future fabrication runs. At this
time, we do not have sub-component level IL data (i.e.,
for the inter-layer transitions alone, or the MMI coupler
alone) because of the limited space that was available to
us on this fabrication run.
Waveguide-coupled microring resonators with a waveg-
uide width of 1.7 µm were formed in the SiN1 layer
with a free spectral range of 150 GHz and measured to
have a loaded quality factor at critical coupling of about
1.2 million, from which we infer a propagation loss of
about 0.2 dB.cm−1 at 1550 nm across the whole wafer.
No significant change in loss was observed by comparing
the transmission of devices before and after LN bonding
and processing. By measuring light propagation through
waveguide spirals formed in the SiN2 layer (lengths of
10.4 cm, 15 cm, and 27 cm), we infer a propagation loss
ranging between 0.3–1.0 dB.cm−1 at 1560 nm, depending
on the location of the waveguide on the wafer. The large
range and higher loss in the SiN2 layer can be attributed
to damage of this layer during the etching of the LN
layer, and material absorption in the top oxide layer. A
more selective LN removal process to avoid such damage
is under development, which should result in more uni-
form propagation losses across the wafer, near the lower
limit of the data reported here. Taken together, these ob-
servations suggest that the passive SiN1 and SiN2 layers
have low optical loss and can be combined with high-
bandwidth EOM devices on the same platform.
B.
Half-wave voltage-length product (VπL)
Figure 3 describes the half-wave voltage (Vπ) measure-
ment method and shows the measured data for a repre-
sentative modulator using Design 3.
As shown in the
schematic drawing, Fig. 3(a), a trapezoidal signal gener-
ated by a pulse pattern generator (81110A, Agilent Tech-
nologies) was used to drive the modulator with a certain
peak-to-peak voltage that exceeds the Vπ value antici-
pated from design simulations.
The driving waveform
is shown by the blue line in Fig. 3(b). The modulated
optical signal was detected with a linear photodetector
(Model 1817-FC, Newport) and the electrical signal was
recorded on an oscilloscope, as shown by the red line
in Fig. 3(b).
Because the applied voltage was greater
than Vπ, the over-drive can easily be seen in the oscil-
loscope trace. This waveform was post-processed using
software (MATLAB) to map the optical transmission to
the applied voltage, and a cosine-squared fit was used to
find Vπ. Trapezoidal waveforms were used because they
perform a more stringent test over triangular ramps, as
discussed in our earlier work [18, 28]. Figure 3(c) shows
the modulated signals as a function of applied voltage
for frequencies of 1 kHz (red) and 1 MHz (blue), with-

<!-- page 5 -->
5
(a)
50 Ω Load
Photodiode
Oscilloscope
Pulse Pattern 
Generator
Laser
(b)
(d)
ER
(c)
Figure 3. (a) Schematic for the measurement of the half-wave voltage, Vπ. (b) Example of the (overdriven) optical response
(red line) to the trapezoidal electrical drive signal (blue line). The vertical axis for the optical response is on an arbitrary linear
scale determined by the photodetector. (c) Example of the normalized optical power measured as a function of the applied
voltage at frequencies of 1 kHz (red) and 1 MHz (blue). The solid line is a fit to the 1 MHz data using the squared-cosine
functional form, which yields Vπ as shown. (d) The same data as in panel (c) for 1 kHz with a logarithmic vertical scale to
quantify the extinction ratio, ER > 34 dB.
out a significant difference between the two cases, or for
other intermediate frequencies. The data was fit using a
cosine-squared equation, resulting in a measured Vπ of
6.4 V (VπL = 3.8 V.cm) in each case. Plotting the 1
kHz data using a logarithmic scale for the vertical axis
as shown in Fig. 3(d) demonstrates an extinction ratio
greater than 34 dB. The average (and one-standard de-
viation) VπL values for the three designs measured across
four chips were: 4.29 V.cm (0.14 V.cm) for Design 1
[signal-electrode gap G = 6 µm], 3.40 V.cm (0.15 V.cm)
for Design 2 [G = 4 µm], and 3.83 V.cm (0.01 V.cm) for
Design 3 [G = 5 µm]. As expected, the design with the
smallest G has the lowest VπL value.
C.
3-dB EO modulation bandwidth
As shown schematically in Fig. 4(a), we used a mod-
ified modulation-sideband method for measuring the
electro-optic response (EOR), and our previous work has
discussed the similarity of this measurement method to
others e.g., based on using a lightwave component an-
alyzer [27–29, 36].
Sinusoidal RF waveforms between
1 GHz and 50 GHz were generated using an RF sweeper
(83651B, Hewlett Packard, Inc.). A high-resolution OSA
(WaveAnalyzer 1500S, Finisar) was used to measure the
spectrum of the modulated signals. High-speed ground-
signal-ground RF probes rated to 110 GHz (Model i110,
FormFactor) sourced and terminated the traveling-wave
electrodes and were used with standard low-loss 1 mm
RF cables and connectors. The EOR was determined by
tracking the peaks of the carrier signal and the gener-
ated sidebands from modulation [37–40].
Active mul-
tipliers were used for the 47 GHz to 78 GHz band
(SFA-503753420-15SF-E1, Eravant) and for the 72 GHz
to 110 GHz band (SFA-753114616-10SF-E1, Eravant).
The amount of RF power delivered to the chip was
recorded using a calibrated thermal RF power meter
(NRP110T.02, Rohde & Schwarz). The two frequency
ranges over which the banded measurements overlap (47–
50 GHz and 72–78 GHz) were used to stitch the EOR re-
sponses into the composite EOR response. This was ac-
complished by calculating a mean value for both traces
in the overlapping frequency range, and vertically off-

<!-- page 6 -->
6
setting the second data set to the first curve. Measure-
ments made in the overlapping RF frequency ranges show
smooth continuity to within a fraction of a dB.
Figure 4(b) compares the EOR curves for the three
EOM designs on one chip. (The Supplementary Infor-
mation, Section S3, contains similar plots on other test
chips, which show similar behavior.) To clearly show the
faster drop-off at modulation frequencies above 5 GHz
in Design 1 compared to Designs 2 and 3, these EOR
curves were each normalized to their value at 3 GHz. (In
the literature, the EOR curve for short TFLN modula-
tors measured to high frequencies is usually normalized
to its value somewhere in the 1–5 GHz range, because the
transmission-line model for the EOR response is usually
less valid at low frequencies when the device length is
comparable with the RF wavelength.)
The main rea-
son for this drop-off in the performance of Design 1 at
higher modulation frequencies is the lack of the undercut
(trench) of the low-resistivity Si substrate, resulting in
higher RF loss and lower EO bandwidth.
Figure 4(c) shows the EOR for the high-bandwidth
Design 3 EOM between 1 GHz and 111 GHz, which is
the upper limit of our measurement apparatus. In this
plot, the EOR is normalized to its value at 1 GHz, which
is taken as 0 dBe, similar to our previous work [28, 29].
The solid line is a fit to the data using a transmission-line
model[41] (see Supplementary Information). The EOR
of TFLN EOM devices typically show a drop between
1 GHz and 5 GHz before a gradual roll-off is observed
over most of the measurement band; this behavior was
also shown on different test chips obtained using a light-
wave component analyzer (LCA) [28]. To examine the
EOR roll-off, two horizontal dashed lines were drawn in
Fig. 4(c) as visual guides.
The upper line was drawn
at the average EOR value between 1 GHz and 3 GHz,
and the lower line was drawn 3 dB below it. The sig-
nificance of the 3-dB frequency f0 is that the EOR rolls
off as [1 + (f/f0)2]−1 as f increases above f0. The EOR
response does not roll off below 3 dB up to the mea-
surement limit, and we can therefore take the 3-dB roll-
off frequency for these devices as greater than 111 GHz.
The Supplementary Information contains similar data on
other test chips, showing a 3-dB modulation bandwidth
greater than 100 GHz for all tested Design 2 and Design
3 EOMs.
When considering the overall system requirements to
drive an EOM, it is helpful to know Vπ as a function of
the modulation frequencies, because the impact of RF
loss, velocity mismatch, and impedance mismatch will
also effect the needed voltage for a π phase-shift[2]. The
driving voltage as a function of modulation frequency f
is given by
Vπ(f) = Vπ(DC)10−m(f)/20
(1)
where Vπ(DC) is the low-frequency half-wave voltage and
m(f) is the measured normalized EOR. For a numerical
estimate of Vπ(DC), we use the half-wave voltage at 1
MHz, measured as described in Section IV B. Figure 4(d)
shows the calculated RF Vπ(f) normalized to Vπ(DC).
An increase of this ratio by a factor of
√
2 is an alter-
native definition of the modulation bandwidth because
it correlates to the EOR of the modulator (and equiv-
alently, the EO S21 value [28, 42]) decreasing by 3 dB
(50%). Based on Fig. 4(d), we can also take as the 3-dB
modulation bandwidth for this EOM as 111 GHz (the
upper limit of our measurement).
V.
DISCUSSION
Design 1, which uses the CPW waveguides, has the
lowest optical insertion loss, but it lacks the substrate
trenches and does not support a comparably high mod-
ulation bandwidth as the other two designs. Design 2
has a lower VπL product than Design 3, but also has
a higher insertion loss. Design 3 offers superior perfor-
mance across all metrics, with a high extinction ratio,
low insertion loss, reasonable VπL product and high 3-
dB EO modulation bandwidth. The combination of these
attractive performance metrics with a scalable, standard
Si wafer based fabrication approach establishes a new
state-of-the-art in hybrid LN modulators, in our opinion.
Future work can improve performance by studying vari-
ations around this benchmark design. In this report, we
now include a broader discussion of hybrid modulators
as part of a PIC, followed by a comparison with earlier
results on hybrid LN modulators also fabricated using a
wafer-scale process with either Si or SiN.
The hybrid optical mode shown in Fig. 1(f) is suitable
for the straight phase-shift sections but does not sup-
port tight bends. However, the SiN1 waveguide can be
used, for example, to form microring resonators used in
frequency comb generation and optical filtering, and for
other structures such as folded modulators [24].
These EOM devices were designed primarily for in-
tegration as part of a larger PIC, and not as stand-
alone fiber-pigtailed devices.
Previously, the fiber-to-
waveguide coupling loss for the SiN1 layer waveguides in
the LIGENTEC process has been measured to be about
1 dB/facet using lensed fibers with a mode field diame-
ter of 2.5 µm. This reference design for the edge couplers
was not changed for these EOM devices. Therefore, the
potential fiber-to-fiber insertion loss can be estimated as
about 2 dB higher than the on-chip IL stated earlier.
In Ref. [19], hybrid SiN/LN modulators were demon-
strated in which the TFLN layer was not etched after
bonding. Here, LN was removed around all waveguides
except in the modulator phase-shift sections, so that the
established performance of LIGENTEC’s passive pho-
tonic components in the SiN1 layer was not impacted.
In the modulator segments, no etching or patterning of
the LN layer was performed for the purposes of optical
mode confinement, but some access holes were etched
through the LN and oxide layers into the Si substrate for
an undercut etch as described earlier. This LN removal
process does not require high resolution or low roughness.

<!-- page 7 -->
7
(b)
(d)
(c)
(a)
Laser
RF Power Meter
Frequency 
Multiplier
RF Signal Generator
OSA
1 – 50 GHz: none
47 – 78 GHz: x4
72 – 110 GHz: x6
0
20
40
60
80
100
Frequency (GHz)
0
20
40
60
80
100
Frequency (GHz)
0
10
20
30
40
50
Frequency (GHz)
0
-3
-6
2
1
0
1.5
0.5
3
-3
-6
0
Figure 4. (a) Schematic for the measurement of the electro-optic response (EOR). (b) EOR measurement data between 1 GHz
and 50 GHz for the three MZM designs on one test chip. (c) EOR measurement data between 1 GHz and 111 GHz for a
Design 3 EOM device, normalized to the corresponding EOR value at 1 GHz. The solid black line is a fit to the data using a
theoretical model of the EOM. (d) The calculated RF Vπ as a function of driving frequency using Eq. 1.
The low-frequency half-wave voltage length product
(VπL) of a hybrid LN MZM is given by the following
equation [43]
VπL = 1
2
neffλG
ne4r33Γmo
(2)
where neff is the effective refractive index of the optical
mode, λ is the wavelength, G is the effective electrode
gap distance between the ground and signal lines, ne is
the extraordinary index of refraction of LN, r33 is the
linear Pockels coefficient in the crystal z-direction along
which the RF fields are oriented, and Γmo is the mode
overlap integral between the optical mode and RF mode.
L is the length of the phase-shift section.
The factor
of 2 in the denominator is included as the structure is
driven in a push-pull configuration.
Although Eq. (2)
may not perfectly describe SWE structures such as T-
rail electrodes, it is a useful indicator of the role of the
various physical parameters.
The measured average VπL values of 4.29 V.cm (De-
sign 1), 3.40 V.cm (Design 2) and 3.83 V.cm (Design 3)
across the reported test sites can be compared to the
values expected from simulations, 2.75 V.cm (Design 1),
3.33 V.cm (Design 2) and 3.96 V.cm (Design 3).
De-
signs 2 and 3 agree quite well with the simulations (dif-
ference of about 2%); however, Design 1 shows a larger
difference. Given the limited number of test chips in this
batch, we do not have a conclusive explanation for this.
Among the many reasons for a difference between the
model and the data are the layer thickness and feature
size assumptions made in the former, and the site-to-site
(field) variations across the fabricated wafer from which
the test chips were extracted. The CPW and SWE elec-
trode designs are also differently sensitive to fabrication
variations [26]. Additional clarity into the deviations and
statistical uncertainties may be obtained in the future by
measuring more devices.
In previous work on hybrid LN/Si modulators, we have
reported VπL = 2.9 V.cm for G = 7 µm and VπL = 3.1
V.cm for G = 8 µm and neff = 2.0
[44].
Here, we
achieved VπL = 3.8 V.cm for G = 5 µm for hybrid
SiN modulators with neff = 1.8.
Using Eq. (2) with
these two earlier reference points, the ratio of the mode
overlap integrals in the hybrid SiN and Si modulators is
ΓSiN
mo /ΓSi
mo = 0.46 −0.49. One of the main reasons for
the lower value of Γmo is that we used a LN thickness of
300 nm here compared to 600 nm in Ref. [44]. Localiz-
ing more of the hybrid mode in LN generally improves
the modulation efficiency. Because of the higher refrac-
tive index of Si compared to SiN, a thicker LN slab could
be used without the hybrid mode becoming too large in
cross-sectional area [26]. Hybrid Si modulators can be in-

<!-- page 8 -->
8
herently more efficient than hybrid SiN modulators, but
replacing the Si with SiN has important benefits such
as achieving in lower insertion loss and potentially sup-
porting higher optical power levels and a wider optical
transparency window.
Compared to the hybrid LN/SiN modulators reported
in Ref. [45] (VπL = 8.8 V.cm with G = 7 µm for a single-
arm design, from which we can infer VπL = 4.4 V.cm
for a push-pull modulator), these modulators have an
improved efficiency. We show a 3-dB EO bandwidth be-
yond 100 GHz whereas no high-bandwidth modulation
was shown in Ref. [45].
We can also compare these results with hybrid LN/SiN
modulators made using a different approach [14]. For a
modulator of length 5 mm, Ref. [14] demonstrated about
31 GHz 3-dB EO bandwidth and VπL = 6.7 V.cm. Our
results show a large improvement in bandwidth from 31
GHz to 111 GHz, and a reduction (i.e., improvement) in
VπL from 6.7 to 3.8 V.cm.
At a different wavelength band, around 1310 nm, we
have reported high-bandwidth hybrid TFLN/SiN modu-
lators that achieved VπL = 2.7 V.cm for G = 5.5 µm [19].
While a direct comparison across different wavelength
bands is difficult, we can use Eq. (2) and scale the earlier
results to approximately VπL = 2.9 V.cm (for G = 5 µm
at 1.55 µm wavelengths). The VπL values of the mod-
ulators reported here are about 30% higher. However,
the modulators made using the single-layer SiN design of
Ref. [19] could not be used at 1550 nm since the inser-
tion loss was too high to be measured (more than 30 dB).
This major problem was overcome by using the bilayer
SiN design in this work which “pulls” the optical mode
below the bonded surface in the transition segment.
Finally, we emphasize that this work incorporates high-
performance EOMs with an established low-loss passive
photonics platform in a modular way. Designs, materials,
and fabrication processes were selected to not adversely
impact previously-established components, and to lower
overall risk. By relying on foundry processing, we can an-
ticipate iterative improvements of performance and scal-
ability to larger PICs. These advances can benefit a wide
range of users across many application areas, including
fiber and free-space communications; analog waveform
acquisition, generation, and processing; sensing; displays
and beamforming; and quantum optics technologies.
VI.
CONCLUSION
In conclusion, we have demonstrated hybrid LN/SiN
EO modulators integrated with low-loss SiN waveguides
made in a foundry process starting with standard (low-
resistivity) Si handle wafers. The optical waveguide rout-
ing, splitting, and tapering was implemented using the
SiN waveguides and no waveguide patterning was neces-
sary on the LN layer. These EOMs offer an attractive
combination of device performance metrics, with extinc-
tion ratio greater than 30 dB, about 3.8 dB on-chip in-
sertion loss, a low-frequency (1 kHz–1 MHz) half-wave
voltage-length product (VπL) around 3.8 V.cm, and a
high-frequency 3-dB EO modulation bandwidth exceed-
ing 110 GHz. This combination of parameters exceeds
the performance of earlier hybrid LN modulators. These
results show that scalable and cost-effective wafer-scale
processing can be used to add high-performance EOM
capability to a low-loss SiN-based passive photonics plat-
form.
Acknowledgments:
This research was developed in
part with funding from the U.S. Government. This paper
describes objective technical results and analysis. The
views, opinions and/or findings expressed are those of the
authors alone and should not be interpreted as represent-
ing the official views or policies of the U.S. Government.
[1] E. L. Wooten, K. M. Kissa, A. Yi-Yan, E. J. Murphy,
D. A. Lafaw, P. F. Hallemeier, D. Maack, D. V. Attana-
sio, D. J. Fritz, G. J. McBrien, and D. E. Bossi, A review
of lithium niobate modulators for fiber-optic communica-
tions systems, IEEE Journal of Selected Topics in Quan-
tum Electronics 6, 69 (2000).
[2] M. M. Howerton and W. K. Burns, Broadband traveling
wave modulators in LiNbO3, in RF Photonic Technol-
ogy in Optical Fiber Links, edited by W. S. C. Chang
(Cambridge University, 2002).
[3] H. Yu, D. Ying, M. Pantouvaki, J. Van Campenhout,
P. Absil, Y. Hao, J. Yang, and X. Jiang, Trade-off
between optical modulation amplitude and modulation
bandwidth of silicon micro-ring modulators, Optics Ex-
press 22, 15178 (2014).
[4] J. Witzens, High-Speed Silicon Photonics Modulators,
Proceedings of the IEEE 106, 2158 (2018).
[5] A. Jain, N. Hosseinzadeh, X. Wu, H. K. Tsang, R. Helkey,
J. E. Bowers, and J. F. Buckwalter, A High Spur-Free
Dynamic Range Silicon DC Kerr Ring Modulator for RF
Applications, Journal of Lightwave Technology 37, 3261
(2019).
[6] S. Shekhar, W. Bogaerts, L. Chrostowski, J. E. Bowers,
M. Hochberg, R. Soref, and B. J. Shastri, Roadmapping
the next generation of silicon photonics, Nature Commu-
nications 15, 751 (2024).
[7] G. Poberaj, H. Hu, W. Sohler, and P. G¨unter, Lithium
niobate on insulator (LNOI) for micro-photonic devices,
Laser & Photonics Reviews 6, 488 (2012).
[8] C. Wang, M. Zhang, X. Chen, M. Bertrand, A. Shams-
Ansari, S. Chandrasekhar, P. Winzer, and M. Lonˇcar,
Integrated lithium niobate electro-optic modulators op-
erating at CMOS-compatible voltages, Nature 562, 101
(2018).
[9] P. O. Weigel, J. Zhao, K. Fang, H. Al-Rubaye, D. Trotter,
D. Hood, J. Mudrick, C. Dallo, A. T. Pomerene, A. L.
Starbuck, C. T. DeRose, A. L. Lentine, G. Rebeiz, and
S. Mookherjea, Bonded thin film lithium niobate modu-

<!-- page 9 -->
9
lator on a silicon photonics platform exceeding 100 GHz
3-dB electrical modulation bandwidth, Optics Express
26, 23728 (2018).
[10] A. J. Mercante, S. Shi, P. Yao, L. Xie, R. M. Weikle, and
D. W. Prather, Thin film lithium niobate electro-optic
modulator with terahertz operating bandwidth, Optics
Express 26, 14810 (2018).
[11] M. He, M. Xu, Y. Ren, J. Jian, Z. Ruan, Y. Xu, S. Gao,
S. Sun, X. Wen, L. Zhou, L. Liu, C. Guo, H. Chen, S. Yu,
L. Liu, and X. Cai, High-performance hybrid silicon and
lithium niobate Mach–Zehnder modulators for 100 Gbits-
1 and beyond, Nature Photonics 13, 359 (2019).
[12] L. Chen, Q. Xu, M. G. Wood, and R. M. Reano, Hybrid
silicon and lithium niobate electro-optical ring modula-
tor, Optica 1, 112 (2014).
[13] P. O. Weigel,
M. Savanier,
C. T. DeRose,
A. T.
Pomerene, A. L. Starbuck, A. L. Lentine, V. Stenger,
and S. Mookherjea, Lightwave Circuits in Lithium Nio-
bate through Hybrid Waveguides with Silicon Photonics,
Scientific Reports 6, 22301 (2016).
[14] N. Boynton, H. Cai, M. Gehl, S. Arterburn, C. Dallo,
A. Pomerene, A. Starbuck, D. Hood, D. C. Trotter,
T. Friedmann, C. T. DeRose, and A. Lentine, A het-
erogeneously integrated silicon photonic/lithium niobate
travelling wave electro-optic modulator, Optics Express
28, 1868 (2020).
[15] Y. S. Lee, G.-D. Kim, W.-J. Kim, S.-S. Lee, W.-G. Lee,
and W. H. Steier, Hybrid Si-LiNbO3 microring electro-
optically tunable resonators for active photonic devices,
Opt. Lett. 36, 1119 (2011).
[16] P. Rabiei, J. Ma, S. Khan, J. Chiles, and S. Fathpour,
Heterogeneous lithium niobate photonics on silicon sub-
strates, Opt. Express 21, 25573 (2013).
[17] I. Bakish, R. Califa, T. Ilovitsh, V. Artel, G. Winzer,
K. Voigt, L. Zimmermann, E. Shekel, C. Sukenik, and
A. Zadok, Voltage-Induced Phase Shift in a Hybrid
LiNbO3-on-Silicon Mach-Zehnder Interferometer, in Ad-
vanced Photonics 2013 (OSA, Rio Grande, Puerto Rico,
2013) p. IW4A.2.
[18] V. Mere, F. Valdez, X. Wang, and S. Mookherjea, A
modular fabrication process for thin-film lithium niobate
modulators with silicon photonics, Journal of Physics:
Photonics 4, 024001 (2022).
[19] V. Mere, F. Valdez, and S. Mookherjea, Improved fabri-
cation of scalable hybrid silicon nitride–thin-film lithium
niobate electro-optic modulators, Journal of the Optical
Society of America B 40, D5 (2023).
[20] J. A. Smith, D. Jevtics, B. Guilhabert, M. D. Dawson,
and M. J. Strain, Hybrid integration of chipscale pho-
tonic devices using accurate transfer printing methods,
Applied Physics Reviews 9, 041317 (2022).
[21] T.
Vandekerckhove,
T.
Vanackere,
J.
De
Witte,
S. Cuyvers, L. Reis, M. Billet, G. Roelkens, S. Clem-
men, and B. Kuyken, Reliable micro-transfer printing
method for heterogeneous integration of lithium niobate
and semiconductor thin films, Optical Materials Express
13, 1984 (2023).
[22] A. Guarino, G. Poberaj, D. Rezzonico, R. Degl’Innocenti,
and P. G¨unter, Electro–optically tunable microring res-
onators in lithium niobate, Nature Photonics 1, 407
(2007).
[23] G. Chen, K. Chen, R. Gan, Z. Ruan, Z. Wang, P. Huang,
C. Lu, A. P. T. Lau, D. Dai, C. Guo, and L. Liu, High per-
formance thin-film lithium niobate modulator on a silicon
substrate using periodic capacitively loaded traveling-
wave electrode, APL Photonics 7, 026103 (2022).
[24] A. Della Torre, F. Dubois, H. Zarebidaki, A. Volpini,
J. Leo, A. Mettraux, A. Manzoor, I. Prieto, D. Grassani,
O. Dubochet, M. Despont, and H. Sattari, Folded electro-
optical modulators operating at CMOS voltage level in a
thin-film lithium niobate foundry process, Optics Express
33, 6747 (2025).
[25] S. Mookherjea, V. Mere, and F. Valdez, Thin-film lithium
niobate electro-optic modulators: To etch or not to etch,
Applied Physics Letters 122, 120501 (2023).
[26] X. Wang, P. O. Weigel, J. Zhao, M. Ruesing, and
S. Mookherjea, Achieving beyond-100-GHz large-signal
modulation bandwidth in hybrid silicon photonics Mach
Zehnder modulators using thin film lithium niobate, APL
Photonics 4, 096101 (2019).
[27] F. Valdez, V. Mere, X. Wang, N. Boynton, T. A. Fried-
mann, S. Arterburn, C. Dallo, A. T. Pomerene, A. L.
Starbuck, D. C. Trotter, A. L. Lentine, and S. Mookher-
jea, 110 GHz, 110 mW hybrid silicon-lithium niobate
Mach-Zehnder modulator, Scientific Reports 12, 18611
(2022).
[28] F. Valdez, V. Mere, X. Wang, and S. Mookherjea, In-
tegrated O- and C-band silicon-lithium niobate Mach-
Zehnder modulators with 100 GHz bandwidth, low volt-
age, and low loss, Optics Express 31, 5273 (2023).
[29] F. Valdez, V. Mere, and S. Mookherjea, 100 GHz Band-
width, 1 Volt Near-infrared Electro-optic Mach-Zehnder
Modulator, Optica 10, 578 (2023), arXiv:2211.13348
[physics].
[30] Y. Xue, R. Gan, K. Chen, G. Chen, Z. Ruan, J. Zhang,
J. Liu, D. Dai, C. Guo, and L. Liu, Breaking the band-
width limit of a high-quality-factor ring modulator based
on thin-film lithium niobate, Optica 9, 1131 (2022).
[31] Y. Pan, M. He, M. Xu, Z. Lin, Y. Lin, W. Ke, J. Liu,
Z. Lin, Y. Zhu, S. Gao, H. Li, X. Liu, C. Liu, S. Yu, and
X. Cai, Compact substrate-removed thin-film lithium
niobate electro-optic modulator featuring polarization-
insensitive operation, Optics Letters 47, 1818 (2022).
[32] M. Xu, S. Gao, H. Tan, and X. Cai, CMOS-level-voltage
Substrate-removed Thin-film Lithium Niobate Modula-
tor, in Optical Fiber Communication Conference (OFC)
2022 (Optica Publishing Group, San Diego, California,
2022) p. Th1J.3.
[33] S. Sakamoto, R. Spickermann, and N. Dagli, Narrow gap
coplanar slow wave electrode for travelling wave electro-
optic modulators, Electronics Letters 31, 1183 (1995).
[34] ´A. Rosa, S. Verstuyft, A. Brimont, D. V. Thourhout, and
P. Sanchis, Microwave index engineering for slow-wave
coplanar waveguides, Scientific Reports 8, 5672 (2018).
[35] J. Xu, C. Wang, Y. Tian, B. Wu, S. Wang, and
H. Zhang, Glass-on-LiNbO3 heterostructure formed via a
two-step plasma activated low-temperature direct bond-
ing method, Applied Surface Science 459, 621 (2018).
[36] F. Valdez, V. Mere, N. Boynton, T. A. Friedmann,
S. Arterburn, C. Dallo, A. T. Pomerene, A. L. Star-
buck, D. C. Trotter, A. L. Lentine, A. Kodigala, and
S. Mookherjea, Buried-Electrode Hybrid Bonded Thin-
Film Lithium Niobate Electro-Optic Mach-Zehnder Mod-
ulators, IEEE Photonics Technology Letters 35, 633
(2023).
[37] Y. Shi, L. Yan, and A. E. Willner, High-speed elec-
trooptic modulator characterization using optical spec-

<!-- page 10 -->
S1
trum analysis, Journal of Lightwave Technology 21, 2358
(2003).
[38] M. Lee, H. E. Katz, C. Erben, D. M. Gill, P. Gopalan,
J. D. Heber, and D. J. McGee, Broadband Modulation of
Light by Using an Electro-Optic Polymer, Science 298,
1401 (2002).
[39] C. Haffner, D. Chelladurai, Y. Fedoryshyn, A. Josten,
B. Baeuerle, W. Heni, T. Watanabe, T. Cui, B. Cheng,
S. Saha, D. L. Elder, Larry. R. Dalton, A. Boltasseva,
V. M. Shalaev, N. Kinsey, and J. Leuthold, Low-loss
plasmon-assisted electro-optic modulator, Nature 556,
483 (2018).
[40] D. Renaud, D. R. Assumpcao, G. Joe, A. Shams-Ansari,
D. Zhu, Y. Hu, N. Sinclair, and M. Loncar, Sub-1 Volt
and high-bandwidth visible to near-infrared electro-optic
modulators, Nature Communications 14, 1496 (2023).
[41] G. Ghione, Semiconductor Devices for High-Speed Opto-
electronics (Cambridge University, 2009).
[42] D. M. Pozar, Microwave Engineering, 4th ed. (Wiley,
2012).
[43] P. O. Weigel, F. Valdez, J. Zhao, H. Li, and S. Mookher-
jea, Design of high-bandwidth, low-voltage and low-loss
hybrid lithium niobate electro-optic modulators, Journal
of Physics: Photonics 3, 012001 (2021).
[44] F. Valdez, V. Mere, X. Wang, N. Boynton, T. A. Fried-
mann, S. Arterburn, C. Dallo, A. T. Pomerene, A. L.
Starbuck, D. C. Trotter, A. L. Lentine, and S. Mookher-
jea, 110 GHz, 110 mW hybrid silicon-lithium niobate
Mach-Zehnder modulator, Scientific Reports 12, 18611
(2022).
[45] M. Churaev, R. N. Wang, A. Riedhauser, V. Snigirev,
T. Bl´esin, C. M¨ohl, M. H. Anderson, A. Siddharth,
Y. Popoff, U. Drechsler, D. Caimi, S. H¨onl, J. Riemens-
berger, J. Liu, P. Seidler, and T. J. Kippenberg, A het-
erogeneously integrated lithium niobate-on-silicon nitride
photonic platform, Nature Communications 14, 3499
(2023).

<!-- page 11 -->
S2
Supplementary Information
Cross-sectional images of the device structure are
shown in Section S1. A representative optical transmis-
sion spectrum of the asymmetric Mach-Zehnder interfer-
ometer is shown in Section S2. Section S3 shows a com-
posite figure of the electro-optic response (EOR) for the
three design types across three different chips. A discus-
sion of the fitting function for the full-band electro-optic
response (EOR) is presented in Section S4.
S1.
CROSS-SECTIONAL IMAGES
Figure S1 shows tilted-angle scanning-electron micro-
scope (SEM) images of the cross-section of the EOM de-
vice with an air trench. In panel (a), the false-colored
SEM image shows the trench access holes and the trench
etched into the silicon (Si) substrate. The scalebar at the
bottom left corner of this image shows a length of 10 µm.
As discussed in the main text, removing some of the low-
resistivity Si substrate (resistivity about 10 Ω.cm) lowers
the RF propagation loss of the traveling-wave electrodes
and supports high-frequency electro-optic (EO) modula-
tion.
In panel (b), the image shows the layers that define the
optical hybrid mode, consisting of a SiN strip waveguide,
a thin layer of separation oxide, and the thin-film lithium
niobate (TFLN) slab with oxide above it. A portion of
the metal electrode is visible at the right-hand side edge
of the image. The scalebar at the bottom left corner of
this image shows a length of 1 µm. The rough edge was
seen because of unpolished manual cleaving of the chip.
For the device chips, the waveguide-fiber coupler at the
edge of the chip was defined using a SiN1 layer component
which has been previously studied by LIGENTEC and
offers about 1 dB per facet coupling loss to lensed fibers
with a mode-field diameter of about 2.5 µm at 1550 nm
(LIGENTEC unpublished data).
S2.
OPTICAL TRANSMISSION SPECTRUM
Figure S2 shows the optical transmission spectrum be-
tween 1520 nm and 1620 nm of a representative EOM
device without applying any voltage to the electrodes.
Similar data was obtained for all the test devices in this
batch (12 devices in total).
Biasing: The path length difference (PLD) segment
in the asymmetric Mach-Zehnder interferometer (MZI)
causes the periodicity seen in the transmission spectrum.
We used this information to determine the optimal wave-
length at which to test the performance of the EOM, by
selecting a wavelength at which the transmission is biased
3 dB below the peak transmission. This corresponds to
the linear-most portion of the inherently-nonlinear mod-
ulation transfer function (MTF) of a MZI-based modu-
lator device [41].
Optical Insertion Loss: The raw data was normalized
so that 0 dB corresponds to no additional insertion loss
compared to a straight-through waveguide only in the
SiN1 layer also fabricated on the same chip. Both de-
vices used the same type of waveguide-fiber edge cou-
plers and have the same physical length on the chip. We
first measured the EOM device, found the wavelength for
a transmission peak near 1550 nm, and then measured
the transmission of the straight-through waveguide at the
same wavelength. Subtracting the first number from the
second, we conclude that the EOM transmission was 3.80
dB lower than the waveguide transmission at 1549.89 nm,
which therefore gives the on-chip insertion loss of this
EOM device. As mentioned in the main text, the mea-
sured average on-chip IL (and one standard deviation)
for test devices across four chips in the three different
MZM designs were: 2.95 dB (1.43 dB) for Design 1, 6.43
dB (2.75 dB) for Design 2, and 3.77 dB (0.83 dB).
Extinction Ratio: An average extinction ratio (ER) of
about 30 dB was measured across a wide range of wave-
lengths around 1550 nm. The high ER shows that the
input and output 3-dB couplers were well-balanced, and
the losses in both arms of the push-pull MZI structure
were also well-balanced after fabrication was completed.
The PLD segment was defined in the SiN1 layer, as were
the 3-dB multi-mode interference (MMI) couplers at the
input and output segments of the MZI. The design and
performance of these components will not change if there
are minor variations in the SiN2, oxide or TFLN thick-
nesses across the wafer. The SiN1 process is already quite
mature at LIGENTEC.
S3.
ELECTRO-OPTIC RESPONSE (EOR)
MEASUREMENTS
Figure S3 shows the EOR data measured for EOMs
with Designs 1, 2 and 3 on three test chips.
These
chips were singulated from the wafer by dicing and have
slightly different layer thicknesses and feature dimen-
sions.
However, the EOR performance of each design
was similar.
The main paper discusses the normalization of the
EOR to its value at 1 GHz, or to the average value at the
lower frequencies between 1 GHz and 3 GHz, or alterna-
tive approaches. Here, for simplicity, each set of EOR
data was normalized to its value at 3 GHz.
Design 1 has the fastest roll-off and drops off by more
than 3 dB well before 50 GHz. (Thus, there was no need
for the RF multipliers to measure the 3-dB modulation
bandwidth of this design, which is around 20 GHz, and
the horizontal frequency axis of the first column in Fig. S3
is limited to 50 GHz.)
Designs 2 and 3 have a 3-dB
modulation bandwidth around 110 GHz or higher. Up to
the measurement limit of our apparatus (111 GHz), we
do not see evidence of a sharp roll-off in the EOR below

<!-- page 12 -->
S3
(a)
(b)
air trench
silicon substrate
silicon dioxide
access holes
T-rail electrodes
SiN waveguide
silicon dioxide
TFLN
Al electrodes
Figure S1. (a) Cross-sectional scanning-electron microscope (SEM) image showing the trenches etched into the Si substrate to
lower the propagation loss of the traveling-wave microwave mode. (b) Higher-resolution SEM image of a test chip showing the
SiN2 waveguide layer and the TFLN bonded region.
the horizontal dashed lines indicating the 3-dB range,
which is already visible in Design 1 below 50 GHz.
The EOR shown in the top right-hand side corner
(Chip 1, Design 3) is shown and discussed in the main
paper [Fig. 4(c) and 4(d)].
S4.
ELECTRO-OPTIC RESPONSE (EOR)
THEORY
As
stated
in
the
main
manuscript,
we
used
a
modulation-sideband method for measuring the electro-
optic response (EOR). A comparison against an EOR
measurement made using a lightwave component ana-
lyzer (LCA) was discussed in our earlier work [28].
The black curve in Fig. 4(c) [main paper] is the fitted
EO modulation response which was calculated using a
traveling-wave model of the EO interaction [41]:
m(ω) = RL + RG
RL

Zin
Zin + ZG


(ZL + Zc)F(u+) + (ZL −Zc)F(u−)
(ZL + Zc) exp[γmL] + (ZL −Zc) exp[−γmL]
 ,
(S1)
with the following definitions:
Zin = Zc
ZL + Zctanh(γmL)
Zc + ZLtanh(γmL),
(S2a)
γm = αm + jω
c nm.
(S2b)

<!-- page 13 -->
S4
1530
1550
1570
1590
1610
Wavelength (nm)
-30
-20
-10
0
Transmission (dBm)
  (1549.89 nm, -3.80 dBm)
Figure S2. The optical transmission of the asymmetric Mach-Zehnder interferometer structure that forms part of the EOM
devices.
-6
-3
0
EOR (dBe)
-6
-3
0
EOR (dBe)
-6
-3
0
EOR (dBe)
-6
-3
0
EOR (dBe)
0
20
40
Frequency (GHz)
-6
-3
0
EOR (dBe)
0
20
40
60
80
100
Frequency (GHz)
0
20
40
60
80
100
Frequency (GHz)
-6
-3
0
EOR (dBe)
Design 1
Design 2
Design 3
Chip 1
Chip 2
Chip 3
Figure S3. The electro-optic response (EOR) for Design 1 (red), Design 2 (blue), and Design 3 (black) on three different chips.
Each data set is individually normalized to the EOR value at 3 GHz.
F(u±(P)) = 1 −exp[u±]
u±
(S2c)
u±(P) = ±αmL + jω
c (±nm −ng)L.
(S2d)
where R{L,G} are the load and generator resistances,
Z{in,L,G} are the input, load, and generator impedances,
αm is the RF propagation loss, nm is the RF effective
index, and γm is the complex propagation constant of
the RF wave along the transmission line and Lps is the
phase-shifter interaction length.
Using this model, we use an iterative nonlinear pro-
gramming solver based on the Nelder-Mead Simplex
method (implemented in MATLAB as “fminsearch”) to
fit the measured data, with the following assumptions:
RL,G = 50 Ω, Lps = 0.6 cm, optical group index
ng = 2.0854 (obtained from a measurement). We assume
that the microwave loss increases with the square-root

<!-- page 14 -->
S5
of the modulation RF frequency f, αm(f) = αm0f 1/2,
where αm0 is the skin-effect loss coefficient and is a fit-
ting parameter to be found by optimization. This scal-
ing behavior is a good, but not perfect, fit to the data,
as we have reported elsewhere on a different batch of
microchips [28].
However, the exact value of the scal-
ing exponent (between 1/4 and 1/2 in some frequency
bands) does not greatly affect either the fitted line or the
conclusions. Since Eq. (S1) is nonlinear, the curve may
not be the unique fit to the data, but we have checked
that the same solution was obtained with slightly differ-
ent starting assumptions for the fitting parameters.
The black curve in Fig. 4(d) [main paper] is the mathe-
matical transformation using Eq. (1) [main paper] of the
black line in Fig. 4(c) [main paper].

