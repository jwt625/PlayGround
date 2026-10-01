---
paper_id: valdez2023
source_url: https://doi.org/10.1364/oe.480519
doi: 10.1364/oe.480519
license: journal=Optica-OA-License-v2; arxiv=CC-BY-NC-ND-4.0
sha256: caa328bcbc2cc1bd2db65c63441108bed40dec547ebc7a1a908c22131e8a96ae
pages: 21
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
arXiv:2211.05208v2  [physics.optics]  25 Jan 2023
Integrated O- and C-band Silicon-Lithium Niobate
Mach-Zehnder Modulators with 100 GHz Bandwidth,
Low Voltage and Low Loss
Forrest Valdez1,∗, Viphretuo Mere1, Xiaoxi Wang1, and Shayan
Mookherjea∗∗1
1University of California, San Diego, Department of Electrical and
Computer Engineering, 9500 Gilman Drive, MC 0407, La Jolla,
California, USA
*fgvaldez@eng.ucsd.edu, ∗∗smookherjea@ucsd.edu
Abstract
Broadband integrated thin-ﬁlm lithium niobate (TFLN) electro-optic modula-
tors (EOM) are desirable for optical communications and signal processing in both
the O-band (1310 nm) and C-band (1550 nm). To address these needs, we design
and demonstrate Mach-Zehnder (MZ) EOM devices in a hybrid platform based on
TFLN bonded to foundry-fabricated silicon photonic waveguides. Using a single
silicon lithography step and a single bonding step, we realize MZ EOM devices
which cover both wavelength ranges on the same chip. The EOM devices achieve
100 GHz EO bandwidth (referenced to 1 GHz) and about 2-3 V.cm ﬁgure-of-merit
(VπL) with low on-chip optical loss in both the O-band and C-band.
1
Introduction
There is considerable interest in designing and manufacturing thin-ﬁlm lithium niobate
(TFLN) Mach-Zehnder (MZ) electro-optic modulator (EOM) devices, based on the si-
multaneous achievements of high optical conﬁnement, high EO bandwidth, low voltage,
and low optical loss that have been demonstrated recently [1–5]. These attributes are
critically important for both optical communications and optical signal processing, and
for the development of multi-functional integrated photonic circuits with capabilities
beyond a traditional silicon (Si) photonics platform.
While the majority of TFLN
based modulators have been designed for the low-loss C-band (around 1550 nm) for
long-haul communication applications, the O-band (around 1310 nm) is important for
short-range optical ﬁber communications and has lower dispersion [6]. When design-
ing devices for diﬀerent wavelengths on a common platform, it is challenging to achieve
precise RF-optical index matching, since sub-micron-scale waveguides in high-index con-
trast platforms are highly dispersive (i.e., the eﬀective modal index, neﬀ, and the modal
cross-sectional area, Aeﬀ, vary with wavelength) compared to diﬀused LN waveguides.
There have already been some notable eﬀorts to design TFLN MZ-EOM devices using
a common approach at both these bands, e.g., by Stenger et al. [7] and Sun et al. [8];
however, state-of-the-art performance has not yet been demonstrated in this way. A
1

<!-- page 2 -->
uniﬁed O-band and C-band 100-GHz-class, few-volt integrated EOM device platform
may not only beneﬁt wideband optical communications, but also help in optical signal
processing, analog-to-digital conversion, frequency shifting, and EO instrumentation
throughout the wider spectral range at which integrated lasers and photodetectors are
now available in various photonics platforms [9–16].
Hybrid TFLN modulators can help integration and scalability of Pockels-eﬀect mod-
ulators with silicon photonics. Our devices are based on low-temperature direct bonding
of unetched TFLN to patterned and planarized Si waveguides, with the main fabrication
steps described in Ref. [17]. For a speciﬁc MZ-EOM device design with short transi-
tion and phase-shift sections, we have shown that 110 GHz 3-dB EO bandwidth can be
achieved while handling 110 mW of optical power in the C-band [18]. Here we study
a wider range of O-band and C-band devices systematically with diﬀerent optical and
RF designs, and explore strategies to reduce the half-wave voltage while also achieving
high bandwidth in the hybrid bonded Si/LN platform. We design O-band and C-band
100-GHz-class MZ EOM devices using the same layer thicknesses of the Si layer, the
TFLN layer and the thin oxide layer between the Si and the TFLN layers. The pre-
cise optical-RF index matching that is necessary for very high EO bandwidths (>100
GHz) was achieved in each wavelength band, ﬁrstly, by precisely tailoring the silicon
waveguide width, and secondly, by adjusting the RF traveling wave electrode based on
measurements of the layer thicknesses that are speciﬁc to each reticle of the wafer. Our
approach results in integrated EO modulators on a Si photonics wafer which achieve
>100 GHz EO bandwidth in both the O and C bands, a voltage-times-length ﬁgure-of-
merit V πL < 3 V.cm in the C-band and 2 V.cm in the O-band (with device lengths
up to L = 1 cm), and an on-chip optical insertion loss (including the 3-dB couplers
and phase-shifter sections) of less than 2 dB. These performance parameters lie well
beyond what is possible using carrier-depletion MZ modulators in a traditional Si pho-
tonics platform [19] while oﬀering easy integration with other Si photonic components
through the continuation of the Si feeder waveguides which continue and extend without
interruptions outside the bonded region [20,21].
In Section 2, we describe the layer and device structure, and compare the EOM
device designs for O-band and C-band operation. The measurements of the EO param-
eters are described in Section 3. Section 4 further discusses some aspects of these hybrid
Si-TFLN devices, followed by the conclusion in Section 5.
2
Hybrid Design for O and C Band Operation
A perspective view of the MZM device is shown in Fig. 1(a). A simpliﬁed cross-section
proﬁle is shown schematically in Fig.1(b), depicting only the layers which are necessary
for modulation; additional optical waveguide layers, dopants, vias or metal layers may
be included as part of the lower Si stack [22,23]. Diﬀerent Si waveguide widths are used
in the input, transition, and phase shifter sections, and wider Si features can conﬁne
nearly all the optical power and allow low-loss optical transitions in the feeder waveguide
across the bonded LN edges [24]. We could include both O-band and C-band designs
in the same Si mask by thinning the Si layer from the standard 220 nm to 150 nm
thickness. The thickness of this layer is nearly constant across the wafer. However, the
thickness of the oxide layer, labeled “hcmp” in Fig. 1(b), can vary across the wafer after
chemical-mechanical polishing (CMP), as we have reported elsewhere [25]. Here, hcmp
is about 40 nm on average, but more accurate information is necessary to design the
appropriate RF traveling-wave electrode structure on the layer of deposited Au on top
of TFLN. Therefore, hcmp was measured in each reticle after CMP was completed, by
2

<!-- page 3 -->
Figure 1: (a) A perspective-view schematic (not to scale) of the hybrid bonded Si/LN
MZM with slow-wave electrodes (SWEs). (b) A cross-section of the hybrid Si/LN MZM
in the phase-shifter section, with a thin oxide layer (thickness hcmp) between the Si and
LN layers. (c) An image of the fabricated chip with C-band (labeled Cxy) and O-band
(labeled Oxy) MZM devices. (d) The simulated conﬁnement factor of light in the LN
ﬁlm (left) and the eﬀective index (right) of the fundamental TE0 mode as a function
of the Si waveguide width. (e) The simulated optical group index of the hybrid mode
as a function of hcmp for the four Si waveguide dimensions (Cx and Ox). The black
diamonds mark the hcmp of the fabricated devices. The black dashed lines correspond
to the simulated RF index of the slow-wave electrodes.
performing ellipsometry on test features.
Figure 1(c) shows a top-view darkﬁeld optical microscope image of a fabricated
hybrid bonded Si/LN chip. The chip was fabricated using a hydrophilic bonding process
with the process described in Ref. [17]. The edge couplers for ﬁber coupling are to the
north-west and south-east edges (not shown in this reduced-area image). The transition
from a mode that is highly-conﬁned in Si (650 nm wide Si waveguide) to a mode in
which light is mostly in TFLN (< 320 nm wide Si waveguide) occurs under the bonded
TFLN area. The TFLN layer thickness is about 580 nm, and it is transparent and
cannot be easily seen in this image, but the perimeter of the bonded ﬁlm is labeled
“TFLN edge” in Fig. 1(c). An oxide cladding was deposited using a plasma-enhanced
chemical vapor deposition (PECVD) process, and the oxide edge is shown in Fig. 1(c).
Other relevant fabrication details are described in [17]. Multimode interference (MMI)
waveguide couplers are used to split or combine the light into two equally weighted
waveguides for the EO phase-shifter section. The MMI coupler width was 6 µm. The
O- and C-band MZMs require diﬀerent length couplers to maintain a high extinction
ratio (ER). Based on simulations, the 3-dB 2x2 MMI lengths were designed to be 49
µm and 39 µm for the O- and C-band devices, respectively. The corners of the MMIs
were angled to reduce optical reﬂections back to the source [26]. Note that in this case,
as opposed to our previous work [17,20], the bonded LN ﬁlm is over the MMI couplers
[see Figs. 1(a) and 1(c)] and does not aﬀect the coupling ratio due to the high optical
3

<!-- page 4 -->
conﬁnement in the Si layer for such wide features.
After the MMI coupler, the Si waveguide width is adiabatically tapered down from
the feeder section (650 nm width) to a smaller value (between 225 nm and 300 nm,
depending on the design) for the phase-shifter section. This results in increasing the
fraction of optical power that resides in the LN ﬁlm. This eﬀect is shown by the fraction
(ΓLN) of the integral over the cross-sectional proﬁle of the Poynting power which resides
in the LN region, as shown in Fig.
1(d).
The Si waveguide width also aﬀects the
optical group index, ng, of the hybrid mode, where the wider waveguide has a larger ng
and this will matched by the RF electrode design. To compare their performance, we
selected two Si waveguide widths for each wavelength band: The C-band modulators
have a hybrid Si waveguide width of either 300 nm or 275 nm (labelled as devices CW:
C-band, wide; and CN: C-band, narrow, respectively), while the O-band modulators
have a width of 250 nm or 225 nm (labelled as devices OW: O-band, wide; ON: O-band,
respectively).
As part of the wafer development processes, the wafers undergo CMP to achieve a
planar surface for bonding. The CMP oxide thickness was measured across the wafer
using ellipsometry and varies less than 2 nm in the bonded regions which are about
2 cm x 1 cm in size.
In the bonded stack, the CMP oxide layer is between the Si
waveguide and the bonded LN ﬁlm and impacts ng as shown in Fig. 1(e). Thinner
CMP oxide results in a tighter conﬁnement of the optical mode to the Si waveguide,
and increases ng and decreases the eﬀective modal area. For the measured chip, the
CMP oxide thickness is about 40 nm, resulting in a simulated ng of 2.38, 2.31, 2.43, and
2.30 for the CW, CN, OW, and ON devices respectively at 1550 nm and 1310 nm.
High-speed EO modulation using traveling wave coplanar waveguides requires index
matching between the RF and optical waves, ideally at all RF driving frequencies and
optical wavelengths of interest. For both Si and LN, the optical refractive index increases
as the wavelength of light decreases. The optical group index of the hybrid mode was
calculated as a function of chromatic (operating wavelength) and geometric dispersion
(the cross-sectional parameters), namely the Si dimensions, TFLN thickness, and the
amount of oxide between the Si strip and TFLN layer [see Fig. 1(e)]. Two diﬀerent
widths of the Si waveguide in each wavelength band were chosen (OW, ON, CW, and
CN), to study RF-optical index matching with either Si waveguide tapering or RF
electrode design.
Slow-wave electrodes (SWE) using inductively loaded or capacitively loaded ele-
ments were designed and fabricated to achieve RF-optical index matching between the
O- and C-band hybrid optical modes and the RF coplanar waveguide (CPW) mode.
SWE structures have been used to enable velocity matching of the electrodes to III-V
semiconductor waveguides [27,28], and in LNOI modulators as well, particularly when
using a low-RF-index substrate such as quartz [29,30] or for suspended/released modu-
lators [31]. SWEs are designed with either capacitively-loaded elements (T-rails), induc-
tively loaded elements (slots), or a combination thereof to maintain velocity matching
and impedance matching [28,32,33]. Figure 2(a) is an SEM image of one of the T-rail
SWEs used in this work with the design parameters labelled. These additional parame-
ters allow the RF dispersion (variation with frequency) to be adjusted. As the distance
between the inner electrode edge and the inner T-rail edge increases [h in Fig 2(a)],
the RF wave becomes slower which results in a larger RF eﬀective index. Furthermore,
the T-rail stem width [t, in Fig 2(a)] also aﬀects the RF wave velocity, with narrower
t corresponding to a slower RF ﬁeld [28]. These three diﬀerent SWE structures were
designed to achieve index matching to the four diﬀerent MZM designs as shown in Fig
2(b)-(d): a slot structure (L = t = 20 µm) for the CN and ON devices, a wider width
4

<!-- page 5 -->
Figure 2: (a) An SEM image of a T-rail SWE design used in this work with the SWE
parameters labelled. (b) The top-view schematic of the periodic inductively loaded slot
feature with slow-wave design parameters used for devices CN and ON. (c) The top-view
schematic of the periodic capacitively loaded T-rail feature used for the CW devices.
(d) The top-view schematic of the periodic capacitively loaded T-rail feature used for
the OW devices.
T-rail for CW (L = 20 µm, t = 10 µm), and a narrow width T-rail for OW (L = 20
µm, t = 5 µm). The dashed lines in Fig 1(e) show the simulated RF indices, nm, of the
three SWEs at 110 GHz fabricated on this chip.
The versatile electrode design oﬀers a useful functionality in overcoming the eﬀects
of minor imperfections in the fabrication of the hybrid devices. The layer thicknesses
were measured after planarization of the silicon and oxide layers was completed, and
we observed some variations in layer thicknesses from die-to-die across the wafer. Be-
fore fabricating the electrodes, we also measured the optical transmission through an
asymmetric Mach-Zehnder interferometer structure on the bonded chip, from which the
optical refractive index of the target structure can be veriﬁed. As Figure 1(e) shows,
the index matching condition can be achieved over a wide range of Si waveguide widths,
CMP oxide thickness, or operational wavelengths by tuning the SWE parameters and
tuning nm to match ng. In this way, the electrode design was slightly altered for each
pattern to trim devices and achieve the highest EO modulation bandwidths in each
case.
3
Measurements
3.1
Electrical Characterization
The electrical response of the modulators was measured using a two-port Keysight
PNA-X network analyzer with broadband frequency extenders, allowing for RF signals
from 100 MHz up to 118 GHz to be applied to the SWEs. High speed ground-signal-
5

<!-- page 6 -->
ground (GSG) probes (FormFactor Inﬁnity Probe) sourced and terminated the slow-
wave transmission lines of the MZMs. Figures 3(a)-(c) show the measured electrical
S-parameters of each of the three SWE designs (T-rail structure with G = 6 µm, T-rail
structure with G = 7 µm, and slot structure with G = 8 µm) with both 1.0 cm and
0.54 cm length electrodes. The measured RF |S21|2 of the 1.0 cm long electrodes have
a -6 dB drop at 75 GHz, while the 0.54 cm long electrodes have a -6 dB drop at greater
than 118 GHz.
Figure 3: Figure 3. The measured S-parameters of the 1.0 cm and 0.54 cm SWE with
the following designs: (a) T-rail and G = 6 µm, (b) T-rail and G = 7 µm, (c) Slot and
G = 8 µm.
The RF index, loss, and impedance of the transmission lines extracted from the
measured S-parameters are shown in Figs. 4(a) – (f) for each of the four electrode designs
[34]. Figures 4(a) and 4(d) show that the SWE structures result in an RF-optical index
mismatch of less than 1% for both the C-band and O-band devices for both electrode
lengths, where the dashed lines correspond to the simulated ng of the four respective
designs. The RF propagation loss, αm, for each of the lines is less than 9 dB/cm at
110 GHz for all the structures [Fig. 4(b) and 4(e)], while the measured characteristic
impedance, Zc, of the devices was about 40 Ωacross the measured frequency range [Fig.
4(c) and 4(f)]. As discussed in section 4.2, this deviation from perfect 50 Ωimpedance,
which is characteristic of the source and detector, slightly lowers the measured EO 3-dB
bandwidth. The RF back-reﬂection (|S11|2) is around -15 dB, and in some cases, around
-20 dB as shown in Fig. 3. The skin-loss coeﬃcient of the fabricated T-rail and Slot
SWEs is 0.7-0.8 dB/cm-GHz1/2, when the extracted RF loss is ﬁt to the typical square-
root frequency dependence equation. As there is less than 1% index mismatch between
the RF and optical traveling waves, the devices are RF loss limited. For these devices
the gold electrodes are 0.75 µm thick. The RF losses can be reduced by increasing the
electrode thickness via electroplating [35]. Alternatively, the Si substrate can be locally
removed [36], or a lower loss material can be used as the substrate (such as quartz [29]);
however, this will require further fabrication steps than needed here.
3.2
High-frequency EO response
Light from separate O- and C-band instrument-grade lasers was edge coupled to the chip
using a lensed ﬁber with a 4 µm nominal spot size. The laser wavelengths were set such
that the asymmetric MZMs were biased at quadrature. The optical input power from
the O- and C-band lasers was +12 dBm and +9 dBm, respectively. No optical ampliﬁers
were used in the measurements. The propagation loss of the wide Si feeder sections and
hybrid mode sections have been previously reported in the C-band as 0.8 dB/cm [21]
and 0.6 dB/cm [25], respectively. The wider feeder sections of the chip (only Si) that
6

<!-- page 7 -->
Figure 4: The extracted RF characteristics of each of the SWE devices from the mea-
sured S-parameters. (a) RF eﬀective index of the four 1.0 cm long SWEs. (b) RF
propagation loss of each of the four 1.0 cm long SWEs. (c) RF driving impedance of
each of the four 1.0 cm long SWEs. (d) RF eﬀective index of the four 0.54 cm long
SWE. (e) RF propagation loss of each of the four 0.54 cm long SWEs. (f) RF driving
impedance of each of the four 0.54 cm long SWEs. The dashed lines in panels (a) and
(d) correspond to the simulated ng for each of the C and O-band MZMs.
route from the input/output edge couplers to the hybrid phase-shifter section is 1.48
cm long; thus, 1.2 dB of the insertion loss is attributed to the feeder sections alone. The
MMI couplers contribute to 0.2 dB of the total loss (per coupler), while there is about
0.1 dB of loss per LN edge. Inverse tapers with a resolution-limited tip width of 180
nm (tapering to 650 nm over 300 µm) were implemented at the input and output edges
of the SOI chips to better match the lensed ﬁber mode to the waveguide modes in both
wavelength bands. The optical loss attributed to the phase-shifter section of the device
(hybrid mode along the electrode length) was estimated by comparing identical devices
with diﬀerent phase shifter lengths. For example, devices CN1 and CN2 are identical in
optical design (Si waveguide, LN ﬁlm) and RF design (SWE parameters), but the SWE
length is 0.46 cm longer for CN1. Assuming all other losses are common, the loss from
the phase-shifter would then be 1.5 dB/cm. The edge coupling losses are then estimated
to be 4.2 dB per edge and 4.8 dB per edge for the C- and O-band, respectively. Although
the waveguide taper tip width of 180 nm was limited by the resolution requirements of
the foundry process, lower edge coupling losses can be achieved with smaller resolution
or by using diﬀerent types of edge couplers [37–40]. The insertion loss of the phase-
shifter section, 3-dB MMI couplers, and LN transitions are thus calculated to be 1.6 dB
and 2.1 dB for the 0.54 cm and 1.0 cm long phase-shifters, respectively.
To measure the EO performance, a 50 Ωload resistor was used to terminate the
lines and a 110 GHz Keysight lightwave component analyzer (LCA) was used after full
calibration of the probing setup. Figures 5(a)-(d) and 6(a)-(d) show the measured EO
S21 normalized to 1 GHz for the four C-band and four O-band MZMs, respectively. The
red curves in Fig. 5 and Fig. 6 are the modelled EO responses for each device that was
calculated using the simulated ng [dashed lines in Fig 4(a) and 4(d)], with the measured
7

<!-- page 8 -->
electrical RF characteristics (nm, αm, and Zc) from Figs.
4(a)-(f) in the following
equations which are based on a traveling-wave model of the EO interaction [41]:
m(ω) = RL + RG
RL

Zin
Zin + ZG


(ZL + Zc)F(u+) + (ZL −Zc)F(u−)
(ZL + Zc) exp[γmL] + (ZL −Zc) exp[−γmL]
 ,
(1a)
Zin = Zc
ZL + Zctanh(γmL)
Zc + ZLtanh(γmL),
(1b)
γm = αm + jω
c nm.
(1c)
F(u±(P)) = 1 −exp[u±]
u±
(1d)
u±(P) = ±αmL + jω
c (±nm −ng)L.
(1e)
where RL,G are the load and generator resistances, Zin,L,G are the input, load, and
generator impedances, and γm is the complex propagation constant of the RF wave
along the transmission line.
Figure 5: The measured (blue) and modelled (red) EO responses of the hybrid bonded
Si/LN MZMs designed for C-band. (a) T-rail SWE with G = 7 µm and L = 0.54 cm.
(b) T-rail SWE with G = 7 µm and L = 1.0 cm. (c) Slot SWE with G = 8 µm and L
= 0.54 cm. (d) Slot SWE with G = 8 µm and L = 1.0 cm.
The measured RF S-parameters from an electrical only measurement (Fig. 3) were
used to model the EO response using Eq. (1a). However, because the impedance of our
SWE was not 50 Ω, the electrical measurements show oscillations versus RF frequency
8

<!-- page 9 -->
[Figs. 4(c) and 4(f)] that result from the impedance mismatch from the 50-40-50 Ω
system (Source-Transmission line-Termination) [42]. To approximately infer the “true”
EO response, because the magnitude of the ripples is relatively small (much less than
0.5 dB), the frequency-dependent Zc(f) data obtained from the PNA-X measurements
was ﬁt to a linear curve from 5 GHz to 110 GHz. Based on the agreement of the data
and the simulated performance to within a fraction of a dB over a wide range of RF
frequencies, we conclude that the EO model described by Eq. (1a) together with this
linear approximation captures the EO behavior adequately well. The initial drop-oﬀat
sub-10 GHz frequencies is due to a combination of skin-loss and the impedance mismatch
to the 50 Ωsource and load. The decaying oscillations that can be seen in Figs. 5 and
6 is also indicative of an impedance mismatch between the SWEs and the source and
termination loads. Furthermore, the measured EO S21 traces in Figs. 5 and 6 have a
gentle slope and do not reach the -6 dB point (where the Vπ of the device would increase
by a factor of two), nor the frequency cutoﬀregime. See section 4.1 and 4.2 below for
more details.
Figure 6: The measured (blue) and modelled (red) EO responses of the hybrid bonded
Si/LN MZMs designed for O-band. (a) T-rail SWE with G = 6 µm and L = 0.54 cm.
(b) T-rail SWE with G = 6 µm and L = 1.0 cm. (c) Slot SWE with G = 8 µm and L
= 0.54 cm. (d) Slot SWE with G = 8 µm and L = 1.0 cm.
3.3
Low-frequency VπL
To measure the near-DC (1 kHz) half-wave voltage (Vπ) of the O- and C-band modula-
tors, these asymmetric MZMs were biased to quadrature by varying the laser wavelength.
A waveform generator (Keysight 33600A) was used to apply 1 kHz sinusoidal signals to
the devices, and the modulated optical signal was captured with a photodetector and an
9

<!-- page 10 -->
oscilloscope. The voltage levels applied to the modulators were set higher than the ex-
pected Vπ as to ensure that the full π phase shift was induced. The waveform captured
on the oscilloscope was post-processed using software to map the optical transmission
to the applied voltage as shown in Fig. 7. From this relationship, a cosine-squared
ﬁt was used to evaluate Vπ. The value of L is taken from the design: there are two
phase-shifter interaction lengths of each of the four MZMs for L = 0.54 cm and 1.0 cm
(the blue and red traces, respectively in Fig. 7).
The half-wave voltage length product (VπL) of an MZM using LN is given by the
following equation
VπL =
neﬀλG
2n4er33Γmo
(2)
where neﬀis the eﬀective refractive index of the optical mode, λ is the operation wave-
length (in vacuum), G is the electrode gap distance between the ground and signal lines,
ne is the extraordinary index of LN, r33 is the linear Pockel’s coeﬃcient in the crystal
z-direction (30.8 pm/V), and Γmo is the mode overlap integral between the traveling
optical mode and RF mode. The factor of 2 in the denominator is included as the
structure is driven in a push-pull conﬁguration. Eq. (2) shows that the required driving
voltage for a π phase-shift reduces as the operation wavelength decreases. Furthermore,
the eﬀective area of the shorter wavelength optical modes is smaller, which implies that
the electrode gap G can be reduced without incurring high optical loss from optical
absorption in the metal electrodes. Thus, the OW designs show the most eﬃcient Vπ of
the group of devices reported here, as shown in Fig. 7(c) with Vπ of 3.78 V and 2.01 V
for the 0.54 cm and 1.0 cm long devices, respectively, which is a more eﬃcient scaling
of Vπ than would be predicted by the ratio of λ at 1310 nm and 1550 nm.
When considering the overall system requirements to drive an EOM, it is important
to know Vπ as a function of the modulation frequencies (> 1 GHz), because the impact of
RF loss, velocity mismatch, and impedance mismatch will also eﬀect the needed voltage
for a π phase-shift [43]. The driving voltage as a function of modulation frequency is
given by
Vπ(ω) = Vπ(DC)10−m(ω)/20
(3)
where m(ω) is the measured normalized (for example, to 1 GHz) EO S21. In this report,
Vπ(DC) is taken as the measured Vπ at 1 kHz as shown in Fig. 6 for each MZM. While
a more accurate representation would be to normalize m(ω) to the same frequency as
Vπ(DC), this approximation is valid as the modulator responses are ﬂat for frequencies
less than 1 GHz (see Fig. 11 below). Figure 8 shows the calculated RF Vπ which shows
the eﬀect of the SWE RF propagation loss, velocity, and impedance mismatch on the
required driving voltage for the C and O-band MZMs. An increase of the Vπ(DC) by
a factor of
√
2 and 2 correlate to the EO S21 of the modulator decreasing by 3-dB and
6-dB, respectively.
4
Discussion
Faced with the need to increase capacity, communications researchers are studying
multiband optical networks which extend wavelength coverage beyond the C and L
bands to other wavelength ranges such as the S, E and O bands [44]. This requires
new components, including switches and transceivers. Modulators in standard C-band
transceivers show impaired performance when tested at other wavelengths. While miti-
gation using digital signal processing (DSP) compensation is being studied [45,46], this
strategy adds to the cost and complexity. Complementary to software-based solutions
10

<!-- page 11 -->
Figure 7: The measured normalized optical power as a function of applied voltage to the
hybrid bonded Si/LN MZMs for (a) CW MZMs with G = 7 µm. (b) CN MZMs with G
= 8 µm, (c) OW MZMs with G = 6 µm, (d) and ON MZMs with G = 8 µm. The blue
traces correspond to the 0.54 cm long modulators and the red traces correspond to the
1.0 cm long modulators.
are hardware (i.e., device) improvements, such as improving the modulator design.
Lithium niobate modulators have been compared favorably to InP-based modulators
for multiband operation, and TFLN modulators, in particular, have been identiﬁed as
a potential future solution to the known limitations of current-generation LN modula-
tors [45, 46]. However, these modulators are not yet mature, and many fundamental
aspects of the new device platform are under study.
Here, we showed that both O-band and C-band hybrid TFLN modulators can be
made using the same TFLN layer thickness, the same Si layer thickness and with a single
bonding step, with the only diﬀerences being in the silicon rib waveguide width, coupler
design, and the customized electrodes, which are fabricated on top of the TFLN layer
after bonding and handle removal. Within the design of each modulator, the MMI-based
coupler is superior to directional-coupler designs for wider bandwidth, but customized
designs were needed for the O-band and the C-band, and thus, the devices are not fully
interchangeable, i.e., the C-band modulator is not ideal for O-band operation, or vice
versa. Nevertheless, in our approach, the Si waveguide width is easily controlled, and the
MMI couplers are designed in the Si layer, outside the bonded (hybrid) region, and they
are not sensitive to the additional fabrication steps such as bonding and handle removal.
When integrated with multiplexers, such hybrid modulators can easily cover wideband
operation on the same chip, and deliver high-bandwidth and low-voltage performance
in each band. A comparison between the diﬀerent types of modulators studied here is
11

<!-- page 12 -->
Figure 8: The calculated RF Vπ as a function of driving frequency using Eq. (3) for
the following Si/LN MZMs: (a) CW MZMs with G = 7 µm. (b) CN MZMs with G =
8 µm, (c) OW MZMs with G = 6 µm, (d) and ON MZMs with G = 8 µm. The blue
traces correspond to the 0.54 cm long modulators and the red traces correspond to the
1.0 cm long modulators.
presented in the following sections.
4.1
O-band and C-band EO Device Comparison
Both the O-band and C-band devices show high bandwidth and low voltage. As shown
in Fig. 9(a), the C-band modulators have a VπL of 2.9 V.cm to 3.1 V.cm, whereas
the O-band modulators have a lower VπL of 2.0 V.cm to 2.3 V.cm. This is because the
higher extraordinary index of refraction of LN, and the tighter optical mode conﬁnement
which allows for a reduced electrode gap at shorter wavelengths. The reduced gap has
a more eﬃcient EO eﬀect, without incurring excess optical propagation loss from the
proximity of the metal structures to the mode. These O-band devices can be further
improved to have the Si waveguide width narrowed further and still maintain single
mode operation and similar conﬁnement factor (ΓLN) as the C-band devices as shown
in Fig. 1(e).
The wide-Si modulators (CW and OW) have a hybrid optical mode which allows for
an electrode gap spacing of G = 7 µm and 6 µm for CW and OW, respectively, instead
of 8 µm for the narrow-Si modulators (CN and ON). The hybrid mode is more conﬁned
in the Si region than the LN ﬁlm compared to the narrower designs (CN and ON), but
the mode eﬀective area is smaller. This tighter conﬁnement allows for the electrode gap
distance to be decreased, which also increases the modulation eﬃciency.
12

<!-- page 13 -->
As seen in Fig. 9(b), the 3-dB bandwidths of the 0.54 cm and 1.0 cm long MZMs are
greater than 100 GHz and greater than 60 GHz, respectively for both O and C band
designs. Note that although the initial 3-dB point of the wide Si waveguide 0.54 cm
long MZMs (CW and OW in Fig. 5(a) and Fig. 6(a), respectively) occurs at 75 GHz,
the response remains ﬂat until 100 GHz with a slope of -0.013 dB/GHz (meaning a 1
dB drop over 100 GHz). An eﬀective EOR slope from 10 GHz to 110 GHz is shown
in Fig. 9(c) and ranges between -0.012 dB/GHz and -0.029 dB/GHz. This indicates
that the modulators still have usable bandwidth up to, and probably higher than, 110
GHz.
Figure 9(d) summarizes the combined ﬁgure-of-merit, 3-dB bandwidth-to-Vπ
ratio, of each modulator which shows the advantage of the O-band devices over the C-
band counterparts. While both sets of modulators have been optimized for high-speed
performance, the reduction of driving voltage for the O-band set allows for a higher
BW/Vπ which in turn reduces the power needed per high-speed modulation in a RF-
photonic link. This may be helpful since communications in data centers and passive
optical networks is energy-constrained.
Figure 9: Summary bar graphs of the measured (a) VπL, (b) 3-dB bandwidth (BW), (c)
EO S21 slope (from 10 GHz to 110 GHz), and (d) 3-dB BW-to-Vπ ratio for the hybrid
bonded Si/LN C-band and O-band MZMs of the fabricated chip.
4.2
Discussion: RF Impedance
The drop in EO response by about 2 dB at low frequencies (around 1-10 GHz) that
is seen in the EO measurements [see Fig. 5 and 6] is attributed to the characteristic
impedance of these traveling SWEs. Due to a design error, the impedance is around
13

<!-- page 14 -->
40 Ω[see Fig. 4(d) and 4(f)], and is not matched to the source and load impedances,
which are both 50 Ω. The shorter devices (L = 0.54 cm) show this eﬀect more clearly
than the longer devices (L = 1.0 cm), where the additional RF loss reduces the back-
reﬂection, and a similar eﬀect has been seen in traditional LN modulators [42]. The
impedance is sensitive to the electrode signal width to electrode gap ratio [41,47] and
the mismatch in these devices can be corrected by redesigning the separation distance
between the arms of the MZM. This would require a new round of lithography in the
silicon layer followed by planarization, bonding, handle removal and electrode formation
(i.e., a complete process ﬂow). Alternatively, to avoid the cost of full refabrication, an
on-chip termination can be fabricated on these devices to provide a customized matching
condition to the SWE lines [35,48]. A smaller load impedance (for example, 30 Ω) would
result in peaking the response (when normalized to a certain frequency such as 1 GHz);
however, this would be at the cost of causing larger back-reﬂections to the RF source,
which is undesirable.
Figure 10: The measured (blue) EO responses of the hybrid bonded Si/LN MZMs
designed for: (a) O-band, Slot SWE with G = 8 µm and L = 1.0 cm. (b) C-band, Slot
SWE with G = 8 µm and L = 1.0 cm. (c) O-band, Slot SWE with G = 8 µm and
L = 0.54 cm. (d) C-band, Slot SWE with G = 8 µm and L = 0.54 cm. The red and
yellow curves correspond to the modelled EO response using Eq (1) and the measured
RF characteristics in Fig 3(d)-(f), assuming ZL of 50 Ωand 40 Ω, respectively.
In the future, we believe that achieving a better match to 50 Ωsource and termi-
nation loads, while maintaining the same levels of index matching and RF propagation
loss that have already been achieved, should substantially improve the EO bandwidth.
Figure 10 is similar to the traces shown in Figs. 5 and 6, with the addition of a yellow
line which corresponds to the modelled EO responses with ZL = 40 Ωto match the char-
acteristic impedance of the fabricated SWEs. In this case, the 3-dB EO bandwidths of
14

<!-- page 15 -->
the O- and C- band MZMs would be around 110 GHz for the 1.0 cm long modulators,
and greater than 110 GHz for the 0.54 cm long modulators. This, in turn, will also
decrease the RF Vπ (Fig. 8).
Low frequency bias drift can aﬀect x-cut LN modulators when there is a buﬀer oxide
layer between the driving electrodes and LN surface [49–52]. There is no such oxide
layer in our devices since the gold (Au) electrodes are patterned directly on the LN
surface. While we do not know if surface charge accumulation plays a signiﬁcant role,
we think it is unlikely as there are large Au ground planes in contact with the surface.
In our measurements, we did not use a DC bias but instead, tuned the wavelength with
a tunable laser, as each of the modulators has an asymmetric path-length diﬀerence.
Long-term drift measurements and compensation using a bias controller will be studied
in the future. We have observed that the measured value of Vπ is constant in the range of
0.1 to 10 MHz [18]. The simulation model ﬁts the measurements well, when assuming
that the transmission line impedance is 40 Ωsystem, as shown in Fig. 11, and the
measured and modelled responses are ﬂat from 0.1 to 1 GHz.
4.3
Discussion: Normalization of EO Response
The EO response was measured over 100 MHz to 110 GHz using the available range
of the LCA instrument. The actual high frequency behavior of |m(ω)| is not aﬀected
by what value of ω is chosen for the denominator in Eq. (1a) Since the actual value
of |m(ω = 0)| in Eq. (1a) is not known, it is usually taken as the value at 1 GHz
[17,20,29,53] though values as high as 5 GHz have also been used [54]. For the devices
(ON1 and ON2) shown in Fig. 11, the EO response changes by +0.11 dB from 100 MHz
to 1 GHz for an L = 0.54 cm device, and by -0.77 dB for an L = 1.0 cm device. Often-
quoted parameters such as the 3-dB bandwidth do depend on the reference point, which
should therefore be clearly stated. Since the half-wave voltage Vπ can be measured down
to very low frequencies, it is convenient to plot Vπ(ω) as shown in Fig. 8, which shows
a direct relationship to both m(ω) through Eq. (3), and graph the trend all the way
from near DC to the highest modulation frequencies measured by the LCA.
5
Conclusion
In conclusion, we have demonstrated that a hybrid bonded Si/TFLN platform can be
used for high-speed and low voltage EOMs in both the O and C wavelength bands using
the same fabrication process, a common layer stack, and on the same chip. Standard Si
manufacturing processes were used to deﬁne the optical routing, splitting, and tapering
of the waveguides. The top oxide layer of the SOI wafer was chemical-mechanically
polished to provide a bondable surface and no etching or patterning of the LN layer
is required. We show that the velocity matching condition for high-speed modulation
can be precisely achieved (to less than 1%) by changing the Si waveguide width (tuning
ng) and the parameters of the travelling slow-wave electrode structures (tuning nm)
across diﬀerent wavelength bands to account for both chromatic dispersion and local
variations in geometry. Four designs were reported here, two in the O-band and two in
the C-band, which diﬀer only in the widths of the features in Si layer and in the traveling
slow-wave electrode designs. The high-frequency EO response of the 0.54 cm and 1.0 cm
long devices cross the 3 dB line (referenced to 1 GHz) at about 100 GHz and 60 GHz,
respectively, for both O and C bands. However, a single number such as 3 dB bandwidth
may not fully capture the device performance, since the gentle slopes of the modulator
response, and the fact that the EO response has not reached the cut-oﬀfrequency
15

<!-- page 16 -->
Figure 11: The measured (solid) and modelled (dashed) EO response normalized to 0.1
GHz for the ON devices with 8 µm gap slot SWEs.
regime indicates useful EO bandwidth beyond 110 GHz in both cases. While all eight
reported modulators are shown to have high electro-optic modulation bandwidth, the
O-band devices prove to have a higher modulation eﬃciency, and the combination of
high bandwidth and low voltage can greatly beneﬁt short-range communications in data
centers.
Funding
NASA (80NSSC17K0166), ONR (N00014-21-1-2805), DOD (HR001120S0008), U.S.
Government.
Acknowledgments
The authors thank: M. R¨using, P. O. Weigel and J. Zhao (formerly of UC San Diego)
for earlier contributions and discussions on this topic; A. Lentine, N. Boynton, T. A.
Friedman, S. Arterburn, C. Dallo, A. T. Pomerene, A. L. Starbuck, and D. C. Trotter
(Sandia National Laboratories) for discussions and fabrication assistance; C. Coleman,
R. Scott, B. Szafraniec, G. Vanwiggeren, V. Moskalenko, K.K. Abdelsalam and G.
Lee (Keysight Technologies) for discussions and measurement assistance. Part of this
work was performed at the San Diego Nanotechnology Infrastructure (SDNI) of UCSD,
a member of the National Nanotechnology Coordinated Infrastructure, which is sup-
ported by the National Science Foundation (Grant ECCS-2025752). This research was
developed in part with funding from the Defense Advanced Research Projects Agency
(DARPA) and the U.S. Government. This paper describes objective technical results
and analysis. The views, opinions and/or ﬁndings expressed are those of the authors
alone and should not be interpreted as representing the oﬃcial views or policies of the
Department of Defense or the U.S. Government.
16

<!-- page 17 -->
Disclosures
The authors declare no conﬂicts of interest.
Data availability.
Data underlying the results presented in this paper are not publicly available at this
time but may be obtained from the authors upon reasonable request.
References
[1] Wolfgang Sohler, Hui Hu, Raimund Ricken, Viktor Quiring, Christoph Vannahme,
Harald Herrmann, Daniel B¨uchter, Selim Reza, Werner Grundk¨otter, Sergey Orlov,
Hubertus Suche, Rahman Nouroozi, and Yoohong Min. Integrated optical devices
in lithium niobate. Optics and Photonics News, 19(1):24–31, 2008.
[2] Andreas Boes, Bill Corcoran, Lin Chang, John Bowers, and Arnan Mitchell. Status
and Potential of Lithium Niobate on Insulator (LNOI) for Photonic Integrated
Circuits. Laser & Photonics Reviews, 12(4):1700256, feb 2018.
[3] Cheng Wang, Mian Zhang, Xi Chen, Maxime Bertrand, Amirhassan Shams-Ansari,
Sethumadhavan Chandrasekhar, Peter Winzer, and Marko Lonˇcar.
Integrated
lithium niobate electro-optic modulators operating at CMOS-compatible voltages.
Nature, 562(7725):101–104, 2018.
[4] Mingbo He, Mengyue Xu, Yuxuan Ren, Jian Jian, Ziliang Ruan, Yongsheng Xu,
Shengqian Gao, Shihao Sun, Xueqin Wen, Lidan Zhou, Lin Liu, Changjian Guo,
Hui Chen, Siyuan Yu, Liu Liu, and Xinlun Cai. High-performance hybrid silicon
and lithium niobate Mach-Zehnder modulators for 100 Gbit s−1 and beyond. Nature
Photonics, 13(5):359–364, 2019.
[5] Di Zhu, Linbo Shao, Mengjie Yu, Rebecca Cheng, Boris Desiatov, CJ Xin, Yaowen
Hu, Jeﬀrey Holzgrafe, Soumya Ghosh, Amirhassan Shams-Ansari, Eric Puma, Neil
Sinclair, Christian Reimer, Mian Zhang, and Marko Loncar. Integrated photonics
on thin-ﬁlm lithium niobate. Advances in Optics and Photonics, 13(2):242–352,
2021.
[6] Physical layer speciﬁcations and management parameters for 40 gb/s and 100 gp/s
operation over ﬁber optic cables. IEEE Standard for Ethernet.
[7] Vincent Stenger, James Toney, Andrea Pollick, James Busch, Jon Scholl, Peter Pon-
tius, and Sri Sriram. Engineered thin ﬁlm lithium niobate substrate for high gain-
bandwidth electro-optic modulators. In CLEO: Science and Innovations, pages
CW3O–3. Optica Publishing Group, 2013.
[8] Shihao Sun, Mingbo He, Siyuan Yu, and Xinlun Cai. Hybrid silicon and lithium
niobate Mach-Zehnder modulators with high bandwidth operating at C-band and
O-band. In CLEO: Science and Innovations, pages STh1F–4. Optica Publishing
Group, 2020.
[9] Molly Piels and John E. Bowers. Photodetectors for silicon photonic integrated
circuits. Elsevier Ltd, 2016.
17

<!-- page 18 -->
[10] Alexander W Fang, Hyundai Park, Oded Cohen, Richard Jones, Mario J Paniccia,
and John E Bowers. Electrically pumped hybrid AlGaInAs-silicon evanescent laser.
Optics Express, 14(20):9203–9210, 2006.
[11] Guang-Hua Duan, Christophe Jany, Alban Le Liepvre, Alain Accard, Marco
Lamponi, Dalila Make, Peter Kaspar, Guillaume Levaufre, Nils Girard, Fran¸cois
Lelarge, Jean-Marc Fedeli, Antoine Descos, Badhise Ben Bakir, Sonia Messaoudene,
Damien Bordel, Sylvie Menezo, Guilhem de Valicourt, Shahram Keyvaninia, Gun-
ther Roelkens, Dries Van Thourhout, Davd J. Thomson, Frederic Y. Gardes, and
Graham T. Reed. Hybrid III–V on Silicon Lasers for Photonic Integrated Circuits
on Silicon. IEEE Journal of Selected Topics in Quantum Electronics, 20(4):158–
170, 2014.
[12] Xianshu Luo, Yulian Cao, Junfeng Song, Xiaonan Hu, Yuanbing Cheng, Cheng-
ming Li, Chongyang Liu, Tsung Yang Liow, Mingbin Yu, Hong Wang, Qi Jie Wang,
and Patrick Guo Qiang Lo. High-throughput multiple dies-to-wafer bonding tech-
nology and III/V-on-Si hybrid lasers for heterogeneous integration of optoelectronic
integrated circuits. Frontiers in Materials, 2(April):1–21, 2015.
[13] Dongjae Shin, Jungho Cha, Sunggu Kim, Yongwhak Shin, Kwansik Cho, Ky-
oungho Ha, Gitae Jeong, Hyeongsun Hong, Kyupil Lee, and Ho-Kyu Kang. O-band
DFB laser heterogeneously integrated on a bulk-silicon platform. Optics Express,
26(11):14768–14774, 2018.
[14] Keshuang Li, Zizhuo Liu, Mingchu Tang, Mengya Liao, Dongyoung Kim, Huiwen
Deng, Ana M Sanchez, R Beanland, Mickael Martin, Thierry Baron, Siming Chen,
Jiang Wu, Alwyn Seeds, and Huiyan Liu. O-band InAs/GaAs quantum dot laser
monolithically integrated on exact (0 0 1) Si substrate. Journal of Crystal Growth,
511:56–60, 2019.
[15] Davide Colucci, Marina Baryshnikova, Yuting Shi, Yves Mols, Muhammad
Muneeb, Yannick De Koninck, Didit Yudistira, Marianna Pantouvaki, Joris
Van Campenhout, Robert Langer, Dries Van Thourhout, and Bernardette Kunert.
Unique design approach to realize an O-band laser monolithically integrated on 300
mm Si substrate by nano-ridge engineering. Optics Express, 30(8):13510–13521,
2022.
[16] Pengyan Wen, Preksha Tiwari, Svenja Mauthe, Heinz Schmid, Marilyne Sousa,
Markus Scherrer, Michael Baumann, Bertold Ian Bitachon, Juerg Leuthold, Bernd
Gotsmann, and Kirsten E. Moselund. Waveguide coupled III-V photodiodes mono-
lithically integrated on Si. Nature Communications, 13(1):1–11, 2022.
[17] Viphretuo Mere, Forrest Valdez, Xiaoxi Wang, and Shayan Mookherjea. A modular
fabrication process for thin-ﬁlm lithium niobate modulators with silicon photonics.
J.Phys. Photonics, 4(2):024001, 2022.
[18] Forrest Valdez, Viphretuo Mere, Xiaoxi Wang, Nicholas Boynton, Thomas A Fried-
mann, Shawn Arterburn, Christina Dallo, Andrew T Pomerene, Andrew L Star-
buck, Douglas C Trotter, Anthony L Lentine, and Shayan Mookherjea. 110 GHz,
110 mW hybrid silicon-lithium niobate Mach-Zehnder modulator. Scientiﬁc Re-
ports, 12(1):1–11, 2022.
[19] Jeremy Witzens. High-speed silicon photonics modulators. Proceedings of the IEEE,
106(12):2158–2182, 2018.
18

<!-- page 19 -->
[20] Xiaoxi Wang, Forrest Valdez, Viphretuo Mere, and Shayan Mookherjea. Monolithic
Integration of 110 GHz Thin-ﬁlm Lithium Niobate Modulator and High-Q Silicon
Microring Resonator for Photon-Pair Generation. In 2022 Conference on Lasers
and Electro-Optics (CLEO), pages 1–2. IEEE, 2022.
[21] Xiaoxi Wang, Forrest Valdez, Viphretuo Mere, and Shayan Mookherjea. Integrated
thin-silicon passive components for hybrid silicon-lithium niobate photonics. Optics
Continuum, 1(10):2233–2244, 2022.
[22] Michael Rusing, Peter O Weigel, Jie Zhao, and Shayan Mookherjea. Toward 3d
integrated photonics including lithium niobate thin ﬁlms: a bridge between elec-
tronics, radio frequency, and optical technology. IEEE Nanotechnology Magazine,
13(4):18–33, 2019.
[23] Nicholas Boynton, Hong Cai, Michael Gehl, Shawn Arterburn, Christina Dallo,
Andrew Pomerene, Andrew Starbuck, Dana Hood, Douglas C. Trotter, Thomas
Friedmann, Christopher T. DeRose, and Anthony Lentine. A heterogeneously in-
tegrated silicon photonic/lithium niobate travelling wave electro-optic modulator.
Optics Express, 28(2):1868–1884, 2020.
[24] Peter O. Weigel, Marc Savanier, Christopher T. Derose, Andrew T. Pomerene, An-
drew L. Starbuck, Anthony L. Lentine, Vincent Stenger, and Shayan Mookherjea.
Lightwave Circuits in Lithium Niobate through Hybrid Waveguides with Silicon
Photonics. Scientiﬁc Reports, 6(November 2015):1–9, 2016.
[25] Peter O. Weigel, Jie Zhao, Kelvin Fang, Hasan Al-Rubaye, Douglas Trotter, Dana
Hood, John Mudrick, Christina Dallo, Andrew T. Pomerene, Andrew L. Star-
buck, Christopher T. DeRose, Anthony L. Lentine, Gabriel Rebeiz, and Shayan
Mookherjea. Bonded thin ﬁlm lithium niobate modulator on a silicon photonics
platform exceeding 100 GHz 3-dB electrical modulation bandwidth. Optics Express,
26(18):23728, 2018.
[26] Jin Zhang, Liangshun Han, Bill Ping-Piu Kuo, and Stojan Radic. Broadband an-
gled arbitrary ratio SOI MMI couplers with enhanced fabrication tolerance. Journal
of Lightwave Technology, 38(20):5748–5755, 2020.
[27] NAF Jaeger and Zachary KF Lee.
Slow-wave electrode for use in compound
semiconductor electrooptic modulators.
IEEE Journal of Quantum Electronics,
28(8):1778–1784, 1992.
[28] S. R. Sakamoto, R. Spickermann, and N. Dagli.
Narrow gap coplanar slow
wave electrode for travelling wave electro-optic modulators. Electronics Letters,
31(14):1183–1185, 1995.
[29] Prashanta Kharel, Christian Reimer, Kevin Luke, Lingyan He, and Mian Zhang.
Breaking voltage–bandwidth limits in integrated lithium niobate modulators using
micro-structured electrodes. Optica, 8(3):357–363, 2021.
[30] Xuecheng Liu, Bing Xiong, Changzheng Sun, Jian Wang, Zhibiao Hao, Lai Wang,
Yanjun Han, Hongtao Li, Jiadong Yu, and Yi Luo. Wideband thin-ﬁlm lithium nio-
bate modulator with low half-wave-voltage length product. Chinese Optics Letters,
19(6):060016, 2021.
19

<!-- page 20 -->
[31] Gengxin Chen, Kaixuan Chen, Ranfeng Gan, Ziliang Ruan, Zong Wang, Pucheng
Huang, Chao Lu, Alan Pak Tao Lau, Daoxin Dai, Changjian Guo, and Liu Liu.
High performance thin-ﬁlm lithium niobate modulator on a silicon substrate using
periodic capacitively loaded traveling-wave electrode. APL Photonics, 7(2):026103,
2022.
[32] R Spickermann and N Dagli. Millimetre wave coplanar slow wave structure on
GaAs suitable for use in electro-optic modulators. Electronics Letters, 29(9):774–
775, 1993.
[33] ´Alvaro Rosa, Steven Verstuyft, Antoine Brimont, Dries Van Thourhout, and Pablo
Sanchis. Microwave index engineering for slow-wave coplanar waveguides. Scientiﬁc
Reports, 8(1):1–8, 2018.
[34] David M Pozar. Microwave engineering. John Wiley & Sons, 2011.
[35] Xuecheng Liu, Bing Xiong, Changzheng Sun, Zhibiao Hao, Lai Wang, Jian Wang,
Yanjun Han, Hongtao Li, and Yi Luo. Capacitively-loaded thin-ﬁlm lithium niobate
modulator with ultra-ﬂat frequency response. IEEE Photonics Technology Letters,
34(16):854–857, 2022.
[36] Zong Wang, Gengxin Chen, Ziliang Ruan, Ranfeng Gan, Pucheng Huang, Zhi-
wen Zheng, Liwang Lu, Jun Li, Changjian Guo, Kaixuan Chen, and Liu Liu.
Silicon–Lithium Niobate Hybrid Intensity and Coherent Modulators Using a Pe-
riodic Capacitively Loaded Traveling-Wave Electrode. ACS Photonics, 9(8):2668–
2675, 2022.
[37] Jing Wang, Yi Xuan, Chunghun Lee, Ben Niu, Lei Liu, Gordon Ning Liu, and
Minghao Qi. Low-loss and misalignment-tolerant ﬁber-to-chip edge coupler based
on double-tip inverse tapers. In Optical Fiber Communication Conference, pages
M2I–6. Optica Publishing Group, 2016.
[38] Lianxi Jia, Chao Li, Tsung-Yang Liow, and Guo-Qiang Lo. Eﬃcient suspended
coupler with loss less than- 1.4 dB between Si-photonic waveguide and cleaved
single mode ﬁber. Journal of Lightwave Technology, 36(2):239–244, 2018.
[39] Xiaodong Wang, Xueling Quan, Min Liu, and Xiulan Cheng.
Silicon-nitride-
assisted edge coupler interfacing with high numerical aperture ﬁber. IEEE Pho-
tonics Technology Letters, 31(5):349–352, 2019.
[40] Xin Mu, Sailong Wu, Lirong Cheng, and HY Fu. Edge couplers in silicon photonic
integrated circuits: A review. Applied Sciences, 10(4):1538, 2020.
[41] Giovanni Ghione. Semiconductor devices for high-speed optoelectronics. Cambridge
University Press, 2009.
[42] Ganesh K Gopalakrishnan, William K Burns, Robert W McElhanon, Catherine H
Bulmer, and Arthur S Greenblatt. Performance and modeling of broadband LiNbO3
traveling wave optical intensity modulators.
Journal of Lightwave Technology,
12(10):1807–1819, 1994.
[43] Marta M Howerton and William K Burns. Broadband traveling wave modulators
in LiNbO3. Cambridge University Press, 2002.
20

<!-- page 21 -->
[44] Nicola Sambo, Vittorio Curri, Gangxiang Shen, Mattia Cantono, Joao Pedro, and
Erwan Pincemin. Guest editorial: Multi-band optical networks. Journal of Light-
wave Technology, 40(11):3360–3363, 2022.
[45] Gabriele Di Rosa, Robert Emmerich, Matheus Sena, Johannes K. Fischer, Colja
Schubert, Ronald Freund, and Andr´e Richter. Characterization, monitoring, and
mitigation of the i/q imbalance in standard c-band transceivers in multi-band sys-
tems. Journal of Lightwave Technology, 40(11):3470–3478, 2022.
[46] Robert Emmerich, Matheus Sena, Robert Elschner, Carsten Schmidt-Langhorst,
Isaac Sackey, Colja Schubert, and Ronald Freund. Enabling s-c-l-band systems with
standard c-band modulator and coherent receiver using coherent system identiﬁ-
cation and nonlinear predistortion. Journal of Lightwave Technology, 40(5):1360–
1368, 2022.
[47] Amirmahdi Honardoost, Reza Saﬁan, Ashutosh Rao, and Sasan Fathpour. High-
speed modeling of ultracompact electrooptic modulators.
Journal of Lightwave
Technology, 36(24):5893–5902, 2018.
[48] Xingrui Huang, Yang Liu, Zhiyong Li, Zhongchao Fan, and Weihua Han. High-
performance and compact integrated photonics platform based on silicon rich
nitride–lithium niobate on insulator. APL Photonics, 6(11):116102, 2021.
[49] Syoji Yamada and Makoto Minakata.
DC drift phenomena in LiNbO3 optical
waveguide devices. Japanese Journal of Applied Physics, 20(4):733, 1981.
[50] CM Gee, GD Thurmond, H Blauvelt, and HW Yen. Minimizing dc drift in LiNbO3
waveguide devices. Applied Physics Letters, 47(3):211–213, 1985.
[51] Hirotoshi Nagata and Kazumasa Kiuchi. Temperature dependence of dc drift of
Ti: LiNbO3 optical modulators with sputter deposited SiO2 buﬀer layer. Journal
of Applied Physics, 73(9):4162–4164, 1993.
[52] Jean Paul Salvestrini, Laurent Guilbert, Marc Fontana, Mustapha Abarkan, and
Stephane Gille.
Analysis and control of the DC drift in LiNbO3 based Mach–
Zehnder modulators. Journal of Lightwave Technology, 29(10):1522–1534, 2011.
[53] Sean P Nelan, Andrew Mercante, Shouyuan Shi, Peng Yao, Eliezer Shahid, Ben-
jamin Shopp, and Dennis W Prather. Integrated lithium niobate intensity mod-
ulator on a silicon handle with slow-wave electrodes. IEEE Photonics Technology
Letters, 34(18):981–984, 2022.
[54] Md Samiul Alam, Essam Berikaa, and David V Plant. Net 350 Gbps/λ IMDD
transmission enabled by high bandwidth thin-ﬁlm lithium niobate MZM. IEEE
Photonics Technology Letters, 34(19):1003–1006, 2022.
21

