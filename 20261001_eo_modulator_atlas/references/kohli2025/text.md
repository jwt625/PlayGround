---
paper_id: kohli2025
source_url: https://doi.org/10.1038/s41377-025-02116-1
doi: 10.1038/s41377-025-02116-1
license: CC-BY-4.0
sha256: 74edaccf994c961facc81a6c190d54f9415f744f8e288abe82dd83c4d63123c7
pages: 11
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Kohli et al. Light: Science & Applications (2025) 14:399 
www.nature.com/lsa
https://doi.org/10.1038/s41377-025-02116-1
A R T I C L E
O p e n A c c e s s
The plasmonic BTO-on-SiN platform – beyond 200
GBd modulation for optical communications
Manuel Kohli
1✉, Daniel Chelladurai
1, Laurenz Kulmer1, Tobias Blatter1, Yannik Horst1, Killian Keller
1,
Michael Doderer1, Joel Winiger1, David Moor
1, Andreas Messner1, Tatiana Buriakova2, Clarissa Convertino3,
Felix Eltes3, Yuriy Fedoryshyn1, Ueli Koch
1 and Juerg Leuthold
1✉
Abstract
An integrated photonics platform that offers high-speed modulators in addition to low-loss and versatile passive
components is highly sought after for different applications ranging from AI to next-generation Tbit/s links in optical
ﬁber communication. For this purpose, we introduce the plasmonic BTO-on-SiN platform for high-speed electro-optic
modulators. This platform combines the advantages provided by low-loss silicon nitride (SiN) photonics with the
highly nonlinear barium titanate (BTO) as the active material. Nanoscale plasmonics enables high-speed modulators
operating at electro-optical bandwidths up to 110 GHz with active lengths as short as 5 µm. Here, we demonstrate
three different modulators: a 256 GBd C-band Mach-Zehnder (MZ) modulator, a 224 GBd C-band IQ modulator – being
both the ﬁrst BTO IQ and the ﬁrst IQ modulator on SiN for data communication – and ﬁnally, a 200 GBd O-band
racetrack (RT) modulator. With this approach we show record data rates of 448 Gbit/s with the IQ modulator and
340 Gbit/s with the MZ modulator. Furthermore, we demonstrate the ﬁrst plasmonic RT modulator with BTO and how
it is ideally suited for low complexity communication in the O-band with low device loss of 2 dB. This work leverages
the SiN platform and shows the potential of this technology to serve as a solution to combat the ever-increasing
demand for fast modulators.
Introduction
A high-speed photonic platform that offers a combi-
nation of electro-optic modulators with low-loss passive
components is essential in many ﬁelds where light must
be manipulated with electrical signals. For instance, such
a combination can be crucial to further advance Tbit/s
optical communication links1, photonic quantum com-
puting2, input/output interfaces to cryogenic environ-
ments3–6,
disaggregated
AI
systems7,
microwave
photonics8,9, and optical computing10,11. To meet the
increased total trafﬁc demands and complexity in systems,
the ideal integrated optical platform should offer high
speed operation, a compact footprint, enable operation
across the largest possible spectral range, and be able to
handle high input powers.
There exists a variety of different platforms and
approaches for integrated high-speed modulators. An
important metric to demonstrate the potential of a tech-
nology is the maximum achievable signal bandwidth.
Demonstrations over 200 GBd in the C-band include
BTO plasmonics with symbol rates up to 216 GBd12, POH
up to 256 GBd13, and TFLN up to 260 GBd14. Most of
these demonstrations were realized by intensity modula-
tion and direct detection (IM/DD) schemes. Coherent
transmission, on the other hand, is ideal for high data
rates
in
long-haul
communications,
where
complex
modulation
formats
with
IQ
modulators
allow
the
encoding of information in both phase and amplitude of
the light. Examples of IQ modulators operating beyond
100
GBd
include
plasmonic-organic
hybrid
(POH)
experiments showing single-polarization net-data rates of
790 Gbit/s at 160 GBd15, silicon photonics with polariza-
tion multiplexed (PMUX) net-data rates of 1 Tbit/s at
105 GBd16, thin-ﬁlm lithium niobate (TFLN) with PMUX
net-rates of 1.96 Tbit/s at 130 GBd17, and InP with PMUX
© The Author(s) 2025
OpenAccessThisarticleislicensedunderaCreativeCommonsAttribution4.0InternationalLicense,whichpermitsuse,sharing,adaptation,distributionandreproduction
in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if
changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If
material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain
permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.
Correspondence: Manuel Kohli (mkohli@ethz.ch) or Juerg Leuthold
(Leuthold@ethz.ch)
1ETH Zurich, Institute of Electromagnetic Fields, Zurich, Switzerland
2Ligentec SA, Ecublens, Switzerland
Full list of author information is available at the end of the article
1234567890():,;
1234567890():,;
1234567890():,;
1234567890():,;

<!-- page 2 -->
net-rates of 2.03 Tbit/s at 192 GBd18. To achieve such
high numbers, most demonstrations utilize probabilistic
shaping
of
high-order
modulation
formats
(up
to
100QAM and more) in addition to PMUX, where data is
encoded on carriers with orthogonal polarization. In
contrast, the O-band is ideally suited for low-complexity,
and therefore low-cost applications with IM/DD, due to
low ﬁber dispersion19. Recently, directly modulated lasers
have been demonstrated to achieve 256 GBd in the
O-band20, which is an attractive option for intra-data
center communications. Integrated solutions with high
symbol rates include TFLN21,22 modulators operating at
210 GBd and InP absorption modulators at 256 GBd23.
Although impressive demonstrations, high-speed mod-
ulators could generally beneﬁt from the advanced passive
performance, scalability, ultra-low loss24,25, transparency
across a large spectral range26,27, and the ability to handle
high input power due to negligible two-photon absorption
and relatively low stimulated Brillouin scattering28 offered
by SiN photonics. Its advantages have brought forward
impressive demonstrations such as frequency comb gen-
eration29,30, on-chip ampliﬁers31, quantum sources32, and
lasers33,34. Yet, SiN does not offer an electro-optic effect
to modulate the light at high frequencies. Among all
nonlinear effects, the Pockels effects is particularly inter-
esting as it offers a pure phase modulation35. Within
interferometric conﬁgurations, one can also implement
amplitude or intensity modulation. With the demonstra-
tion of ultra-low losses in TFLN36, it has been developed
into a highly versatile platform with a Pockels effect for
many different applications37,38. In contrast to TFLN,
however, SiN is CMOS compatible and already available
on 300 mm wafers as it can be grown directly on silicon
photonics. Therefore, signiﬁcant effort has been dedicated
to integrating Pockels-effect modulators onto the SiN
platform.
Examples
include
PZT-based
modulators
reaching
40 GBd39,
SiN-loaded
TFLN
reaching
80 GBd40,41,
and
heterogeneously
integrated
TFLN
reaching 80 GBd42–45. SiN loaded BTO has been lever-
aged to demonstrate ultra-low-power tuning46. By utiliz-
ing a plasmonic slot waveguide, it is possible to achieve
modulation in a most compact footprint and very high
speeds. More recently, we introduced plasmonic BTO-on-
SiN and demonstrated 216 GBd in the C-band12. BTO has
emerged as a viable candidate for integrated electro-optic
modulators47–50. It offers one of the largest Pockels
coefﬁcients among known materials51, it is suited for
cryogenic environments3, and allows wafer-scale integra-
tion with advanced platforms2,52. Furthermore, BTO with
the combination of plasmonics offers exceptionally good
thermal stability up to 250 ° C in addition to longtime
stable operation at 90 ° C53. It is therefore conceivable that
the combination of SiN with BTO gives access to a scal-
able integrated optical platform that offers highest speed
on a compact footprint, with an ability to handle high
power across a wide spectrum.
In this work, we show the potential of the BTO-on-SiN
integrated optical platform that allows for the imple-
mentation of a wide variety of active components targeted
for different applications, more speciﬁcally short-haul and
long-haul communication. For instance, we demonstrate
fully integrated Mach-Zehnder (MZ), IQ, and racetrack
(RT) modulators. Operation with this BTO-on-SiN plat-
form is shown up to highest speed of 256 GBd and line
rates of up to 448 Gbit/s. Such high speed has been
enabled by implementing plasmonic metal-insulator-
metal active sections that operate up to 110 GHz. The
favorable frequency response enables operation with as
little as 1.13 Vpp at 200 GBd – which by any standard
makes it an attractive solution for driverless operation.
The plasmonic approach further allows integration of the
active components on a most compact footprint, which
may be as small as 5, 15 and 17.5 μm for the RT, the MZ
and the IQ modulators, respectively. This work demon-
strates the ﬁrst BTO-based IQ modulator and the ﬁrst IQ
modulator on SiN. The IQ modulator is the ideal con-
ﬁguration for long-haul communication. The combina-
tion of the low loss SiN passive technology with the
resonant BTO-plasmonic RT modulators in O-band,
ideally suited for low-cost intensity-modulation/direct-
detection scheme, yields devices with low 2 dB on-chip
losses. Lastly, we show operation across a large spectral
window by demonstrating devices for the 1300 and
1550 nm window. All devices were fabricated on the same
chip with the same method, and we thus elevate the
plasmonic BTO approach to a modulator platform that
can be tailored to the speciﬁc application. To tackle the
energy efﬁciency required for new optical engines, we
demonstrate that high-speed performance of these mod-
ulators is achieved even with low-complexity linear ofﬂine
digital signal processing (DSP), which could enable
energy-efﬁcient
real-time
processing.
The
plasmonic
BTO-on-SiN platform can reach up to 196 GBd with
linear equalization only.
The paper is based in part on work that was initially
presented at OFC’24 and CLEO’24 conferences54,55.
Results
BTO-on-SiN platform
The
BTO-on-SiN
platform
aims
to
combine
the
advantages provided by the low-loss SiN, the nonlinear
BTO with one of the highest Pockels effect among known
materials, and the nanoscale plasmonics offering highest
speed in most compact conﬁgurations. A basic building
block of a Pockels modulator is the phase shifter. More
complex conﬁgurations of modulators, such as the MZ
and IQ, can be composed of multiple phase shifters. A
schematic of the high-speed plasmonic BTO-on-SiN
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 2 of 11

<!-- page 3 -->
phase shifter, capable of 110 GHz operation, is shown in
Fig. 1a. The modulator can be divided into two main
parts, the SiN passives used to route the light on the chip,
and the phase shifters on the top layers composed of BTO
and gold.
The silicon nitride waveguide is fully embedded in SiO2
and features a cross-section of 800 ´ 800 nm2 in the
C-band and 600 ´ 800 nm2 in the O-band. The ﬁber-to-
chip coupling is solved by employing amorphous silicon
overlay grating couplers in a simple scheme. By adding a
metallic mirror to the same structure, the coupling efﬁ-
ciency is improved from −2.2 dB to −1.4 dB in experi-
ment and from −1.1 dB to −0.44 dB in simulation56. The
O-band gratings are designed to be dual-polarization with
a high efﬁciency of −2.5 dB in simulation and −3 dB in
experiment for coupling both TE and TM polarizations
with the same grating56. A heater based on a 100-nm-
thick and 2-µm-wide platinum strip can be used to con-
trol the phase relation in interferometric conﬁgurations
such as MZ and IQ modulator. This metal is located
~1 µm above the SiN waveguide separated by SiO2 clad-
ding. For future improvements in terms of footprint and
energy efﬁciency, the phase tuners could be optimized by
increasing the thermal isolation57, or replaced for example
with solutions based on Pockels effect or on liquid
crystals46,58–60.
The phase shifters are composed of the following parts:
a vertical directional coupler (VDC) to couple the light to
the BTO, a photonic-to-plasmonic converter (PPC) to
focus the light into the active section, and the plasmonic
waveguide. The active phase shifters are composed of a
metal-insulator-metal waveguide with gold and BTO to
form the plasmonic slot waveguide. To couple light from
the SiN into the phase shifters, there are two stages. First,
the signal is coupled to a BTO waveguide with a VDC
from the SiN waveguide layer, see Fig. 1b. The length of
the VDC in the C-band modulators, i.e., the MZ and the
IQ modulator, is 80 µm in length. For the O-band RT
modulator, it is shortened to 40 µm to achieve a shorter
round-trip path for the light in the RT section. In Fig. 1b,
the cross-section of the directional coupler is shown. The
800-nm-thick SiN waveguide is tapered from a width of
800 nm to 200 nm, whereas the ~200-nm-thick BTO
waveguide located roughly 100 nm above the SiN, is
tapered from a width of 150 nm to the single-mode
waveguide of 800 nm in the C-band and 600 nm in the
O-band. Cutback measurements in the C-band indicate
that the propagation loss in the BTO is as low as 4.5 dB/
cm and therefore has negligible losses for the short pro-
pagation distances of below 200 µm. The transition loss of
the VDC is 0.14 dB/transition determined from cutback
measurements. In the PPC, light is focused into the
plasmonic slot by tapering the BTO waveguide and
bringing the metal closer until it touches the BTO to form
the plasmonic slot, see Fig. 1c. The working principle of
this plasmonic converter is described in our previous
work and 3D ﬁnite-element (FEM) simulations indicate
losses below 1 dB12. The measured losses are discussed in
Silicon dioxide
Silicon nitride
Barium titanate
Gold
Platinum
a
b
c
IQ
e
f
d MZM
Racetrack
Fig. 1 Schematic of the BTO-on-SiN platform. a Overview of the high-speed BTO-on-SiN modulator featuring the combined advantages of low-
loss SiN, highly nonlinear BTO, and nanoscale plasmonics. The high-speed device consists of a directional coupler from SiN to BTO, a short BTO
waveguide (<200 µm in length), a photonic-to-plasmonic converter and a plasmonic waveguide section. b Cross-section of the SiN-to-BTO
directional coupler with below 0.2 dB/transition loss. c Cross-section of the plasmonic BTO waveguide featuring two gold metal plates with BTO in
between. Schematics of the different measured modulators with d C-band MZ modulator, e C-band IQ modulator, and f O-Band RT modulator
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 3 of 11

<!-- page 4 -->
the sections below. With the routing done in SiN, this
phase shifter can be placed on top of more complicated
structures to form different types of modulators. In the
following, we discuss the experimental results of three
modulator types: the C-band MZ, the C-band IQ, and the
O-band resonant RT modulators.
C-Band MZ modulator
A schematic overview of the C-Band MZ modulator is
shown in Fig. 2a. Light is ﬁrst split into two arms with
multi-mode interferometers in SiN. It is then coupled to
the plasmonic phase-shifter sections and mapped back to
waveguides where they are recombined in a multi-mode
interferometer coupler (MMI). Figure 2b shows an optical
microscope image of the fabricated device. The plas-
monic phase-shifter section is 150 nm wide and 15 µm
long. The photonic-to-plasmonic converter is 15 µm
long. The ﬁber-to-ﬁber insertion losses are shown in
Fig. 2c. In the current fabrication run, we found insertion
losses (ILs) of −20.3 dB at 1550 nm. Through cutback
measurements, we found grating coupler losses of 2.8 dB
per coupler, plasmonic propagation losses of 0.5 dB/μm
in the 150-nm-wide slot, and 3.5 dB loss per photonic-to-
plasmonic coupler. However, simulations and reference
measurements indicate that the ﬁber-to-ﬁber losses can
be ideally as low as 8.1 dB. These lower losses can be split
onto 5 dB of losses in the plasmonic section (0.33 dB/µm
as derived from measured material properties), 1 dB per
PPC (3D FEM simulations, see ref. 12), 0.1 dB per VDC
transition for a total of 0.2 dB, and 0.44 dB per grating
coupler with metallic mirror, see ref. 56. There is thus
room for fabrication improvement. Particularly, the PPC
show much higher losses than anticipated due to fabri-
cation issues.
We measure a Vπ of 3.6 V in the phase shifter or 1.8 V
in the Mach-Zehnder conﬁguration in push-pull conﬁg-
uration at DC. The frequency response of the phase
shifters in the MZ modulator from 10 MHz to 110 GHz is
shown in Fig. 2d. We ﬁnd a drop-off between the MHz
and the lower GHz ranges, leading to a V π of 6.4 V at
40 GHz.
We
extract
an
effective
Pockels
effect
of
~180 pm/V at 40 GHz by comparing measured and
simulated Vπ. The drop is due to the frequency depen-
dence of the Pockels effect in BTO and can be directly
observed in a plasmonic conﬁguration12,48. The response
ﬂattens after the initial drop and is followed by a small
resonance around 75 GHz. We attribute this resonance to
an LC peaking due to parasitic inductances of the prober
set-up and device. The frequency response drops off
around 110 GHz. The drop-off can be explained by an RC
limit. This occurs due to the capacitance of the modulator
and the 50 Ohm source. We expect improvements in the
V π and bandwidth with further optimizations of the
device cross section and fabrication. Nevertheless, the
BTO plasmonic MZ modulator reaches 110 GHz with a
3-dB drop-off between 10 GHz and 110 GHz, which suf-
ﬁces for highest symbol rates of 256 GBd. Therefore, this
modulator is more than capable of being employed in the
next generations of Tbit/s links.
C-Band IQ modulator
The schematic of the IQ modulator can be seen in Fig. 3a.
It is composed of two parallel MZ modulators constituting
the in-phase and quadrature phase modulators. A third
a
Heater
VDC
Conv.
PS
Frequency response [GHz]
10
35
60
85
110
–3
0
3
6
–6
Modulation [dB]
d
Fiber-to-fiber
transmission [dB] 
1540
1550
1560
Wavelength [nm]
–20
–25
c
b
MZM
Schematic MZM
Optical microscope image MZM
Fiber-to-fiber insertion loss
Frequency response
Fig. 2 Characterization of C-band MZ modulator. a Schematic of the MZ modulator with phase shifters in each arm and a platinum heater to set
the operating point of the modulator. b Optical microscope image of fabricated MZ modulators. The highlighted area shows a single device. c Fiber-
to-ﬁber insertion losses. d The initial drop below 10 GHz is inherent to the frequency response of the BTO Pockels effect as seen in plasmonic devices.
Yet, the modulator’s frequency response relevant for the high-speed data experiment drops only 3 dB in the range between 10 GHz and 110 GHz
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 4 of 11

<!-- page 5 -->
platinum heater is added to set the phase difference
between the in-phase and quadrature signals. The footprint
of a single IQ modulator is 0.75 ´ 2.15 mm2, dominated by
the footprint of the three heaters (400 ´ 150 µm2). The
plasmonic sections in this IQ modulator are 17.5 µm in
length and 100 nm wide. Figure 3d shows an optical
microscope image of the chip containing the IQ mod-
ulators. The diced PIC was placed on a PCB. Wire bonds
connect the PIC to the PCB to contact the IQ modulators
with the DC current sources for the heaters.
The ﬁber-to-ﬁber insertion loss of the IQ modulator is
shown in Fig. 3f. The modulator features a total insertion
loss of 22.5 dB at 1530 nm and 23.9 dB at 1550 nm.
Through cutback measurements, we attribute ~2.75 dB to
the grating couplers, 12.25 dB to plasmonic losses in the
active section (~0.7 dB/µm) in the 100 nm wide plasmonic
section and ~ 3dB per photonic-to-plasmonic coupler.
Simulations indicate that the losses can be reduced to
~12 dB
ﬁber-to-ﬁber
IL
in
this
conﬁguration.
This
includes higher losses due to the narrower slot and longer
length in comparison to the MZ modulator.
The electro-optic response was measured by applying a
sinusoidal signal to the phase shifters in the IQ mod-
ulator. Figure 3f shows the electro-optic modulation as a
function of frequency. We measure the characteristic
drop between MHz and GHz, similarly to the MZ mod-
ulator. However, in the IQ modulator, the higher capa-
citance, due to a longer modulator and a smaller
plasmonic gap, limits the bandwidth. The frequency
response starts to drop at ~70 GHz. We measure a Vπ of
4 V in the phase-shifter or 2 V in the Mach-Zehnder
conﬁguration at DC voltages. The increased V π in com-
parison to the MZ modulator could potentially arise from
surface effects or dead layers in the BTO due to etching53,
since the longer length and smaller width of the mod-
ulator should result in a lower V π. Additionally, the
bandwidth limitations further suggest that wider slots are
preferable.
1530
1550
1540
1560
1570
Fiber-to-fiber
transmission [dB]
Wavelength [nm]
–28
–3
0
3
Modulation [dB]
b
f
a
Heater
MZM 1
110
–25
–6
MZM 2
–9
–12
Frequency [GHz]
10
35
60
85
–22
Fiber-to-fiber insertion loss
e
d
10 μm  
c
Schematic IQ modulator
Frequency response
Fig. 3 Characterization of C-band IQ modulator. a Schematic of the IQ modulator consisting of two MZ modulators. b Frequency response of the
IQ modulator featuring a cutoff at around 80 GHz. We ﬁnd a 3-dB drop between 10 and 70 GHz. c SEM image of a phase shifter in the IQ modulator.
d Optical microscope image of the IQ modulators wire bonded to a PCB. e Measurement setup for the IQ modulator. The PIC was diced and wire-
bonded onto a PCB to contact all DC connections of the IQ modulator. f Fiber-to-ﬁber transmission of the IQ modulator
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 5 of 11

<!-- page 6 -->
O-Band RT modulator
The O-band RT modulator consists of a silicon nitride
bus waveguide with a horizontal directional coupler
(HDC) to a RT containing a plasmonic phase shifter, see
Fig. 4a. Light is coupled from the bus waveguide into the
RT. There is constructive or destructive interference
between the light in the bus waveguide and the light from
the RT. This is dependent on the phase difference accu-
mulated through one round trip. Thus, there are trans-
mission
maxima
and
minima
depending
on
the
wavelength. With a phase shifter, the optical path length
can be modulated within the RT, changing the spectral
position of the minima, respective maxima. Figure 4b
depicts the normalized on-chip transmission as a function
of wavelength for two different voltages. The red and blue
curves have different voltages applied with a ΔV = 2 V.
We ﬁnd ﬁber-to-ﬁber transmission loss of −9.4 dB in the
center of the O-band at 1310–1315 nm. Measurements of
reference waveguides show on-chip device losses of less
than 2 dB. The quality factor of the RT modulator is
measured to be Q = 1931 with an extinction ratio of more
than 6 dB. The free spectral range is measured to be
1.79 nm. By applying a DC voltage to the phase shifter
after fully poling the BTO, we measure a tuning efﬁciency
of 0.3 nm/V. This corresponds to a V π of 3 V at DC. The
frequency response of the modulator in the through-state
is shown in Fig. 4d. Typically, resonant based modulators
suffer from temperature sensitivity. Plasmonic racetrack
modulators, however, have shown an improved tem-
perature sensitivity over silicon microring modulators61.
Furthermore, the combination of plasmonics with BTO
has shown stable modulation up to 110 °C in a MZ con-
ﬁguration53. A similar temperature sensitivity is expected
with this approach due to the small quality factor and
small dimensions of the RT modulator.
Discussion
Performance in data experiments
The high-speed performance of the modulator is tested
with data experiments in the C- and in the O-band for
long- and short-haul applications.
The C-band MZ modulator operates at symbol rates up
to 256 GBd. The measurement setup and DSP chain to
achieve this high performance are described in the
methods section. The bit-error ratio (BER) as a function
of the transmitted symbol rate is shown in Fig. 5a. We
reach a maximum symbol rate of 256 GBd using 2PAM
with a BER of 2.6710−2 with the full DSP consisting of
linear and nonlinear equalization. This BER is below the
soft-decision forward error correction (SD-FEC) limit
with 20% overhead of 4.010−2, see ref. 62. For symbol
rates up to 196 GBd, the BER is below the hard-decision
FEC (HD-FEC) limit of 3.810−3, see ref. 63. The results of
higher order modulation formats are shown in Fig. 5a in
purple and light blue. We achieve a symbol rate of
170 GBd using 4PAM for a BER of 3.75⋅10−2.
a
c
PS
HDC
ER > 6 dB
IL < 2 dB
Q ≈ 1931  
1313
1314
1315
1316
Transmission [dB]
Wavelength [nm]
–10
–5
0
0.3 nmV–1
Δ V = 2 V
Δ V = 0 V
b
–3
0
3
70
50
30
–30
–70
–50
Frequency [GHz]
Modulation [dB]
d
0.5 mm
Racetrack
Schematic racetrack
Device insertion loss
Optical microscope image racetrack
Frequency response
Fig. 4 Characterization of O-band RT modulator. a Schematic of the O-band RT modulator with a 5 µm long plasmonic phase shifter. b On-chip
transmission as a function of wavelength and bias voltage. The modulator features a quality factor Q of 1931 and less than 2 dB on-chip loss. c Optical
microscope image of the fabricated RT. d Normalized modulation response in the upper and lower sidebands of the optical carrier in the on-state as
a function of frequency
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 6 of 11

<!-- page 7 -->
This corresponds to a maximum line rate with the MZ
modulator of 340 Gbit/s. In the case of the 8PAM trans-
mission experiment, 96 GBd is transmitted with a BER of
3.98 × 10−2.
Furthermore, the modulator is tested in a ﬁber-
transmission experiment with 400 m ﬁber length. The
eye diagrams of these experiments are shown in Fig. 5c.
There is only a small degradation when compared against
the back-to-back eye diagrams. In detail, using 2PAM,
256 GBd was transmitted with a BER of 3.1010−2, while
we transmitted 4PAM 160 GBd with a BER of 4.0010−2.
We estimate the capacitance of the modulator to be
~30 fF from the frequency response, which leads to an
energy consumption of ~10 fJ/bit in the MZ modulator.
The experiments are typically conducted with a full DSP
consisting of linear and nonlinear equalization. However,
certain applications require simpler complexity, e.g.
number of multiplications in the data analysis. This is
especially relevant for low-cost and energy-efﬁcient IM/
DD links. We thus further analyze the modulators by
employing only linear equalization in the DSP chain,
speciﬁcally with a timing recovery and a feed-forward
equalization (FFE) with 21 taps in the IM/DD links. We
demonstrate that the BTO-on-SiN modulators can offer
up to 196 GBd even in this simpler setting, see Fig. 5a. For
symbol rates up to 160 GBd, the BER is below the KP4-
FEC limit. Only for higher symbol rates, nonlinear
equalization is necessary to compensate for non-idealities
in the electrical path. To analyze FFE’s performance with
the MZ modulator, the BER of different 2PAM signals is
plotted as a function of the number of taps in the FFE in
Fig. 5b. With an FFE of less than 10 taps, the MZ mod-
ulator can transmit data below the KP4-FEC limit for
symbol rates up to 140 GBd. With the 160 GBd signal, the
KP4-FEC limit is achieved with only 21 taps.
With the more sophisticated C-band IQ modulator we
show 224 GBd with a BER of 3.7910−2 transmission with
a coherent setup, see Methods. Figure 6a shows the data
experiment and the received constellation diagrams of the
C-band IQ modulator. The transmitted 224 GBd in
4QAM with a total line rate of 448 Gbit/s below the SD-
FEC limit marks the highest data rate achieved both with
BTO as active material on any substrate and with any
nonlinear materials on silicon nitride. Similarly to the MZ
modulator, simpliﬁed DSP with linear equalization only
can be employed for symbol rates up to 192 GBd staying
below the SD-FEC limit and up to 160 GBd below the
HD-FEC limit. Full DSP can be used up to 160 GBd to
remain below the KP4-FEC limit.
The resonant RT modulator was operated with 200 GBd
in the O-band with a low 2 dB device insertion loss. The
results of the data transmission experiment are shown in
Fig. 6b. The modulator reaches 200 GBd using 2PAM with
a BER of 3.1010−2. For symbol rates below 180 GBd, the
transmitted signal has a BER below the HD-FEC limit.
The same simpliﬁed DSP as for the MZ modulator can be
employed up to 176 GBd. For rates up to 140 GBd, the
BER is below the HD-FEC limit. Lower data rates in
comparison to the C-band MZ modulator can be
explained by three main reasons. First, the ampliﬁer
Nr. LMS Taps
140
180
220
260
Symbol rate [GBd]
10–2
<10–5
10–1
SD-FEC
HD-FEC
KP4-FEC
a 
20
10–3
BER
2PAM
4PAM
8PAM
Lin. EQ.
Full DSP
10–4
60
100
BER
c  
b  
10–2
10–3
10–4
10–5
10–1
140 GBd
160 GBd
196 GBd
61
81
1
41
256 GBd 2PAM
BER: 3.11.10–2
160 GBd 4PAM
BER: 4.00.10–2
HD-FEC
SD-FEC
21
101
121
C-band Mach-zehnder modulator
Analysis of the linear equalization
Fiber experiment
Fig. 5 Data experiment of MZ modulator a Data experiment results of the MZ modulator. It reaches 256 GBd using 2PAM, 170 GBd using 4PAM,
and 96 GBd using 8PAM, while staying below the SD-FEC limit with 20% overhead. With simpliﬁed DSP consisting of linear equalization only, we
reach 196 GBd below the SD-FEC and 160 GBd below the KP4-FEC limit. b BER as a function of the taps in the LMS ﬁlter. We compare 140 GBd with
the 160 and 196 GBd. Only a small number of taps are required for the 140 GBd signal. Still, a minimum of 3 is necessary to equalize. For the 160 GBd
21 taps allow transmission below the KP4-FEC limit. c Eye diagrams of the 400 m ﬁber transmission using full DSP. In 2PAM, 256 GBd can be
transmitted with a BER of 3.1110−2. In the 4PAM transmission experiment, the 400 m ﬁber showed a larger penalty in comparison to back-to-back
measurements. 160 GBd was successfully transmitted with a BER of 4.0010−2
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 7 of 11

<!-- page 8 -->
employed in the O-band has a higher noise ﬁgure. Second,
the photodetector is optimized for the C-band and has a
lower responsivity in the O-band than those at 1550 nm.
Thirdly, the optical bandwidth of the resonant modulator
is lower than that of the MZ modulator due to the reso-
nant effect. However, this could be solved by optimizing
the design. For instance, resonant plasmonic devices with
an organic electro-optic material operating in the C-band
have already shown operation at 220 GBd employing RT
modulators with bandwidths in excess of 100 GHz and
presumably 200 GHz61. Similarly high numbers may be
anticipated with BTO plasmonics by optimizing the
plasmonic length, the coupling into the RT, and the total
length of the cavity.
We demonstrate a high-speed BTO-on-SiN platform
offering beyond 200 GBd data transmission using IM/DD
in the C- and O-band and coherent communications in
the C-band. All devices were fabricated on the same chip.
The potential of this technology is demonstrated in
multiple modulators. For instance, the C-band Mach-
Zehnder modulator reaches 256 GBd in 2PAM and
170 GBd in 4PAM for a line rate of 340 Gbit/s. This
modulator features a V π = 1.8 V at DC. The high-speed
performance allows the employment of simpliﬁed DSP
consisting only of timing recovery and linear equalization
with 21 taps, while reaching 196 GBd. This allows a
reduction of DSP complexity and therefore a reduction of
energy consumption. We further demonstrate the ﬁrst
BTO-based IQ modulator and the ﬁrst high-speed IQ
modulator on the SiN platform. This modulator reaches
224 GBd 4QAM for a total of 448 Gbit/s line rate. Finally,
the high-speed BTO plasmonic phase shifter can be
employed in an RT modulator. We demonstrate 200 GBd
with an O-band BTO RT modulator featuring a 5 µm long
phase shifter and an on-chip loss of less than 2 dB. The
same simpliﬁed DSP can be employed up to 176 GBd. We
show a tuning efﬁciency of 0.3 nm/V with a quality factor
of 1931. These demonstrations show that the BTO-on-
SiN platform offers a solution for high-speed electro-optic
modulators in short- and long-haul communications.
Materials and methods
Fabrication
The modulators in this work were all fabricated on the
same chip. The SiN waveguides were produced in a
photonic foundry on 8-inch wafers. An oxide cladding
was used to cover the waveguides prior to being prepared
for the wafer-scaled integration of the BTO active mate-
rial. The modulators themselves were fabricated on the
same die in a back-end-of-the-line (BEOL), chip-scale
process on top of the BTO-on-SiN. The BTO was pat-
terned using electron beam lithography. Afterwards, the
material was etched to form the directional couplers and
the BTO waveguides. The metallization to form the
plasmonic slot was conducted with electron beam eva-
poration
and
patterns
were
deﬁned
with
poly-
methylmethacrylate (PMMA). Amorphous silicon was
deposited
using
plasma-enhanced
chemical
vapor
deposition (PECVD). The gratings were etched locally
using inductively coupled plasma reactive ion etching
(ICP-RIE) and the silicon was removed selectively to the
materials underneath without damaging the surface. In
the next step, the chip was coated with SiO2 deposited
using PECVD as a cladding. For the heater structures,
Symbol rate [GBd]
Symbol rate [GBd]
140
180
60
Lin. EQ.
Full DSP
100
140
180
220
SD-FEC
HD-FEC
KP4-FEC
b 
a
SD-FEC
HD-FEC
KP4-FEC
Lin. EQ.
Full DSP
220
10–2
<10–5
10–1
10–3
BER
10–4
10–2
<10–5
10–1
10–3
BER
10–4
O-band racetrack modulator
C-band IQ modulator
Fig. 6 Data experiment of IQ and RT modulator. a Data experiment results of the IQ modulator reaching 224 GBd below the SD-FEC, 192 GBd
below the HD-FEC and 160 GBd below the KP4-FEC limit using 4QAM. With simpliﬁed DSP, data rates up to 192 GBd can be transmitted below the
SD-FEC limit. b Data experiment of the O-band RT modulator. The crossed values represent the full DSP, whereas the dots depict the simpliﬁed DSP.
The RT reaches 200 GBd below the SD-FEC with full DSP and 176 GBd with the linear equalization only
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 8 of 11

<!-- page 9 -->
further metallization steps needed to be added with
crossings between metal layers in addition to SiN and
BTO waveguides. Approximately 1 µm distance between
the individual layers was used to lower loss and parasitic
coupling. Connections between different metal layers in
addition to openings for contacting were made by pho-
tolithography. These structures were etched using ICP-
RIE. Finally, the chip was diced into different parts. The
chip with the IQ modulators was bonded onto a PCB with
a conductive glue and ﬁnally wire bonds were made to
connect the PCB to the IQ modulators with the DC
control voltages.
Characterization
The cutback measurements were conducted using a
tunable laser source in the C-band, respective O-band. An
optical power meter was used to track the ﬁber insertion
losses. Cutback measurements allowed an approximation
of the losses in the individual components. The DC Vπ
was measured by applying a voltage directly from a small
signal source to the modulator and tracking the optical
output. Prior to sweeping the voltage, the BTO was fully
poled. The high-speed electro-optic bandwidth measure-
ments were conducted by applying a sinusoidal signal to
the modulator. This signal was combined using a high-
speed bias-tee. Two different methods to generate the
signal were used. For measurements up to 70 GHz, an RF
source was directly used. The modulation sidebands were
tracked with an optical spectrum analyzer. For the mea-
surements from 70–110 GHz, an RF mixer was used to
increase the frequency of the RF source. An overlap at
70 GHz allowed to match the signal of the generated
signals. For calibration of the setup, the losses in the
electrical path were measured by connecting it to an
electrical spectrum analyzer. The losses of the probes
were
taken
from
the
data
sheet
provided
by
the
manufacturer.
Data transmission experiment
The measurement setups of the data transmission
experiments are shown in Fig. 7. In (a, b), the setup for the
MZ modulator measurement is shown, in (c, d) for the RT
modulator, and in (e, f) for the IQ modulator.
The MZ modulator was measured in an IM/DD setup.
On the transmitter side, see Fig. 7a, a tunable laser in the
C-band is coupled to the chip with ~20.3 dBm input
power to generate an optical carrier at λ = 1550 nm. A
periodically repeated signal is generated with an arbitrary
waveform generator (AWG). These square-root-raised
cosine shaped bit sequences were generated for symbol
rates up to 256 GBd. The signal is combined with a DC
bias using a high-frequency bias-tee. The biasing is used
DSP
DSO
EDFA
PD
400 m
VDC
TLS
AWG
1550 nm
IDC
EDFA
TLS
1550 nm
Optical hybrid
PD + DSO
AWG
SiN chip
I2
I1
I3
TLS
AWG
1310 nm
SiN chip
SiN chip
DSO
PDFA
PD
DSP
Optical
Electrical
a Transmitter MZM
c Transmitter racetrack modulator
e Transmitter IQ modualtor
f
Receiver IQ modulator
d Receiver racetrack modulator
b Receiver MZM
DSP
Bias tee
Bias tee
Bias tee
VDC
VDC
Fig. 7 Measurement setup for data experiments. a, b IM/DD measurement setup for the C-band MZ modulator transmission experiment. c, d IM/
DD measurement setup for the O-band RT modulator. e, f Coherent measurement setup for the C-band data transmission experiment of the IQ
modulator
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 9 of 11

<!-- page 10 -->
to pole the domains of the BTO and achieve best mod-
ulation efﬁciency. This voltage value was adapted to a
value between 2 V and 3 V for all measurements in this
work. On the receiver side, see Fig. 7b, the signal was
ampliﬁed using an erbium-doped ﬁber ampliﬁer (EDFA).
We recorded the signal with a photodetector and a real-
time digital sampling oscilloscope (DSO). We conducted
two experiments with the MZ; one is optical back-to-
back, and in the other the modulated signal is sent
through a 400 m long ﬁber before the EDFA.
The measurement setup for the IQ modulator, see
Fig. 7e, f is optimized for coherent communication. The
MZ modulators were operated in push-pull. The carrier
was set to λ = 1550 nm with ~21 dBm input power to the
chip. On the receiver side, an optical hybrid consisting of
two photodetectors with a local oscillator was employed
after being ampliﬁed with an EDFA. The signal was then
recorded with a DSO.
The measurement setup for the RT modulator, see
Fig. 7c, is similar to the C-band MZ modulator. The laser
is exchanged with a tunable laser in the O-band. The
carrier was set to λ = 1315.7 nm with an input power (in
the ﬁber before the device) of ~13 dBm. The EDFA was
exchanged with a praseodymium-doped ﬁber ampliﬁer
(PDFA). A 70 GHz C-band photodetector was used, for
which we expect a reduced responsivity in the O-band.
Two different types of DSP were employed for all
measurements. The ﬁrst type is linear equalization only.
In the case of the MZ and the RT modulators, this sim-
pliﬁed DSP consists of a timing recovery64 and a feed
forward equalizer (FFE) ﬁlter with 21 taps only. For the IQ
modulator, an additional carrier recovery was conducted
prior to the FFE. The full DSP was adapted for each
modulation format and modulator individually. In the
case of the MZ modulator, the full DSP consisted of a
timing recovery with a T/2-spaced FFE similar to the
simpliﬁed DSP. This FFE featured 151 taps. The nonlinear
equalization was based on a 7-symbol pattern mapping
(MAP). Finally, a second T-spaced FFE with 251 taps was
applied. In the case of the 8PAM modulation format, the
MAP was reduced to 5-symbols and the second FFE was
increased to 1001 taps. In the case of the 4PAM, a third-
order Volterra was added prior to the 7-symbol pattern
mapping. Interestingly, full DSP for the 4PAM signal did
not perform much better than the simpliﬁed DSP. By
increasing the T-spaced FFE to 151 taps, 160 GBd in
4PAM was successfully transmitted below the SD-FEC
limit. The full DSP of the RT modulator was identical to
the DSP of the 2PAM signal with the MZ modulator. In
the case of the IQ modulator, however, the DSP consisted
of a carrier recovery, timing recovery, and a FFE with
251 taps. The nonlinear equalization consisted of a
7-symbol pattern mapping with a second FFE of 251 taps.
For all modulators, an approximate driving voltage of
1.13 Vpp was applied. This value was measured at 200 GBd
with electrical back-to-back measurements. Losses of the
electrical path were subtracted with values from the
datasheets of the individual components at 50 GHz to
include the approximate average losses as seen by the
electrical signal.
Acknowledgements
This work was funded by the EC H2020 projects NEBULA (871658) and
PlasmoniAC (871391). We thank the cleanroom operations team of the Binnig
and Rohrer Nanotechnology Center (BRNC) for their help and support.
Author details
1ETH Zurich, Institute of Electromagnetic Fields, Zurich, Switzerland. 2Ligentec
SA, Ecublens, Switzerland. 3Lumiphase AG, Stäfa, Switzerland
Author contributions
M.K. conceived the concept, designed, developed, and characterized the
devices. J.L. conceived and supervised, and U.K. supervised this research. M.K.,
D.C., J.W., D.M., T.B., Y.H., A.M., Y.F., U.K., and J.L. gave conceptual suggestions.
M.K., D.C., K.K., M.D., T.B., C.C., F.E., and Y.F. were involved in the development of
fabrication methods and the fabrication of devices. The SiN wafers were
fabricated by Ligentec with a lead by T.B. The growth and integration of the
BTO was completed by Lumiphase with a lead by C.C. and F.E. M.K., D.C., J.W.,
D.M., and A.M. contributed to methodology and experiments. Data
experiments were conducted by M.K., L.K. T.B., and Y.H. The original draft was
written by M.K., while all authors were involved in the review and editing
process.
Data availability
The data that support the ﬁndings of this study are available from the
corresponding authors on reasonable request.
Conﬂict of interest
F.E. and C.C. are involved in commercializing the barium titanate photonic
technologies at Lumiphase AG.
Supplementary information The online version contains supplementary
material available at https://doi.org/10.1038/s41377-025-02116-1.
Received: 14 August 2024 Revised: 14 October 2025 Accepted: 2 November
2025
References
1.
Winzer, P. J. & Neilson, D. T. From scaling disparities to integrated parallelism: a
decathlon for a decade. J. Lightwave Technol. 35, 1099–1115 (2017).
2.
PsiQuantum Team A manufacturable platform for photonic quantum com-
puting. Nature 641, 876–883 (2025).
3.
Eltes, F. et al. An integrated optical modulator operating at cryogenic tem-
peratures. Nat. Mater. 19, 1164–1168 (2020).
4.
Lecocq, F. et al. Control and readout of a superconducting qubit using a
photonic link. Nature 591, 575–579 (2021).
5.
Yousseﬁ, A. et al. A cryogenic electro-optic interconnect for superconducting
devices. Nat. Electron. 4, 326–332 (2021).
6.
Bisang, D. et al. Plasmonic modulators in cryogenic environment featuring
bandwidths in excess of 100 GHz and reduced plasmonic losses. ACS Pho-
tonics 11, 2691–2699, https://doi.org/10.1021/acsphotonics.4c00507 (2024).
7.
Lee, B. G. et al. Beyond CPO: a motivation and approach for bringing optics
onto the silicon interposer. J. Lightwave Technol. 41, 1152–1162 (2023).
8.
Marpaung, D., Yao, J. P. & Capmany, J. Integrated microwave photonics. Nat.
Photonics 13, 80–90 (2019).
9.
Burla, M. et al. 500 GHz plasmonic Mach-Zehnder modulator enabling sub-
THz microwave photonics. APL Photonics 4, 056106 (2019).
10.
Ying, Z. F. et al. Electronic-photonic arithmetic logic unit for high-speed
computing. Nat. Commun. 11, 2154 (2020).
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 10 of 11

<!-- page 11 -->
11.
Sun, C. et al. Single-chip microprocessor that communicates directly using
light. Nature 528, 534–538 (2015).
12.
Kohli, M. et al. Plasmonic ferroelectric modulator monolithically integrated on
SiN for 216 GBd data transmission. J. Lightwave Technol. 41, 3825–3831,
https://doi.org/10.1109/JLT.2023.3260064 (2023).
13.
Kulmer, L. et al. Single carrier net 400 Gbit/s IM/DD over 400 m ﬁber enabled
by plasmonic Mach-zehnder modulator. In Proceedings of the 2024 Optical
Fiber Communications Conference and Exhibition (OFC), 1–3 (IEEE, 2024).
14.
Mardoyan, H. et al. First 260-GBd single-carrier coherent transmission over 100
km distance based on novel arbitrary waveform generator and thin-ﬁlm
lithium niobate I/Q modulator. In Proceedings of the 2022 European Conference
on Optical Communication (ECOC), 1–4 (IEEE, 2022).
15.
Kulmer, L. et al. 256 GBd single-carrier transmission over 100km SSMF by a
plasmonic IQ modulator. In Proceedings of the 49th European Conference on
Optical Communications (ECOC 2023), 1449–1452 (IEEE, 2023).
16.
Berikaa, E. et al. Silicon photonic single-segment IQ modulator for net 1 Tbps/λ
transmission using all-electronic equalization. J. Lightwave Technol. 41,
1192–1199 (2023).
17.
Xu, M. Y. et al. Dual-polarization thin-ﬁlm lithium niobate in-phase quadrature
modulators for terabit-per-second transmission. Optica 9, 61–62 (2022).
18.
Wakita, H. et al. 100-GHz-bandwidth InP-based on-board coherent Tx front-
end enabling 2-Tb/s/λ optical transmission. In Proceedings of the 2024 Optical
Fiber Communications Conference and Exhibition (OFC), 1–3 (IEEE, 2024).
19.
Zhou, X., Urata, R. & Liu, H. Beyond 1 Tb/s intra-data center interconnect
technology: IM-DD OR coherent?. J. Lightwave Technol. 38, 475–484 (2020).
20.
Ozolins, O. et al. Optical ampliﬁcation-free 310/256 gbaud OOK, 197/145
gbaud PAM4, and 160/116 gbaud PAM6 EML/DML-based data center links. In
Proceedings of the Optical Fiber Communication Conference (OFC) 2023. (Optica
Publishing Group, 2023) https://doi.org/10.1364/OFC.2023.Th4B.2.
21.
Berikaa, E. et al. TFLN MZMs and next-gen DACs: enabling beyond 400 Gbps
IMDD O-band and C-band transmission. IEEE Photonics Technol. Lett. 35,
850–853 (2023).
22.
St-Arnault, C. et al. Net 1.6 Tbps (4×400Gbps/λ) O-band IM/DD transmission
over 2 km using uncooled DFB lasers on the LAN-WDM grid and Sub-1V drive
TFLN modulators. In Proceedings of the 2024 Optical Fiber Communications
Conference and Exhibition (OFC), 1–3 (IEEE, 2024).
23.
Ostrovskis, A. et al. Heterogenous InP electro-absorption modulator with Si
waveguides for beyond 200 Gbps/λ optical interconnects. In Proceedings of
the 2024 Optical Fiber Communications Conference and Exhibition (OFC), 1–3
(IEEE, 2024), 1-3.
24.
Puckett, M. W. et al. 422 million intrinsic quality factor planar integrated all-
waveguide resonator with sub-MHz linewidth. Nat. Commun. 12, 934 (2021).
25.
Liu, J. Q. et al. High-yield, wafer-scale fabrication of ultralow-loss, dispersion-
engineered silicon nitride photonic circuits. Nat. Commun. 12, 2236 (2021).
26.
Wilmart, Q. et al. A versatile silicon-silicon nitride photonics platform for
enhanced functionalities and applications. Appl. Sci. 9, 255 (2019).
27.
Heideman, R. et al. Large-scale integrated optics using TriPleX waveguide
technology: from UV to IR. In Proceedings of SPIE 7221, Photonics Packaging,
Integration, and Interconnects IX (SPIE, 2009).
28.
Gyger, F. et al. Observation of stimulated Brillouin scattering in silicon nitride
integrated waveguides. Phys. Rev. Lett. 124, 013902 (2020).
29.
Kim, S. et al. Dispersion engineering and frequency comb generation in thin
silicon nitride concentric microresonators. Nat. Commun. 8, 372 (2017).
30.
Raja, A. S. et al. Electrically pumped photonic integrated soliton microcomb.
Nat. Commun. 10, 680 (2019).
31.
Liu, Y. et al. A photonic integrated circuit–based erbium-doped ampliﬁer.
Science 376, 1309–1313 (2022).
32.
Mahmudlu, H. et al. Fully on-chip photonic turnkey quantum source for
entangled qubit/qudit state generation. Nat. Photonics 17, 518–524 (2023).
33.
Gundavarapu, S. et al. Sub-hertz fundamental linewidth photonic integrated
Brillouin laser. Nat. Photonics 13, 60–67 (2019).
34.
Jin, W. et al. Hertz-linewidth semiconductor lasers using CMOS-ready ultra-
high-Q microresonators. Nat. Photonics 15, 346–353 (2021).
35.
Thomaschewski, M. & Bozhevolnyi, S. I. Pockels modulation in integrated
nanophotonics. Appl. Phys. Rev. 9, 021311 (2022).
36.
Zhu, X. R. et al. Twenty-nine million intrinsic Q-factor monolithic micro-
resonators on thin-ﬁlm lithium niobate. Photonics Res. 12, A63–A68 (2024).
37.
Snigirev, V. et al. Ultrafast tunable lasers using lithium niobate integrated
photonics. Nature 615, 411–417 (2023).
38.
Zhang, M. et al. Broadband electro-optic frequency comb generation in a
lithium niobate microring resonator. Nature 568, 373–377 (2019).
39.
Alexander, K. et al. Nanophotonic Pockels modulators on a silicon nitride
platform. Nat. Commun. 9, 3444 (2018).
40.
Zhang, P. et al. High-speed electro-optic modulator based on silicon nitride
loaded lithium niobate on an insulator platform. Opt. Lett. 46, 5986–5989
(2021).
41.
Jiang, Y. H. et al. Monolithic photonic integrated circuit based on silicon nitride
and lithium niobate on insulator hybrid platform. Adv. Photonics Res. 3,
2200121 (2022).
42.
Ruan, Z. L. et al. High-performance electro-optic modulator on silicon nitride
platform with heterogeneous integration of lithium niobate. Laser Photonics
Rev. 17, 2200327 (2023).
43.
Vanackere, T. et al. Heterogeneous integration of a high-speed lithium niobate
modulator on silicon nitride using micro-transfer printing. APL Photonics 8,
086102 (2023).
44.
Valdez, F., Mere, V. & Mookherjea, S. 100 GHz bandwidth, 1 volt integrated
electro-optic Mach–Zehnder modulator at near-IR wavelengths. Optica 10,
578 (2023).
45.
Churaev, M. et al. A heterogeneously integrated lithium niobate-on-silicon
nitride photonic platform. Nat. Commun. 14, 3499 (2023).
46.
Ortmann, J. E. et al. Ultra-low-power tuning in hybrid barium Titanate–silicon
nitride electro-optic devices on silicon. ACS Photonics 6, 2677–2684 (2019).
47.
Abel, S. et al. Large Pockels effect in micro- and nanostructured barium tita-
nate integrated on silicon. Nat. Mater. 18, 42–47 (2019).
48.
Winiger, J. et al. PLD epitaxial thin-ﬁlm BaTiO3 on MgO −dielectric and
electro-optic properties. Adv. Mater. Interfaces 11, 2300665 (2024).
49.
Dong, Z. M. et al. Monolithic barium Titanate modulators on silicon-on-
insulator substrates. ACS Photonics 10, 4367–4376 (2023).
50.
Posadas, A. B. et al. RF-sputtered Z-cut electro-optic barium Titanate mod-
ulator on silicon photonic platform. J. Appl. Phys. 134, 073101 (2023).
51.
Zgonik, M. et al. Dielectric, elastic, piezoelectric, electro-optic, and elasto-optic
tensors of BaTiO3 crystals. Phys. Rev. B 50, 5941–5949 (1994).
52.
Eltes, F. et al. A BaTiO3-based electro-optic pockels modulator monolithically
integrated on an advanced silicon photonics platform. J. Lightwave Technol.
37, 1456–1462 (2019).
53.
Messner, A. et al. Plasmonic ferroelectric modulators. J. Lightwave Technol. 37,
281–290 (2019).
54.
Kohli, M. et al. 256 GBd barium-titanate-on-SiN mach-zehnder modulator. In
Proceedings of the 2024 Optical Fiber Communications Conference and Exhibi-
tion (OFC), 1–3 (IEEE, 2024).
55.
Kohli, M. et al. Barium Titanate racetrack modulator on silicon nitride for 200
GBd data communication in the O-band. In Proceedings of the 2024 Conference
on Lasers and Electro-Optics (CLEO), 1–3 (IEEE, 2024).
56.
Kohli, M. et al. C- and O-band dual-polarization ﬁber-to-chip grating couplers
for silicon nitride photonics. ACS Photonics 10, 3366–3373 (2023).
57.
Dong, P. et al. Thermally tunable silicon racetrack resonators with ultralow
tuning power. Opt. Express 18, 20298–20304 (2010).
58.
Pfeiﬂe, J. et al. Silicon-organic hybrid phase shifter based on a slot waveguide
with a liquid-crystal cladding. Opt. Express 20, 15359–15376 (2012).
59.
Hattori, A. et al. Integrated visible-light polarization rotators and splitters for
atomic quantum systems. Opt. Lett. 49, 1794–1797 (2024).
60.
van Iseghem, L. et al. Low power optical phase shifter using liquid crystal
actuation on a silicon photonics platform. Optical Mater. Express 12, 2181–2198
(2022).
61.
Eppenberger, M. et al. Resonant plasmonic micro-racetrack modulators with
high bandwidth and high temperature tolerance. Nat. Photonics 17, 360–367
(2023).
62.
Schuh, K. et al. Single carrier 1.2 Tbit/s transmission over 300 km with PM-64
QAM at 100 GBaud. In Proceedings of the 2017 Optical Fiber Communications
Conference and Exhibition (OFC), 1–3 (IEEE, 2017).
63.
El-Fiky, E. et al. First demonstration of a 400 Gb/s 4λ CWDM TOSA for data-
center optical interconnects. Opt. Express 26, 19742–19749 (2018).
64.
Josten, A. et al. Modiﬁed Godard timing recovery for Non Integer over-
sampling receivers. Appl. Sci. 7, 655 (2017).
Kohli et al. Light: Science & Applications (2025) 14:399 
Page 11 of 11

