---
paper_id: wang2024b
source_url: https://doi.org/10.1038/s41586-024-07369-1
doi: 10.1038/s41586-024-07369-1
license: https://creativecommons.org/licenses/by/4.0
sha256: fee9b52d9b931a710ed5876df00b5d57e1fdd6322280d50d4f443741c4d44afa
pages: 15
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
784  |  Nature  |  Vol 629  |  23 May 2024
Article
Lithium tantalate photonic integrated 
circuits for volume manufacturing
Chengli Wang1,2,3,4, Zihan Li2,3,4, Johann Riemensberger2,3, Grigory Lihachev2,3, 
Mikhail Churaev2,3, Wil Kao2,3, Xinru Ji2,3, Junyin Zhang2,3, Terence Blesin2,3, Alisa Davydova2,3, 
Yang Chen1, Kai Huang1, Xi Wang1, Xin Ou1 ✉ & Tobias J. Kippenberg2,3 ✉
Electro-optical photonic integrated circuits (PICs) based on lithium niobate (LiNbO3) 
have demonstrated the vast capabilities of materials with a high Pockels coefficient1,2. 
They enable linear and high-speed modulators operating at complementary metal–
oxide–semiconductor voltage levels3 to be used in applications including data-centre 
communications4, high-performance computing and photonic accelerators for AI5. 
However, industrial use of this technology is hindered by the high cost per wafer and 
the limited wafer size. The high cost results from the lack of existing high-volume 
applications in other domains of the sort that accelerated the adoption of silicon-on- 
insulator (SOI) photonics, which was driven by vast investment in microelectronics. 
Here we report low-loss PICs made of lithium tantalate (LiTaO3), a material that has 
already been adopted commercially for 5G radiofrequency filters6 and therefore 
enables scalable manufacturing at low cost, and it has equal, and in some cases 
superior, properties to LiNbO3. We show that LiTaO3 can be etched to create low-loss 
(5.6 dB m−1) PICs using a deep ultraviolet (DUV) stepper-based manufacturing 
process7. We demonstrate a LiTaO3 Mach–Zehnder modulator (MZM) with a half-wave 
voltage–length product of 1.9 V cm and an electro-optic bandwidth of up to 40 GHz.  
In comparison with LiNbO3, LiTaO3 exhibits a much lower birefringence, enabling 
high-density circuits and broadband operation over all telecommunication bands. 
Moreover, the platform supports the generation of soliton microcombs. Our work 
paves the way for the scalable manufacture of low-cost and large-volume 
next-generation electro-optical PICs.
Next-generation ultrahigh-speed PICs based on electro-optical mate-
rials are poised to play a role in energy-efficient data centres, optical 
communications, 5G and 6G radiofrequency filters and in particu-
lar in AI workload-driven high-performance computing, provided 
that scalable low-cost manufacturing becomes possible. In the past 
two decades, PICs based on silicon (silicon photonics) have rapidly 
transitioned from academic research to widespread use in telecom-
munications8 and data centres9. One crucial factor driving the com-
mercial feasibility of this technological revolution is the high-volume 
availability and cost-effectiveness of SOI wafers. These SOI wafers, 
prepared using smart-cut (ion slicing) techniques10, enable the man-
ufacture of silicon photonics but crucially are more widely used in 
consumer microelectronics. Globally, more than 3 million SOI wafers 
are produced each year, with the wafer diameter being as large as 
300 mm8. Using a similar technique, LiNbO3 has been fabricated into 
lithium niobate-on-insulator (LNOI) structures, offering an entirely 
new class of ultrahigh-speed, low-voltage electro-optical PICs3,11,12 that 
can become key components in future energy-efficient communi-
cation systems. Despite the tremendous scientific progress and the 
increased application range of LiNbO3 PICs, the path to commerciali-
zation remains difficult. Unlike SOI technology, LNOI lacks a larger 
volume of consumer electronics driving its demand, resulting in eco-
nomic limitations to its commercialization. By contrast, another fer-
roelectric material, LiTaO3, which has similar structural properties to 
LiNbO3, has entered the large-volume production stage, driven by its 
applications in 5G filters13,14, and is projected to achieve a production 
capacity of 750,000 lithium tantalate-on-insulator (LTOI) wafers a year 
by 2024 (ref. 15). This substantial volume enables considerable benefits 
in terms of low-cost production when adopting LTOI as a platform for 
PICs, but PICs based on this material have not been reported to date. 
LiTaO3, as well as having a large production volume, exhibits compa-
rable, or in some cases superior, properties to LiNbO3. LiTaO3 is an 
oxygen octahedral ferroelectric crystal with a crystal structure that is 
nearly identical to that of LiNbO3, replacing Nb atoms with the heavier 
Ta atoms. This change gives LiTaO3 not only a greater mass density 
but also stronger chemical bonds, resulting in increased strength and 
chemical stability16. The optical bandgap of LiTaO3 (3.93 eV) is larger 
than that of LiNbO3 (3.78 eV)17–19, enabling nonlinear optical conversion 
https://doi.org/10.1038/s41586-024-07369-1
Received: 26 June 2023
Accepted: 2 April 2024
Published online: 8 May 2024
Open access
 Check for updates
1National Key Laboratory of Materials for Integrated Circuits, Shanghai Institute of Microsystem and Information Technology, Chinese Academy of Sciences, Shanghai, China. 2Institute of 
Physics, Swiss Federal Institute of Technology Lausanne, EPFL, Lausanne, Switzerland. 3Center of Quantum Science and Engineering, EPFL, Lausanne, Switzerland. 4These authors contributed 
equally: Chengli Wang, Zihan Li. ✉e-mail: ouxin@mail.sim.ac.cn; tobias.kippenberg@epfl.ch

<!-- page 2 -->
Nature  |  Vol 629  |  23 May 2024  |  785
to the visible and even ultraviolet20-wavelength range. Furthermore, 
the material exhibits a greatly decreased optical anisotropy, that 
is, the magnitude of the optical birefringence is reduced more than 
10-fold compared with LiNbO3, which suppresses mode mixing, as can 
occur in tight waveguide bends. Moreover, LiTaO3 features a similar 
Pockels coefficient (r33 = 30.5 pm V−1) to the well-established LiNbO3 
(r33 = 30.9 pm V−1) with a moderately larger electrical permittivity, 
ϵ33 = 43, implying that the modulation efficiency of the two materials 
is expected to be almost identical. Furthermore, LiTaO3 benefits from 
a larger optical damage threshold, which is relevant for high-power 
applications. Of particular relevance for applications in the realm of 
microwave-optical quantum transduction21,22, the nearly 10-fold-lower 
microwave loss tangent of LiTaO3 (refs. 23,24) is a promising avenue to 
improve device performance to unity conversion efficiency, which has 
so far eluded efforts using LiNbO3 owing to the limited quality factors 
of microwave resonators22. Historically, despite the beneficial optical 
material properties, the use of LiTaO3 for optical modulators in optical 
communication networks has been limited. One of the reasons is that 
the Curie temperature of LiTaO3 (610–700 °C, depending on the Li:Ta 
ratio) is much lower than the temperature needed for the fabrication 
of optical waveguides by ion diffusion (typically more than 1,000 °C), 
which compounded the use of LiTaO3 for bulk modulators on the basis 
of the ion diffused waveguide25. For this reason, legacy bulk modulator 
technology has used LiNbO3. However, the commercial use of LTOI in 
wireless applications, owing to its suitable acoustic properties, com-
bined with the above optical properties, makes it an ideal platform 
for scalable manufactured electro-optical PICs, although such a use 
has never been demonstrated or pursued. Although free-standing 
‘whispering gallery’ mode resonators have been fabricated from LiTaO3 
single crystals26, as a result of femtosecond laser direct writing27 or 
focused ion beam milling28, scalable manufactured PICs have remained 
an outstanding challenge.
Here, we overcome this challenge and implement what is to our 
knowledge the first PIC platform using LTOI based on direct etching7, 
and demonstrate ultralow optical loss, electro-optical tuning, switching 
through the Pockels effect and soliton-microcomb generation through 
the optical Kerr effect of LiTaO3. We achieve this by transferring the 
diamond-like carbon (DLC)-based masking etching process, originally 
developed for LiNbO3, to LiTaO3, and propose a new solution to remove 
LiTaO3 redeposition, which highlights the flexibility of our process for 
the fabrication of a variety of ferroelectric photonics platforms. We 
also demonstrate a DUV approach to electrode manufacturing. Taken 
together, our work establishes a basis for scalable volume manufactur-
ing of ultrahigh-speed electro-optical PICs.
LiTaO3 PICs
The fabrication process for LTOI wafers and optical waveguides is 
depicted in Fig. 1 (details in Methods). The LTOI wafer was fabricated 
by the smart-cut technique14. The process flow is schematically illus-
trated in Fig. 1a. In contrast to the well-established LNOI preparation 
process, which uses helium-ion implantation with an energy greater 
than 200 keV29, the fabrication of LTOI favours hydrogen ions with an 
implanted energy of 100 keV and a beam current ten times higher, as 
found in most commercial ion implanters, thereby simplifying the 
wafer production. The fabrication recipes of LTOI are more closely 
x (mm)
y (mm)
Thickness (nm)
DLC deposition
DLC etching
IBE
Redeposition cleaning
and mask stripping
b
c
f
CH4 →a–C + 2H2
Ion implantation
Wafer bonding
Splitting
CMP
a
C + 2O →CO2     
Ar+
g
2.4 nm
Height
d
e
LiTaO3
SiO2
x
DLC
LiTaO3
SiO2
Si
h
i
j
–40
–20
–40
–20
0
20
40
600
605
610
615
620
625
630
20
40
0
Fig. 1 | LTOI substrates and optical waveguides. a, Schematic of the LTOI 
wafer-bonding workflow showing hydrogen-ion implantation, bonding, 
splitting and chemical mechanical polishing (CMP). b, Photograph of the 
bonded wafer demonstrating uniform and defect-free bonding. c, Thickness 
map of the LiTaO3 thin film on the wafer. The x,y axes represent the distance 
from the wafer centre. d, Atomic force micrograph of the LiTaO3 thin film 
surface. Scale bar, 500 nm. e, High-resolution scanning transmission 
electron-microscopy image of the LiTaO3–SiO2 bonding interface. The arrow 
represents the x-cut crystal orientation. Scale bar, 2 nm. f, Schematic of the 
fabrication workflow for LTOI optical waveguides, including DLC hard-mask 
deposition by plasma-enhanced chemical vapour deposition (PECVD) from the 
methane precursor, DLC dry etching through oxygen plasma, and LiTaO3 
etching by argon ion-beam etching (IBE), followed by redeposition and mask 
removal. The layers are DLC (black), LiTaO3 (light blue), SiO2 (purple) and Si 
(grey). Spheres show C (black), O (red) and Ar+ (green). g, Colourized scanning 
electron micrograph (SEM) of LTOI microring resonator (blue). Scale bar, 
50 μm. h, Colourized SEM of etched LTOI microring and bus waveguide 
coupling section. Scale bar, 2 μm. i, Colourized SEM of etched LTOI waveguide 
and sidewall. Scale bar, 2 μm. j, Colourized SEM cross-section of etched LTOI 
waveguide (blue) on top of SiO2 bottom cladding (purple). Scale bar, 500 nm.

<!-- page 3 -->
786  |  Nature  |  Vol 629  |  23 May 2024
Article
aligned with the high-volume commercial production of SOI wafers, 
resulting in higher efficiency and lower costs in the production of LTOI 
than of LNOI. The fabricated LTOI wafer has a 4-inch (102 mm) size with 
a surface roughness of 0.25 nm and a non-uniformity of less than 30 nm 
(Fig. 1b–d). The crystallinity of LiTaO3 and the LiTaO3–SiO2 interface 
remain of high quality after the completion of the production process, 
and so does the sharpness of the bonding interface, as can be seen 
in the high-resolution scanning transmission electron microscopy 
image (Fig. 1e). Photonic building blocks such as optical ring resonators 
(Fig. 1g), racetrack resonators and waveguide spirals are also fabricated. 
The lithography, dry etching and by-product cleaning processes were 
optimized to achieve both favourable coupling regions (Fig. 1h) and 
well-defined, smooth sidewalls (Fig. 1i) of the LTOI PICs. As detailed in 
Methods, the removal of non-volatile by-products for LTOI requires a 
different chemical than for LNOI7. The cleaved cross-section featur-
ing steep sidewall angles of almost 70° with respect to the surface is 
shown in Fig. 1j.
Next, we characterized the LiTaO3 PICs  (D101_LT_A2) using 
frequency-comb calibrated tunable diode laser spectroscopy30 to deter-
mine the optical loss and absorption of optical microresonators with a 
waveguide width of 2.0 μm across the 4-inch wafers (Fig. 2a). We find 
mean intrinsic loss rates, κ0/2π, of between 35.2 MHz and 72.9 MHz with 
eight of the nine fields performing better than 50.8 MHz. The microres-
onator intrinsic loss rate κ0/2π = 35.2 MHz corresponds to a propagation 
loss, α, of 7.3 dB m–1 for the unreduced LiTaO3 wafer that is used for 
optical applications. We also characterized the optical loss of the LTOI 
platform fabricated from the reduced-LiTaO3 bulk wafers, known in 
the filter industry as acoustic grade or black LiTaO3, which undergo 
an extra chemical reduction step, typically by annealing in carbon 
powder to minimize the pyroelectric effect31 (Methods and Extended 
Data Fig. 1). The LTOI fabricated from the reduced wafer exhibits an 
intrinsic loss rate κ0/2π = 42 MHz in the best field and a mean value 
of κ0/2π = 82 MHz across the whole wafer. This corresponds to losses 
α = 8.8 dB m−1 and α = 17.1 dB m−1, which is below the published losses 
of the wafer-scale fabrication of LNOI PICs32, with an average loss of 
27 dB m−1, making our DUV-based process directly applicable to widely 
used mass-manufactured LTOI wafer substrates. An optical-resonance 
transmission spectrum (D101_LT_A2_F1_C4_02_WG4) and fit is shown 
in Fig. 2b, which indicates an intrinsic loss rate of κ0/2π = 26.8 MHz; 
this corresponds to a propagation loss of α = 5.6 dB m−1 for unreduced 
LiTaO3. We also fabricated optical-waveguide spirals with a waveguide 
cross-section of 1.75 μm × 0.6 μm and found a propagation loss of 
around 9 dB m−1 (Extended Data Fig. 2). A histogram of fitted intrinsic 
loss rates for the microresonator is shown in Fig. 2c. The contributions 
of optical absorption and scattering from bulk and sidewall imperfec-
tions can be separated by thermal response spectroscopy33 (Fig. 2d). An 
intensity-modulated pump laser was tuned to the centre of the optical 
resonance, and the frequency-modulation response of the optical 
0
30
60
120
150
180
90
Angle T (degrees) 
1.95
2.00
2.05
Effective index
104
103
106
105
107
108
Modulation frequency (Hz)
104
102
0
10–2
Response | F/FKerr (0)|2
Probe O = 1,545 nm 
Nabs/2S = 2.0 MHz
Frequency (THz)
180
190
200
210
220
230
240
Frequency (THz)
180
190
200
210
220
230
240
Dispersion ﬁt
Dispersion ﬁt
0
–5
5
10
15
Dint/2S (GHz)
0
–5
5
10
15
Dint/2S (GHz)
0
20
40
60
80
100
Frequency (MHz)
0
20
40
60
80
100
120
Occurrence
N0/2S
 = 26.8 MHz
–500
0
500
Frequency (MHz)
TE00
TM00
0
0.5
1.0
Normalized transmission
39.3 MHz 38.5 MHz
48.8 MHz
43.6 MHz 41.2 MHz
50.8 MHz
72.9 MHz 48.2 MHz
35.2 MHz
y
z
a
h
i
f
e
g
b
c
d
T
T
z
y
eo
o
o
LiTaO3 TE00
LiNbO3 TE00
LiTaO3 TM00
LiNbO3 TM00
E 
O 
S 
C
L
E 
O 
S 
C
L
eo
eo
eo
o
Fig. 2 | Optical characterization of LTOI PICs. a, Wafer-scale map of mean 
intrinsic loss, κ0/2π, for similar resonators. b, Normalized resonance 
transmission spectrum of an optical racetrack microresonator at 209.358 THz. 
c, Statistical distribution of intrinsic loss, κ0/2π, of the optical racetrack 
microresonator. Mean = 40.4 MHz, median = 36.4 MHz. d, Nonlinear optical- 
response measurement (solid red line) and fit (solid black line) of the thermo- 
optical (red dashed line) and Kerr (blue dashed line) nonlinear responses of  
the optical microresonator, demonstrating ultralow optical absorption loss.  
e, Illustration of the LiNbO3 (red) strongly negative uniaxial and LiTaO3 (blue) 
weakly positive uniaxial crystal birefringence. For LiTaO3, ordinary refractive 
index nLT.o = 2.119, extraordinary refractive index nLT.e = 2.123 and the difference 
ΔnLT = 0.004; for LiNbO3, ordinary refractive index nLN.o = 2.21, extraordinary 
refractive index nLN.e = 2.14 and the difference ΔnLN = −0.07. f, Illustration of the 
curve angle, θ, and fundamental transverse electric (TE00) and transverse 
magnetic (TM00) mode profiles in LTOI. Scale bar, 1 μm. g, Numerical simulation 
of fundamental TE00 and TM00 optical-mode effective refractive indices of LNOI 
(red) and LTOI (blue) as a function of the angle between the waveguide and the y 
axis of the x-cut LNOI or LTOI film. The reduced birefringence of LTOI precludes 
unwanted birefringent mixing between fundamental TE00 or TM00 modes in 
thick waveguides. Arrows indicate birefringent mode mixing. h, Dispersion 
profile of an LTOI racetrack microresonator with the waveguide cross-section 
2 μm × 0.5 μm and a slab 100 nm thick. D1/2π = 82.234 GHz, D2/2π = 197.88 kHz. 
L, C, S, E and O telecommunication bands are marked with vertical dashed lines. 
i, Dispersion profile of an LNOI racetrack microresonator with similar 
cross-section and strong mode mixing at frequencies above 215 THz, which 
occupies the E-band and the O-band in the optical communication. 
D1/2π = 80.83 GHz, D2/2π = 105.72 kHz.

<!-- page 4 -->
Nature  |  Vol 629  |  23 May 2024  |  787
microresonator resulting from the thermo-optical and Kerr effects was 
read out with a second laser tuned to the side of another resonance. We 
modelled the frequency dependence of the thermal effect arising from 
the optical absorption and the optical Kerr effect using finite-element 
simulations and fitted the combined response33,34. We found that the 
absorption limit of our LTOI microresonator is κabs/2π = 2.0 MHz, cor-
responding to an absorption-limited propagation loss of α = 0.4 dB m−1, 
which is close to recent results obtained for LNOI34. Therefore, the 
main source of loss in our tightly confining LiTaO3 waveguides is scat-
tering losses.
The optical birefringence of LiTaO3 is more than one order of mag-
nitude smaller than that of LiNbO3 (Fig. 2e) and therefore enables 
the fabrication of thick waveguides without incurring mode mixing 
between the fundamental modes in waveguide bends7,35. Mode mixing 
occurs in x-cut LiNbO3 waveguide bends when the TE mode transitions 
from the extraordinary (eo) to the ordinary (o) axes above a critical 
LiNbO3 thickness that at a wavelength of 1.55 μm lies at around 700 nm 
and at a wavelength of 1.3 μm lies at around 600 nm, largely indepen-
dently of the slab thickness or waveguide width. In contrast, the low 
and positive uniaxial birefringence of LiTaO3 precludes mode mixing 
in x-cut waveguides with a horizontal-to-vertical aspect ratio greater 
than one. We simulate the effective mode indices of the fundamental 
polarization modes of LNOI and LTOI for a waveguide thickness of 
600 nm, a waveguide width of 2 μm and a wavelength of 1.55 μm as 
a function of the angle between the propagation and the eo crystal 
axes (Fig. 2f,g). For LiNbO3, we found a crossing of the fundamental TE 
and TM modes at an angle of 25°, whereas no mode crossing is found 
for an LTOI waveguide with the same dimension. This observation is 
in excellent agreement with the results from our optical dispersion 
measurement, Dint = ωμ − ω0 − (D1 × μ), where μ indicates the azimuthal 
mode index for the mode with frequency ωμ, D1 is a free spectral range, 
Dint is an integrated dispersion and ω0/2π = 205 THz, for the LTOI (D101_
LT_A2_F1_C4_WG4) and LNOI (D133_02_F2_C5_01_WG4) waveguides, 
which are depicted in Fig. 2h. The optical microresonators have similar 
anomalous dispersion, but the dispersion profile of the LTOI micro-
resonator remains smooth over the full measurement span of 185 THz 
to 240 THz, whereas the LNOI microresonator exhibits striking mode 
mixing at frequencies above 215 THz. The birefringence complicates the 
design of compact PICs and is useful only in some special cases, such as 
birefringence phase matching36. Adjustments to the waveguide geom-
etry and working wavelength can weaken the mode mixing caused by 
strong birefringence in LNOI7,37, but such adjustments result in reduced 
optical confinement and chip compactness. By contrast, LTOI offers 
much lower birefringence, thereby providing greater flexibility in 
waveguide design and manufacturing, and mode-mixing-free opera-
tion over all telecommunications bands from 1,260 nm to 1,625 nm, 
ranging from the O to the L band. Furthermore, the resonance shift 
induced by the photorefractive effect in an LTOI microresonator is 
nearly fivefold smaller than that in an LNOI microresonator (Extended 
Data Fig. 3 and Methods), which is consistent with the results obtained 
from bulk crystals38.
Electro-optical modulation
To demonstrate the utility of the LTOI platform for electro-optics, 
we created a tunable high Q-factor microresonator. The resonator 
0
10
20
30
40
DC offset (V)
0
2
4
6
8
10
Resonance shift (GHz)
a
b
c
e
AFG
PM
20 MHz
ECDL
PD
OSC
z
255 MHz V–1
d
0
2
4
6
8
Frequency detuning (GHz)
0.4
0.6
0.8
1.0
1.2
Transmission
z
y
0
10
20
30
40
50
Frequency (GHz)
–10
–5
0
Electro-optic S21 (dB)
0
5
10
Voltage (V)
0
0.2
0.4
0.6
0.8
1.0
1.2
Normalized transmission (Arb. units)
VS = 6.4 V
VS L = 1.6 V cm
VS = 7.6 V
VS L = 1.9 V cm
1,310 nm
1,550 nm
–3 dB
–6 dB
f
g
Fig. 3 | Electro-optical tuning and switching in LTOI. a, Colourized SEM of 
LTOI (blue) racetrack optical microresonator with gold electrodes (yellow). 
Scale bar, 300 μm. b, Colourized SEM of pulley resonator and bus waveguide 
coupling section. Scale bar, 5 μm. c, Measured resonance shift as a result of 
tuning voltage. The linear fit indicates a voltage tuning response of 255 MHz V–1. 
Left inset: schematic of the measurement set-up for microresonator tuning 
measurement with phase-modulation (PM) sideband calibration. ECDL, external 
cavity diode laser; AFG, arbitrary frequency generator; OSC, oscilloscope; PD, 
fast photodiode. Right inset: normalized cavity transmission measurement 
showing electro-optical tuning of an LTOI resonance. Each colour step 
corresponds to an increase in DC tuning voltage of 5 V. d, Optical micrograph of 
2.5 mm-long MZM. Scale bar, 200 μm. e, Colourized SEM of MZM waveguides 
and electrodes. Scale bar, 10 μm. f, Electro-optical bandwidth (S21 parameter, 
measured as the power ratio) of MZM for a device length of 2.5 mm at a 
wavelength of 1,550 nm. g, Normalized optical transmission as a function of 
applied voltage on travelling-wave electrodes at wavelengths of 1,310 nm and 
1,550 nm, showing a voltage–length product,VπL, of 1.6 V cm at the O-band and 
1.9 V cm at the C-band.

<!-- page 5 -->
788  |  Nature  |  Vol 629  |  23 May 2024
Article
has a racetrack design with an apex radius of 100 μm and a straight 
section length of 400 μm (Fig. 3a) with a uniform waveguide width 
of 2 μm and pulley-style coupling sections (Fig. 3b). Metal electrodes 
were fabricated using a DUV-lithography-based lift-off process that 
allowed us to manufacture electrodes with an alignment tolerance 
of less than 100 nm to the optical waveguide (Methods). We applied 
a voltage across two of the four electrodes to measure the voltage 
tuning coefficient and measure the resonance position using an 
external-cavity diode laser (ECDL) (Fig. 3c and Methods). We found 
a voltage tuning efficiency of 255 MHz V–1 using a single electrode pair, 
which corresponds to 510 MHz V–1 if both phase-shifter sections are 
modulated. We also fabricated a travelling-wave MZM composed of 
two 50:50 adiabatic Y-splitters at either end and a push–pull optical 
waveguide phase-shifter pair with a length of 2.5 mm (Fig. 3d). The 
waveguide width was 1.2 μm and the gap between the LiTaO3 waveguide 
sidewalls and the gold electrode was 2.5 μm on each side (Fig. 3e). The 
etching depth was 220 nm, leaving a 400 nm slab. This maintained a 
consistent group refractive index, ng, of around 2.25 for both microwave 
and optics waves. The transmission through the MZM with a 10 kHz 
triangular voltage sweep is plotted in Fig. 3g. The MZM can work at 
two communication bands of 1,310 nm and 1,550 nm simultaneously, 
owing to the use of the broadband adiabatic Y-splitter. The measured 
Vπ was 6.4 V for 1,310 nm and 7.6 V for 1,550 nm, corresponding to a 
VπL of 1.6 V cm and 1.9 V cm, respectively. The difference in VπL arises 
mainly from the overlap difference between the optical modes and the 
electric field at distinct wavelengths and increased optical frequency. 
The measured VπL of 1.9 V cm is similar to the state-of-the-art results 
for LNOI at 1,550 nm1,3, with similar electrode structures, as expected 
given that LiNbO3 and LiTaO3 have almost identical Pockels coefficients 
(Extended Data Table 1).
We then characterized the small-signal electro-optic bandwidth of 
the fabricated devices (Methods). The measured 3 dB electro-optic 
bandwidth was more than 40 GHz (Fig. 3(f)).
Soliton-microcomb generation
Finally, we investigated the LiTaO3 platform for soliton-microcomb 
generation. The strong optical confinement, high Q-factor, anomalous 
dispersion and substantial Kerr nonlinearity of our LTOI microresona-
tors make them naturally suitable for dissipative Kerr soliton (DKS) 
generation39,40. However, LiTaO3 is recognized as a Raman-active crystal-
line material, displaying multiple robust vibrational phonon branches 
in various polarization configurations41, which can have a detrimental 
effect on soliton generation. This Raman interference presents a com-
mon challenge when attempting to generate solitons in ferroelectric 
crystal platforms. For instance, despite extensive research efforts, 
achieving solitons in the x-cut configuration of LNOI has remained 
elusive1,42. It is well known that the Raman effect is polarization depend-
ent, typically exhibiting maximum strength when the pump light is 
polarized along the polar axis of the crystal1,17. We investigated such 
a polarization-dependent Raman effect in both x-cut LNOI and LTOI 
(Methods). A reduction in Raman intensity was achieved when the 
polarization of incident light transitions from being parallel to the 
y axis to being parallel to the z axis (the rotation angle θ changes from 
–70
–60
–50
–40
–30
Power (dBmW)
a
b
e
c
f
0°
90°
180°
270°
0°
90°
180°
270°
180
185
190
195
200
205
Frequency (THz)
–70
–60
–50
–40
–30
Power (dBmW)
1,650
1,600
1,550
1,500
Wavelength (nm)
0
0.2
0.4
0.6
0.8
1.0
Time (μs)
0
0.5
1.0
Ipd (Arb. units)
101
102
103
104
105
106
Frequency (Hz)
Sff (dBc Hz–1)
–100
–50
0
–200 –100
0
100 200
Frequency at 30.1 GHz (kHz)
–160
–120
–80
RF power (dBmW)
180
185
190
195
200
205
Frequency (THz)
–70
–60
–50
–40
–30
Power (dBmW)
 Raman shift
PD
EDFA
SSB
VCO
CW pump
OSA
ESA
PD
OSC
TE
z
y
FBG
180
185
190
195
200
205
Frequency (THz)
0
20
40
60
80
100
0.6
0.7
0.8
0.9
1.0
Raman intensity
(Arb. units)
z
y
z
y
z
y
T
Polarization
FPC
T = 0º
T = 90º
Rotation angle (degrees)
d
Soliton step
Fig. 4 | DKS generation in LTOI microresonators. a, Optical spectrum of a 
single soliton microcomb featuring a sech2-spectral profile with a 3 dB 
bandwidth of 4.9 THz, corresponding to an FWHM pulse duration of 63 fs at a 
pulse repetition rate, frep, of 81 GHz. Inset, the light generated during the rapid 
laser scan measured by filtering out the pump light. The soliton step is marked 
with light blue shading. Ipd, photodiode current. b, Optical spectrum of a single 
soliton with a repetition rate of 30.1 GHz; ΔTFWHM = 71 fs. Inset, the relative phase 
position inside the microresonator. c, Optical setup for soliton generation in 
x-cut LTOI microresonators. The orientation and TE polarization are also 
indicated in the schematic diagram of the fabricated LTOI chip. Rapid laser scans 
were generated using a single-sideband modulator (SSB) and voltage-controlled 
oscillator (VCO), continuous wave (CW) laser and an erbium-doped fibre 
amplifier (EDFA). FBG, fibre Bragg grating; FPC, fibre polarization controller. 
The soliton microcombs were analysed using an optical spectrum analyser (OSA) 
and the nonlinearly generated light and microwave beat notes were recorded 
with a PD and analysed with an OSC and electrical spectrum analyser (ESA), 
respectively. d, Variation of the Raman intensity with different LTOI crystal 
rotation angles. Inset, the angle between the orientation (z axis) and the 
polarization of the excitation laser. e, Optical spectrum of a three-soliton state 
with a repetition rate of 30.1 GHz. Inset, three solitons inside the microresonator. 
ΔTFWHM = 52 fs. f, Single-side band-phase noise-power spectral density (Sff) of a 
30.1 GHz microwave beat note generated from the multisoliton state in e. Inset, 
spectrum of a microwave beat note with a resolution bandwidth of 30 Hz.

<!-- page 6 -->
Nature  |  Vol 629  |  23 May 2024  |  789
0° to 90°; Fig. 4d). We therefore used racetrack microresonators with 
the straight waveguide section oriented along the z axis (θ = 90°) to 
minimize the Raman interference. This configuration ensures that the 
TE mode predominantly aligns with the non-polar y axis, as depicted 
in the schematic chip diagram in Fig. 4c,d. We used the rapid single 
sideband tuning scheme described in ref. 43 to overcome thermal non-
linearities and initiate solitons at a pump power of 90 mW on-chip using 
ECDL and an erbium-doped fibre amplifier for pumping. The optical 
set-up for single-soliton generation is depicted in Fig. 4c (and described 
in Methods). We achieved single-soliton generation at pulse repeti-
tion rates of 81 GHz (D101_LT_A2_F9_C4_02_WG6, Fig. 4a) and 30.1 GHz 
(D101_LT_A2_F3_C6_01_WG3, Fig. 4b). The full width at half-maximum 
(FWHM) spectral bandwidth of the 81 GHz single soliton is 4.9 THz, 
corresponding to a pulse duration of 63 fs. The 30.1 GHz single-soliton 
state features a bandwidth of 4.0 THz and supports a pulse duration of 
71 fs. Various multi-soliton states were also achieved, and we depict an 
example state with three solitons in Fig. 4e (D101_LT_A2_F2_C4_01_WG3). 
In each of the ten tested devices with the orientation θ = 90°, solitons 
were consistently generated. Conversely, none of the attempts to gen-
erate solitons were successful in the ten devices oriented at θ = 0°. 
This demonstrates that altering the crystal orientation to mitigate the 
Raman effect can be an effective method for generating solitons. The 
low repetition rate of 30.1 GHz solitons allows the direct detection of 
the microwave repetition beat note on a fast photodiode. We measured 
the phase noise of the microwave beat note using an electrical spec-
trum analyser and found a phase noise level of –86 dBc Hz−1 at an offset 
frequency of 10 kHz and –114 dBc Hz−1 at an offset frequency of 1 MHz 
(Fig. 4f), which is higher than earlier measurements using Si3N4 optical 
microresonators44 and in z-cut LiNbO3 (ref. 42). It is notable that here 
DKS generation was achieved in an x-cut ferroelectric crystal sample 
for the first time. This further advances the application of ferroelectric 
materials for microcomb researchers, given that x-cut samples offer 
direct access to the largest electro-optic tensor component1,42.
In summary, we have developed LiTaO3 PICs that are low loss, exhibit 
low birefringence and have similar properties to those fabricated 
from lithium niobate. Crucially, LiTaO3 is already used commercially 
in large volumes for wireless filters, thereby providing a path to scal-
able manufacturing at low cost. Our LTOI PICs achieve similar loss and 
electro-optical performance to the well-established LNOI technol-
ogy that has major potential for use in data-centre interconnects3, 
long-haul optical communications4 and quantum photonics45,46. The 
use of low-cost substrates is of central importance for adoption in appli-
cations such as data-centre interconnects, in which the die size is large, 
owing to the requirements of low modulator voltage and the length of 
travelling wave modulator devices. In our work, we not only establish 
a smart-cut process for the manufacture of LTOI wafer substrates, but 
also demonstrate a complete manufacturing process, including the 
etching of LiTaO3, the removal of the redeposition of etch products 
on the waveguide sidewall, and the manufacture of thick metal elec-
trodes for functional electro-optic devices. We also demonstrate key 
performance metrics, such as low propagation losses of 5.6 dB m−1 and 
a high electro-optic modulation efficiency of VπL = 1.9 V cm at 1,550 nm. 
Our process is fully wafer-scale and based on deep-ultraviolet photo-
lithography and lays the foundation for the scalable manufacture of 
high-performance electro-optical PICs that can harness the scale of 
LTOI wafer fabrication for 5G filters, which is continuing on wafer sizes 
of 150 mm and 200 mm. Our LTOI platform is particularly promising 
for applications that can directly exploit the superior properties of 
the material, such as reduced birefringence. Our platform is capable 
of processing signals across all optical communications bands (1,260–
1,620 nm) in a single PIC, owing to the successful suppression of funda-
mental mode mixing. It also supports soliton-microcomb generation 
in the x-cut, whereas for LiNbO3, soliton-microcomb generation has so 
far been observed only in the z-cut1,42, which has compounded the com-
bination of electro-optical and Kerr nonlinearities47. Moreover, the low 
birefringence allows for the ultra-broadband dispersion engineering of 
LTOI waveguides and for electro-optical frequency comb generation, 
in which the bandwidth is limited by dispersion12 and birefringence48, 
and octave-spanning bandwidth has not yet been achieved. LTOI is 
equally promising for the quantum transduction of single microwave 
photons21,22, which has recently garnered attention as a way to overcome 
the thermal bottlenecks of interfacing with superconducting quantum 
computers49, because the dielectric loss tangent of LiTaO3 (ref. 23) is 
nearly 10-fold lower than that of LiNbO3 (ref. 24).
Online content
Any methods, additional references, Nature Portfolio reporting summa-
ries, source data, extended data, supplementary information, acknowl-
edgements, peer review information; details of author contributions 
and competing interests; and statements of data and code availability 
are available at https://doi.org/10.1038/s41586-024-07369-1.
1.	
Zhu, D. et al. Integrated photonics on thin-film lithium niobate. Adv. Opt. Photonics 13, 
242–352 (2021).
2.	
Boes, A. et al. Lithium niobate photonics: unlocking the electromagnetic spectrum. 
Science 379, eabj4396 (2023).
3.	
Wang, C. et al. Integrated lithium niobate electro-optic modulators operating at 
CMOS-compatible voltages. Nature 562, 101–104 (2018).
4.	
Xu, M. et al. Dual-polarization thin-film lithium niobate in-phase quadrature modulators 
for terabit-per-second transmission. Optica 9, 61–62 (2022).
5.	
Shen, Y. et al. Deep learning with coherent nanophotonic circuits. Nat. Photon. 11, 441–446 
(2017).
6.	
Butaud, E. et al. Innovative Smart Cut piezo on insulator (POI) substrates for 5G acoustic 
filters. In 2020 IEEE International Electron Devices Meeting (IEDM) (ed. Datta, S.)  
34.6.1–34.6.4 (IEEE, 2020).
7.	
Li, Z. et al. High density lithium niobate photonic integrated circuits. Nat. Commun. 14, 
4856 (2023).
8.	
Thomson, D. et al. Roadmap on silicon photonics. J. Opt. 18, 073003 (2016).
9.	
Margalit, N. et al. Perspective on the future of silicon photonics and electronics. Appl. 
Phys. Lett. 118, 220501 (2021).
10.	
Bruel, M. & Auberton-Hervé, B. A. Smart-Cut: a new silicon on insulator material 
technology based on hydrogen implantation and wafer bonding. Jpn. J. Appl. Phys. 36, 
1636 (1997).
11.	
He, M. et al. High-performance hybrid silicon and lithium niobate Mach–Zehnder 
modulators for 100 Gbit s–1 and beyond. Nat. Photon. 13, 359–364 (2019).
12.	
Zhang, M. et al. Broadband electro-optic frequency comb generation in a lithium niobate 
microring resonator. Nature 568, 373–377 (2019).
13.	
Ballandras, S. et al. New generation of SAW devices on advanced engineered substrates 
combining piezoelectric single crystals and silicon. In 2019 Joint Conference of the IEEE 
International Frequency Control Symposium and European Frequency and Time Forum 
(EFTF/IFC) 1–6 (IEEE, 2019).
14.	
Yan, Y. et al. Wafer-scale fabrication of 42° rotated y-cut LiTaO3-on-insulator (LTOI) 
substrate for a SAW resonator. ACS Appl. Electron. Mater. 1, 1660–1666 (2019).
15.	
SOITEC. Capital markets day 2021. https://www.soitec.com/en/capital-markets-day-2021 
(2021).
16.	
Gruber, M. et al. Atomistic origins of the differences in anisotropic fracture behaviour of 
LiTaO3 and LiNbO3 single crystals. Acta Mater. 150, 373–380 (2018).
17.	
Zanatta, A. R. The optical bandgap of lithium niobate (LiNbO3) and its dependence with 
temperature. Results Phys. 39, 105736 (2022).
18.	
Dhar, A. & Mansingh, A. Optical properties of reduced lithium niobate single crystals.  
J. Appl. Phys. 68, 5804–5809 (1990).
19.	
Çabuk, S. & Mamedov, A. Urbach rule and optical properties of the LiNbO3 and LitTaO3.  
J. Opt. A: Pure Appl. Opt. 1, 424 (1999).
20.	 Meyn, J.-P. & Fejer, M. M. Tunable ultraviolet radiation by second-harmonic generation in 
periodically poled lithium tantalate. Opt. Lett. 22, 1214–1216 (1997).
21.	
Javerzac-Galy, C. et al. On-chip microwave-to-optical quantum coherent converter based 
on a superconducting resonator coupled to an electro-optic microresonator. Phys. Rev. A 
94, 053815 (2016).
22.	 Han, X., Fu, W., Zou, C.-L., Jiang, L. & Tang, H. X. Microwave-optical quantum frequency 
conversion. Optica 8, 1050–1064 (2021).
23.	 Jacob, M. V. et al. Temperature dependence of permittivity and loss tangent of lithium 
tantalate at microwave frequencies. IEEE Trans. Microw. Theory Tech. 52, 536–541 (2004).
24.	 Yang, R.-Y., Su, Y.-K., Weng, M.-H., Hung, C.-Y. & Wu, H.-W. Characteristics of coplanar 
waveguide on lithium niobate crystals as a microwave substrate. J. Appl. Phys. 101, 014101 
(2007).
25.	 Tormo-Marquez, V., Díaz-Hijar, M., Carrascosa, M., Shur, V. Y. & Olivares, J. Low loss 
optical waveguides fabricated in LiTaO3 by swift heavy ion irradiation. Opt. Express 27, 
8696–8708 (2019).
26.	 Soltani, M. et al. Ultrahigh Q whispering gallery mode electro-optic resonators on a 
silicon photonic chip. Opt. Lett. 41, 4375–4378 (2016).
27.	
Lu, Y., Johnston, B., Dekker, P., Withford, M. J. & Dawes, J. M. Channel waveguides in 
lithium niobate and lithium tantalate. Molecules 25, 3925 (2020).
28.	 Yan, X. et al. High optical damage threshold on-chip lithium tantalate microdisk 
resonator. Opt. Lett. 45, 4100–4103 (2020).

<!-- page 7 -->
790  |  Nature  |  Vol 629  |  23 May 2024
Article
29.	 Jia, Y., Wang, L. & Chen, F. Ion-cut lithium niobate on insulator technology: recent 
advances and perspectives. Appl. Phys. Rev. 8, 011307 (2021).
30.	 Del’Haye, P., Arcizet, O., Gorodetsky, M. L., Holzwarth, R. & Kippenberg, T. J. Frequency 
comb assisted diode laser spectroscopy for measurement of microcavity dispersion.  
Nat. Photon. 3, 529–533 (2009).
31.	
Xiao, X. et al. Performance of LiTaO3 crystals and thin films and their application. Crystals 
13, 1233 (2023).
32.	 Luke, K. et al. Wafer-scale low-loss lithium niobate photonic integrated circuits. Opt. 
Express 28, 24452–24458 (2020).
33.	 Liu, J. et al. High-yield, wafer-scale fabrication of ultralow-loss, dispersion-engineered 
silicon nitride photonic circuits. Nat. Commun. 12, 2236 (2021).
34.	 Shams-Ansari, A. et al. Reduced material loss in thin-film lithium niobate waveguides.  
APL Photonics 7, 081301 (2022).
35.	 Pan, A., Hu, C., Zeng, C. & Xia, J. Fundamental mode hybridization in a thin film lithium 
niobate ridge waveguide. Opt. Express 27, 35659–35669 (2019).
36.	 Lu, C. et al. Highly tunable birefringent phase-matched second-harmonic generation in 
an angle-cut lithium niobate-on-insulator ridge waveguide. Opt. Lett. 47, 1081–1084 
(2022).
37.	
Wang, J., Chen, P., Dai, D. & Liu, L. Polarization coupling of X-cut thin film lithium niobate 
based waveguides. IEEE Photonics J.12, 2200310 (2020).
38.	 Chen, F. S. Optically induced change of refractive indices in LiNbO3 and LiTaO3. J. Appl. 
Phys. 40, 3389–3396 (1969).
39.	 Herr, T. et al. Temporal solitons in optical microresonators. Nat. Photon. 8, 145–152 
(2014).
40.	 Kippenberg, T. J., Gaeta, A. L., Lipson, M. & Gorodetsky, M. L. Dissipative Kerr solitons in 
optical microresonators. Science 361, eaan8083 (2018).
41.	
Repelin, Y., Husson, E., Bennani, F. & Proust, C. Raman spectroscopy of lithium niobate 
and lithium tantalate. Force field calculations. J. Phys. Chem. Solids 60, 819–825 
(1999).
42.	 He, Y. et al. High-speed tunable microwave-rate soliton microcomb. Nat. Commun. 14, 
3467 (2023).
43.	 Stone, J. R. et al. Thermal and nonlinear dissipative-soliton dynamics in 
Kerr-microresonator frequency combs. Phys. Rev. Lett. 121, 063902 (2018).
44.	 Liu, J. et al. Photonic microwave generation in the X- and K-band using integrated soliton 
microcombs. Nat. Photon. 14, 486–491 (2020).
45.	 Zhao, J., Ma, C., Rüsing, M. & Mookherjea, S. High quality entangled photon pair 
generation in periodically poled thin-film lithium niobate waveguides. Phys. Rev. Lett. 
124, 163603 (2020).
46.	 Nehra, R. et al. Few-cycle vacuum squeezing in nanophotonics. Science 377, 1333–1337 
(2022).
47.	
Gong, Z., Shen, M., Lu, J., Surya, J. B. & Tang, H. X. Monolithic Kerr and electro-optic 
hybrid microcombs. Optica 9, 1060–1065 (2022).
48.	 Hu, Y. et al. High-efficiency and broadband on-chip electro-optic frequency comb 
generators. Nat. Photon. 16, 679–685 (2022).
49.	 Youssefi, A. et al. A cryogenic electro-optic interconnect for superconducting devices. 
Nat. Electron. 4, 326–332 (2021).
Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in 
published maps and institutional affiliations.
Open Access This article is licensed under a Creative Commons Attribution 
4.0 International License, which permits use, sharing, adaptation, distribution 
and reproduction in any medium or format, as long as you give appropriate 
credit to the original author(s) and the source, provide a link to the Creative Commons licence, 
and indicate if changes were made. The images or other third party material in this article are 
included in the article’s Creative Commons licence, unless indicated otherwise in a credit line 
to the material. If material is not included in the article’s Creative Commons licence and your 
intended use is not permitted by statutory regulation or exceeds the permitted use, you will 
need to obtain permission directly from the copyright holder. To view a copy of this licence, 
visit http://creativecommons.org/licenses/by/4.0/.
© The Author(s) 2024

<!-- page 8 -->
Methods
LTOI wafer fabrication
The thin-film LiTaO3 wafers were fabricated by ion cutting and wafer 
bonding, starting with ion implantation into a 525-μm-thick bulk LiTaO3 
wafer. Commercially available optical-grade and acoustic-grade bulk 
LiTaO3 wafers were used. Both grades of wafer are congruent compo-
sitions and their cost is essentially the same. The difference between 
acoustic-grade and optical-grade LiTaO3 lies only in whether the mate-
rial has undergone a chemical reduction process (typically annealed in 
carbon powder)31: acoustic-grade LiTaO3 is reduced but optical-grade 
LiTaO3 is not. Of the two, the optical-grade LiTaO3 wafers exhibited 
slightly better crystalline quality, evidenced by a narrower FWHM 
extracted from the X-ray radiation diffraction rocking curve (Extended 
Data Fig. 1c,e). Hydrogen ions with an energy of 100 keV and a fluence 
of 7.0 × 1016 cm–2 were implanted into a 4-inch x-cut bulk LiTaO3 wafer. 
An ion-damaged layer was introduced under the surface. Subsequently, 
the implanted wafer was flipped and bonded to a blank 525-μm-thick 
high-resistivity silicon carrier wafer covered with 4.7-μm-thick thermal 
silicon dioxide. A subsequent thermal annealing step (190 °C for 10 h) 
facilitated the separation of the residual bulk wafer and the exfoliated 
LiTaO3 thin film. After that, we performed edge removal of the LiTaO3 
thin film and chemical mechanical polishing to remove the rough and 
defect-rich layer of LiTaO3 that was strongly affected by H-ion implanta-
tion and thin the LiTaO3 film to the desired thickness of 600 nm. The 
established process is adaptable to accommodate wafer sizes of 6 or 
8 inches.
LTOI PICs fabrication
We adapted and improved the DLC-based process recently dem-
onstrated for the LNOI platform7 to fabricate LiTaO3 PICs. The 
process flow is schematically illustrated in Fig. 1f. First, we depos-
ited a 30-nm-thick layer of Si3N4, a 480-nm-thick layer of DLC and a 
60-nm-thick layer of Si3N4 by PECVD as the main hard mask for sub-
sequent IBE. Then we defined the photonic waveguides and compo-
nents by deep-ultraviolet stepper photolithography and transferred 
the pattern first into a thin Si3N4 layer by fluorine-based dry etching 
and subsequently into the DLC hard mask layer by oxygen-based 
dry etching in a reactive ion etcher. The main etch of the photonic 
device layer was performed by IBE removing 500 nm of LiTaO3 and 
leaving a 100-nm-thick continuous LiTaO3 slab across the wafer. After 
dry etching, it is known for LiNbO3 (ref. 7) that an extra wet etch-
ing step with RCA-1 solution is needed because of the non-volatile 
by-product accumulating on the waveguide sidewall. However, LiTaO3 
exhibits not only a higher mass density but also stronger chemical 
bonds (Extended Data Table 1), resulting in increased mechanical 
and chemical strength, so a method for removing by-products is 
not needed for LiTaO3. Here, we removed the LiTaO3 redeposition 
with a more-alkaline solution of 3:1 KOH(40%):H2O2(30%). After 
that, we annealed the wafer at 500 °C in an oxygen atmosphere 
and deposited a 2-μm-thick upper cladding with PECVD based on 
a hydrogen-free precursor to avoid overtone absorption from optical 
phonons of the Si–OH stretch vibration around 1.5 μm. The subsequent 
chip release entailed processes involving dry etching of chip bounda-
ries in SiO2 using fluorine-based chemistry, additional etching of the 
silicon carrier using the Bosch process, and backside wafer grinding. 
Apart from the low-loss optical waveguides, high-quality electrodes 
are also required for the PICs. We develop a DUV stepper-compatible 
lift-off process with a dielectric sacrificial layer (Extended Data Fig. 5a). 
After finishing the fabrication of the optical waveguides, we deposited 
a SiO2 layer with PECVD (step 1) with a thickness that exceeds the 
desired metal thickness for the lift-off process. Then the pattern 
was defined by DUV-stepper lithography and transferred to the 
SiO2 by dry etching. It is critical to undercut the sacrificial layer for 
the lift-off process, which we performed by dipping the wafer into 
buffered hydrofluoric acid (step 2; Extended Data Fig. 5b) for a short 
time. Metal deposition and the following photoresist lift-off created 
the metal electrodes (steps 3 and 4). As for the air cladding devices, 
the SiO2 layer could be removed with another wet etching in buffered 
hydrofluoric acid (step 5; Extended Data Fig. 5c).
Photorefractive effect comparison between LiTaO3 and LiNbO3
The measured LTOI (D101_LT_A2_F2_C3_02_WG2) and LNOI (D101_LN_
F2_C3_02_WG2) racetrack microresonators have identical waveguide 
dimensions of 2 μm × 0.5 μm and a slab thickness of 100 nm. Both 
microresonators are uncladded and have the same free spectral range 
of 80 GHz. A CW pump laser was Pound–Drever–Hall locked to the 
cavity resonance and the resonance shift over time was monitored by a 
wavemeter. During the experiment, the input optical power delivered 
by the input lensed fibre was maintained at 3 mW for both the LNOI 
and LTOI samples, which experienced identical fibre–chip coupling 
losses of 6 dB per facet, resulting in an equivalent power of approxi-
mately 0.75 mW in the bus waveguide for each sample. The extinction 
ratio and loaded linewidth of the resonances were determined using a 
frequency-comb-calibrated transmission curve measurement, as illus-
trated in Fig. 2b. Both the LNOI and LTOI samples exhibited an extinction 
ratio of 50%. The loaded linewidth for the LNOI resonance was measured 
at 265.4 MHz (κ0/2π = 33.4 MHz and κex/2π = 232 MHz), whereas for 
the LTOI resonance, it was found to be 150.8 MHz (κ0/2π = 18.8 MHz 
and κex/2π = 132 MHz), indicating higher intra-cavity power for the 
LTOI sample. The resonance frequency shift, denoted as Δf, over 
time t, induced by photorefractivity, was modelled as a result of the 
charge-accumulation process: Δf = Δf0(1 − e−t/T), where T represents 
the time constant of the charge accumulation process and Δf0 is the 
equilibrium photorefractive-induced frequency shift. The data pre-
sented in Extended Data Fig. 3 show a notably smaller time constant, 
T, and equilibrium shift, Δf0, for LTOI than for LNOI. This observation 
is consistent with results obtained from measurements in bulk LiTaO3 
crystals, as previously reported38.
Electro-optical device characterization
Linear electro-optic tuning measurement was performed in the C-band 
and O-band using an ECDL. A fibre polarization controller was used to 
ensure the excitation of the TE mode. The laser frequency was calibrated 
with a 250 MHz phase modulation by detecting the sidebands around 
the resonance. A voltage was applied across two of the four electrodes 
and an oscilloscope was used to record the shift of the resonance to 
achieve the measurement of the voltage tuning coefficient. For Vπ meas-
urement, the MZM modulator was driven using a 10 kHz triangular volt-
age signal while real-time monitoring of the optical transmission signal 
was done. The extinction ratio was measured to be 15 dB and could be 
further improved by using directional couplers. For the high-speed 
electro-optical bandwidth measurement, a pair of high-speed micro-
wave probes was used to deliver the microwave signal to the input 
port of the transmission line. The output of the transmission line was 
terminated with a load of 50 Ω. The light was coupled into and collected 
out of the chip using tapered lensed fibres. The modulated optical 
signal was pre-amplified and filtered through an erbium-doped fibre 
amplifier and a bandpass filter, then detected by a 50 GHz photodiode 
(XPDV2120RA-VF-FP). The S21 response (ratio of powers) was measured 
by a 67 GHz vector network analyser (VNA, R&S ZNA67).
Soliton and Raman measurement
We used a rapid single-sideband tuning scheme described previously43 
to overcome thermal nonlinearities and initiate solitons at a pump 
power of 90 mW on-chip using an ECDL and an erbium-doped fibre 
amplifier for pumping. The optical set-up for soliton generation is 
depicted in Fig. 4c. The microresonator had a waveguide cross-section 
of 2 μm × 0.5 μm with a 100-nm-thick slab. Raman measurement 
was performed using a confocal RM5 Raman microscope. A 532 nm 

<!-- page 9 -->
Article
excitation laser was pumped with a 20× air objective lens and the 
Raman scattering signal was collected by the same objective. We used 
a UHTS300 spectrometer with a grating of 1,800 grooves per mm. 
A half-waveplate was used to change the polarization direction of the 
excitation laser. The measured Raman spectra of x-cut LNOI and LTOI 
for the excitation laser with polarization angles of 0°, 30°, 60° and 90° 
with respect to the z axis are depicted in Extended Data Fig. 4a,b. Both 
samples have several strong vibration phonon branches with large 
linewidths. Typical Raman peaks are labelled. The strongest peak at 
517 cm−1 corresponds to the silicon substrate. The Raman intensity 
becomes lower when the polarization of incident light transitions from 
being parallel to the z axis to being parallel to the y axis. Remarkably, 
this reduction is more pronounced in LTOI than in LNOI (Extended 
Data Fig. 4c). We therefore used racetrack microresonators with the 
straight waveguide section oriented along the z axis to minimize the 
Raman interference. This configuration ensured that the TE mode 
predominantly aligned with the non-polar y axis, as depicted in the 
schematic chip diagram in Fig. 4c,d.
Material properties comparison between LiTaO3 and LiNbO3
LTOI not only has the advantage of higher production volume but also 
has similar or even better performance than LNOI, owing to the inher-
ent properties of LiTaO3. A summary of a comparison of the material 
properties of LiTaO3 and LiNbO3 is shown in Extended Data Table 1. All 
the properties listed correspond to the congruent compositions, which 
are both more readily available and more widely used in various fields 
than their stoichiometric counterparts.
Data availability
The data used to produce the plots in this paper are available at Zenodo 
at https://doi.org/10.5281/zenodo.10215427 (ref. 50).
Code availability
The code used to produce the plots in this paper is available at Zenodo 
at https://doi.org/10.5281/zenodo.10215427 (ref. 50).
 
50.	 Wang, C., & Kippenberg, T. J. Lithium tantalate photonic integrated circuits for volume 
manufacturing. Zenodo https://doi.org/10.5281/zenodo.10215426 (2023).
Acknowledgements The samples were fabricated in the EPFL Center of 
MicroNanoTechnology (CMi) and the Institute of Physics (IPHYS) clean-room. The LTOI wafers 
were fabricated in Shanghai Novel Si Integration Technology (NSIT) and the SIMIT-CAS.  
This work has received funding from the National Natural Science Foundation of China 
(62293521). T.J.K. acknowledges funding from the European Research Council grant 835329 
(ExCOM-cCEO) and from the EU Horizon Europe research and innovation program through 
grant 101113260 (HDLN). J.R. acknowledges funding from the SNSF through Ambizione 
Fellowship 201923. C.W. acknowledges financial support from the China Scholarship Council 
(202104910464). X.O. acknowledges the National Key R&D Program (2022YFA-1404601) from 
the Ministry of Science and Technology of China.
Author contributions C.W. and Y.C. fabricated the LTOI wafers with technical support from  
K.H. and X.W. C.W. and Z.L. fabricated the devices. J.R., Z.L. and C.W. performed numerical 
simulations and designed the devices. C.W. and Z.L. characterized the samples. C.W., Z.L., G.L., 
J.Z., M.C., X.J., A.D., T.B. and W.K. performed experiments. J.R., C.W, G.L. and Z.L. analysed the 
data, prepared the figures and wrote the manuscript with input from all authors. T.J.K. and X.O. 
supervised the project.
Funding Open access funding provided by EPFL Lausanne.
Competing interests T.J.K. is a co-founder and shareholder of Luxtelligence SA, a foundry 
commercializing LiNbO3 PICs, as well as DEEPLIGHT SA, a start-up commercializing PIC-based 
frequency-agile low-noise lasers.
Additional information
Supplementary information The online version contains supplementary material available at 
https://doi.org/10.1038/s41586-024-07369-1.
Correspondence and requests for materials should be addressed to Xin Ou or  
Tobias J. Kippenberg.
Peer review information Nature thanks Detlef Kip and the other, anonymous, reviewer(s) for 
their contribution to the peer review of this work. Peer reviewer reports are available.
Reprints and permissions information is available at http://www.nature.com/reprints.

<!-- page 10 -->
42 MHz
F1
51 MHz
F2
43 MHz
F6
48 MHz
F4
35 MHz
F3
73 MHz
F5
39 MHz
F8
49 MHz
F9
39 MHz
F7
43 MHz
F1
135 MHz
F2
42 MHz
F6
123 MHz
F4
130 MHz
F3
63 MHz
F5
48 MHz
F8
78 MHz
F9
NA
F7
17.4
17.42
Rocking Angle (degree)
Reduced (black)
Unreduced (white) 
b
c
d
e
a
Acoustic grade LT bulk wafers
Optical grade LT bulk wafers
Smart-cut
Smart-cut
Rocking Angle (degree)
Extended Data Fig. 1 | Optical loss of the acoustic and optical grade LTOI 
integrated photonic devices. (a) Photo of the bulk acoustic and optical grade 
LiTaO3 wafers. The black and yellow wafers are reductive and original acoustic 
grade LiTaO3 bulk wafers, respectively. The white wafers are optical grade.  
(b) Wafer map of the optical loss of LiTaO3 microresonators fabricated on 
acoustic grade LTOI wafers. The resonators are 2 μm wide racetracks with 80 GHz 
free spectral range. Figure shows the most probable values of the intrinsic loss 
rate κ0/2π for a resonator from every field. The test chip from one field on the 
acoustic grade wafer was damaged before the measurement (shown as NA).  
(c) XRD rocking curve measured on the LiTaO3 (110) Bragg peak of the acoustic 
grade LTOI wafer after bonding and polishing. (d,e) Same as (b,c) but for optical 
grade LTOI wafer.

<!-- page 11 -->
Article
3020
3040
3060
3080
3100
3900
3920
3940
3960
3980
1 = 7516.3 fs1/mm 
2 = -163 fs2/mm 
3 = 602.93 fs3/mm 
180
190
200
210
220
230
240
7495
7500
7505
7510
7515
7520
1 (fs/mm)
3100
3200
3300
3400
3500
3600
3700
3800
3900
OFDR (20 dB/div.)
OFDR (20 dB/div.)
OFDR (20 dB/div.)
OFDR (20 dB/div.)
180
190
200
210
220
230
240
Frequency (THz)
Frequency (THz)
7
8
9
10
11
12
13
3100
3200
3300
3400
3500
3600
3700
3800
3900
Distance  x  ng
-1 (mm)
Distance  x  ng
-1 (mm)
Distance  x  ng
-1 (mm)
Distance  x  ng
-1 (mm)
Loss (dB/m)
a
b
d
c
e
f
g
500 µm
Extended Data Fig. 2 | Linear loss and dispersion measurement. (a) Optical 
micrograph of LTOI spiral (D101_LT_A2_F2_C7_Spiral5) with a length of 39 cm 
and a waveguide cross-section 1.75 μm × 0.6 μm. The rectangular footprint is 
5 × 2 mm2. (b) Segmented Fourier transform of the optical frequency domain 
reflectometry (OFDR) signal showing the strength of the coherent optical 
backreflection as a function of the optical length. Colors encode the central 
wavelength of the segments and correspond to illustrations in panel (d). An 
optical distance of 88 cm can be identified according to the reflection peaks of 
the front and back facets of the waveguide spiral chip. A minor fabrication 
defect is found at an optical distance of 3.36 m. (c) Same as panel (b) but 
individual traces offset by 3 dB to highlight the linear fit for extraction of the 
optical propagation loss. (d) Optical propagation losses extracted from the  
fits in panel (c) showing a propagation loss of around 8 dB/m in the optical C- 
and L-bands and around 10 dB/m in the O-band. (e,f) Same as panel (b) but 
highlighting the regions around the front and back facet reflections of the  
chip. The colored markers point to the extracted position of the facet and are 
used to infer the dispersion of the spiral. (f) Frequency-dependent optical 
group velocity v
β c
=
= 2.25
g
1
−1
 of the waveguide spiral and fit of the anomalous 
dispersion profile of β2 = -163 fs mm−2. The dispersion contribution of the  
300 μm long tapers is neglected.

<!-- page 12 -->
PD
Power
Meter
mW
CW pump
PM
PM
PID
Wave
meter
OSC
FPC
Microresonator
MIX
PD
PD
Power
Meter
mW
90:10
99:1
75:25
Att.
0
10
20
30
40
50
60
70
80
90
Time (s)
0
20
40
60
80
100
Resonance Shift (MHz)
LiNbO3 measured
LiNbO3 experimental fit
LiTaO3 measured
LiTaO3 experimental fit
a
b
Extended Data Fig. 3 | Photorefractive effect comparison between LiTaO3 
and LiNbO3. (a) Schematic of resonance drifting measurement setup. A pump 
laser is Pound-Drever-Hall (PDH) locked to the resonance microresonators and 
the wavelength is monitored by a wavemeter. (b) Photorefractive-induced 
resonance frequency shift in LNOI (red) and LTOI (blue) microresonators.

<!-- page 13 -->
Article
0
20
40
60
80
θ (degree)
0.6
0.7
0.8
0.9
1
Intensity Ratio
y
z
Orientation
Raman shift (cm-1)
LNOI
LTOI
Intensity (Arb. units)
Intensity (Arb. units)
0°
30°
60°
90°
LTOI
0°
30°
60°
90°
LNOI
200
300
400
600
700
800
500
E(1TO)
E(1TO)
A(4TO)
A(4TO)
E(2TO)
A(1TO)
E(8TO)
a
b
c
θ
Polarization
Extended Data Fig. 4 | Raman intensity comparison between LNOI and 
LTOI. Raman spectra of (a) x-cut LNOI and (b) LTOI for the excitation laser with 
polarization angles 0°, 30°, 60°, 90° with respects to the z-axis. (c) Variation of 
the Raman intensity ratio of LNOI and LTOI for the excitation laser with 
different polarization angles. The Raman intensity is obtained by integrating 
all peaks that correspond to LiNbO3 or LiTaO3 and is normalized. Inset 
illustrates the angles between the orientation and the polarization of the 
excitation laser.

<!-- page 14 -->
Si
LT
PR
Metal
1µm
5 µm
a
b
c
1. SiO deposition and
 photolithography
2. SiO dry and wet etching
3. Metal evaporation
4. Lift-off
5. SiO removal
SiO
SiO
Extended Data Fig. 5 | Scalable manufacturing process based on DUV 
stepper lithography for thick high speed gold electrodes. (a) Fabrication 
process flow of the electrodes. (b) Scanning electron micrography (SEM) of the 
cross-section of a silicon dummy wafer after step 2 in panel (a). The photoresist 
is colored in red, silicon dioxide in pink, and silicon in blue. (c) SEM of fabricated 
electrodes and LiTaO3 waveguide. The gold is colored in yellow and LiTaO3 in 
light blue.

<!-- page 15 -->
Article
Extended Data Table 1 | Comparison of the material properties of LiTaO3 and LiNbO3
The lattice constant, density, Curie point, bandgap, dielectric constant, refractive index, birefringence, electro-optic coefficient, optical damage threshold and coercive electrical field are 
compared.

