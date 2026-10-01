---
paper_id: powell2024
source_url: https://doi.org/10.1364/cleo_si.2024.sm2d.2
doi: 10.1364/cleo_si.2024.sm2d.2
license: CC-BY-NC-ND-4.0
sha256: 96bfb0e32c8c8bfcde55452c169edfd5257cf48b22e5d9afbcc50ce9ebdb2c9f
pages: 5
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
A sub-volt near-IR lithium tantalate electro-optic modulator
Keith Powell,1, a) Dylan Renaud,1 Xudong Li,1 Daniel Assumpcao,1 C. J. Xin,1 Neil Sinclair,1, b) and
Marko Lončar1, c)
John A. Paulson School of Engineering and Applied Science, Harvard University, 29 Oxford St., Cambridge,
MA 02138 USA
(Dated: 5 May 2025)
We demonstrate a low-loss integrated electro-optic Mach-Zehnder modulator in thin-film lithium tantalate at 737 nm,
featuring a low half-wave voltage-length product of 0.65 V·cm, an extinction ratio of 30 dB, low optical loss of 5.3
dB, and a detector-limited bandwidth of 20 GHz. A small < 2 dB DC bias drift relative to quadrature bias is measured
over 16 minutes using 4.3 dBm of on-chip power in ambient conditions, which outperforms the 8 dB measured using a
counterpart thin-film lithium niobate modulator. Finally, an optical loss coefficient of 0.5 dB/cm for a thin-film lithium
tantalate waveguide is estimated at 638 nm using a fabricated ring resonator.
I.
INTRODUCTION
Thin-film lithium niobate (TFLN) electro-optic (EO) inte-
grated circuits have shown promise in furthering optical sci-
ence and technology1–5. Their advantage derives from the
combination of large Pockels coefficient (∼30 pm/V) across
a wide wavelength range (LN bandgap is 3.63 eV) and the
ability to realize low-loss optical waveguides using conven-
tional nanofabrication techniques (lithography, etching, etc).
Recently, thin-film lithium tantalate (TFLT) has become com-
mercially available, and has been explored for electro-optic
circuits at telecommunication wavelength6–13. This has been
primarily motivated by the similar or even improved proper-
ties LT has compared to LN: EO coefficient of r33~30 pm/V14,
bandgap of 3.93 eV15, 23× lower birefringence than LN
at 633 nm16, 5× lower photorefraction than LN at visi-
ble wavelengths17,18, 2500× higher optical damage threshold
than LN for green light19,20, and 10× lower RF loss tangent
than LN16,21.
Lower birefringence and reduced effects of charge trans-
port are the main drivers of TFLT photonics. For example,
it is well-known that photo-induced charge transport along
with photovoltaic effects can enhance EO relaxation, which
manifests as an uncontrolled variation of the optical phase
of light in the crystal due to charge migration. In particu-
lar, the relaxation rate will increase with more applied op-
tical power and can be exacerbated with applied DC or RF
field.
This effect reduces the DC stability of electro-optic
circuits, such as Mach-Zehnder modulators (MZMs), and has
been one of the main challenges faced by TFLN photonics22.
Several methods have been used to mitigate EO relaxation ef-
fects and, hence, overcome bias point drifts of EO MZMs.
These include operation at reduced optical powers (<10 dBm)
and use of heaters with the thermo-optic effect to operate at
the desired bias point4,22, both of which restrict usability and
practicality, or may be infeasible for the application at hand.
Recently, telecommunication-wavelength TFLT MZMs were
demonstrated by our group6 and others10,12 to have superior
a)keith@luminacorp.com.au
b)neils@seas.harvard.edu
c)loncar@seas.harvard.edu
performance to TFLN modulators. We inferred a slow EO
relaxation for on-chip optical powers up to 12.1 dBm6.
The performance of electro-optic devices in TFLT at other
wavelengths has yet to be demonstrated. Of particular inter-
est are the visible and near-IR wavelengths23, which are rele-
vant for applications in imaging24, clocks25, data centers26,
displays27, spectroscopy28, and quantum information29, for
instance. Motivated by these, TFLN MZMs have been de-
veloped at 738 nm30, from 400-700 nm31, at 768 nm by
upconversion32, and recently at 456 nm33. Stronger photo-
induced EO relaxation is expected at shorter wavelengths ap-
proaching the LN bandgap.
Here we design and fabricate a TFLT MZM operating at
a near-IR wavelength of 737 nm that exhibits superior DC
bias stability in ambient conditions compared to an equivalent
TFLN MZM fabricated with a similar process. Specifically,
we measure < 2 dB laser power fluctuations over 16 minutes
using TFLT compared to 8 dB over the same timescale us-
ing TFLN for an on-chip power of 4.3 dBm when DC biasing
the modulator at quadrature. Furthermore, our TFLT MZMs
feature low half-wave voltage length product of 0.65 V·cm,
a high extinction ratio of 30 dB, low optical loss of 5.3 dB,
and a detector-limited bandwidth of 20 GHz. Note that in-
terest in 737 nm wavelength is motivated by silicon vacancy
color centers in diamond ( SiV−), one of leading solid state
quantum memories34. To further assess the optical loss of our
TFLT waveguides, we fabricate ring resonators and measure
optical quality factors up to 2.8×105 which corresponds to a
0.5 dB/cm propagation coefficient. This measurement is per-
formed at 638 nm wavelength due to the limited tuning range
of our 737 nm-wavelength laser.
II.
DEVICE FABRICATION
An optical microscope image of a fabricated unbalanced
MZM is shown in Fig. 1a. It consists of a directional coupler
as an input beamsplitter and a L = 5 mm long electrode in
the ground-signal-ground configuration followed by another
directional coupler at the output. Grating couplers (not shown
in Fig. 1a) are used to couple light on and off the chip to near-
IR single-mode fibers. The directional coupler is chosen to
minimize optical insertion loss and is carefully optimized to
arXiv:2505.00906v1  [physics.optics]  1 May 2025

<!-- page 2 -->
2
reach 50:50 splitting.
A cross-section of the TFLT device stack is shown in Fig.
1b. The optical layer of the device is defined using 150 keV
electron-beam lithography with 500 nm-thick ma-N2405 re-
sist on top of 200 nm-thick x-cut TFLT-on-SiO2. The waveg-
uide width is designed to be 600 nm. The SiO2 layer is 2
µm-thick and is on a Si substrate. The TFLT is etched by 100
nm using an Ar+-based inductively-coupled plasma reactive-
ion etching. Etch-induced re-deposition is removed using a
high-pH solution. The devices are then annealed in an O2
atmosphere at 520°C for 2 h to mitigate etch-induced imper-
fections. For the MZMs, an 800 nm-thick SiO2 cladding layer
is then deposited by plasma-enhanced chemical vapor deposi-
tion. The ring resonators used to evaluate optical loss are left
un-cladded. Trenches for the electrodes are patterned by 375
nm photolithography with SPR700-1.0 resist and are subse-
quently dry etched using C3F8 and Ar+ gases. Electron-beam
metal evaporation and lift-off is used to define the electrodes
(800 nm-thick Au on 15 nm of Ti). All MZMs are hotplate-
heated at 300°C for 5 h to remove trapped charges which can
negatively impact DC drift effects. The TFLN MZM is fab-
ricated using a similar process with the same waveguide and
electrode geometries. Further fabrication details are outlined
in our previous work6.
III.
RESULTS
A scanning electron microscope image of one of the grat-
ing couplers is shown in Fig. 1c. Using a supercontinuum
source and separate chip with grating couplers connected by a
waveguide, we estimate 3 dB-bandwidth of the coupler to be
35 nm, with a peak efficiency of a few percent (15.7 dB loss)
per coupler. Further design and fabrication improvements are
expected to increase grating coupling efficiency to 30 per-cent
(5 dB loss). Next, using continuous-wave laser light at 737
nm, we direct light through the MZM and estimate its loss to
be 5.3 dB (excluding grating coupler loss) across a 28 mm
device length including routing waveguides. The transmis-
sion is mainly limited by metal absorption, bending loss, and
fabrication-induced sidewall roughness.
Next we characterize the electro-optic performance of the
MZM using 737 nm laser light with 4.3 dBm on-chip power.
First, a Hz-rate varied applied voltage reveals a high extinc-
tion ratio of 29.6 dB (Fig. 1d). This ratio suggests the direc-
tional couplers have a splitting ratio of 49.5:50.5, which are
near optimal of 50:50. This measurement yields a Vπ of 1.3
V (Fig. 1d), corresponding to a low 0.65 V·cm VπL, which
is comparable to TFLN near-IR MZMs30. We then use a Vec-
tor Network Analyzer (VNA, Agilent E8364B) to send high
frequency electrical signals to the modulator. The modula-
tor is biased at quadrature and the resultant modulated opti-
cal signals are directed to a 20 GHz-bandwidth photodetector.
The detector is connected to the VNA to form the electric-
optic (EO) measurement loop. The EO performance (S21) and
electrical reflection (S11) of the MZM is shown in Fig. 1e.
The 3-dB EO roll-off frequency is ∼5 GHz, whereas the 3
dB roll-off frequency in terms of half-wave voltage Vπ, that
is the 6 dB line in S21 Fig. 1e, exceeds the bandwidth of
our photodetector. The rapid roll-off in combination with flat
high-frequency response suggests that our device suffers from
imperfect impedance matching rather than velocity matching.
This can be addressed using a thicker bottom oxide layer in
conjunction with redesigned, e.g. segmented, electrodes. The
impedance mismatch is also consistent with the strong reflec-
tion measured in S11 (Fig. 1e).
Next, we measure the DC bias stability of our MZM over
long timescales. First we apply a 0.1 Hz-frequency square
wave to the modulator using an on-chip optical power of 4.3
dBm at 737 nm and measure the modulator response with a
photodetector. The input drive signal and corresponding out-
put optical signal is shown in Fig. 2a in red and blue, respec-
tively. At this frequency electro-optic relaxation is difficult to
observe. Thus, to elucidate EO relaxation at longer timescales
we apply a step voltage (from in-phase to quadrature) to the
device and hold the voltage constant. Using 4.3 dBm of on-
chip optical power, we measure a 2 dB of optical power drift
over a 16 minute timescale (Fig. 2b). We perform the same
measurement with 4.3 dBm on-chip power using an equiva-
lent TFLN MZM, which displays a DC bias drift of 8 dB over
the same time scale.
Finally to further investigate optical propagation loss of our
waveguides, we fabricate grating-coupled 239 µm-diameter
micro-ring resonators. The waveguide width for the rings is 1
µm. A SEM image of one of the resonator devices is shown
in the inset of Fig. 3b. Given the limited tuning range of our
737 nm laser, we instead use a tunable 638 nm-wavelength
laser and measure the resonance spectrum of a ring (Fig. 3a).
Laser power fluctuations are observed due to multimode out-
put of the laser and coupling setup (Fig. 3a inset), which
we calibrate away to clearly observe the 205 pm free spectral
range of our ring (Fig. 3a). A fit of the resonance linewidth
produces a FWHM of 2.7 pm (Fig. 3b), corresponding to a
loaded quality (Q) factor of 2.8 × 105. Since the resonator is
strongly overcoupled, this approximates the intrinsic quality
factor, which corresponds to a loss coefficient of 0.5 dB/cm.
We adjusted the laser power to ensure that the resonance
linewidth measurement was not impacted by thermo-optic or
photo-refractive effects. This loss coefficient is comparable to
that measured in TFLN30.
IV.
CONCLUSION
We demonstrated a thin-film lithium tantalate Mach-
Zehnder electro-optic modulator with a slow EO relaxation
and low half-wave voltage length product of 0.65 V·cm of in-
terest for applications of near-IR opto-electronics. The mod-
ulator features a low optical loss of 5.3 dB, an extinction ra-
tio of 30 dB, and a detector-limited bandwidth of 20 GHz,
with the latter being suitable for several spectroscopic and
atomic applications, including interfacing with the SiV−cen-
ter in diamond. A balanced MZM design, as we have demon-
strated at telecommunication wavelength6, could improve the
DC stability further in addition to improved material process-
ing strategies: annealing, doping, surface treatments or differ-

<!-- page 3 -->
3
IN
OUT
OUT
IN
(a)
(b)
Ground (G)
Signal (S)
Ground (G)
1 μm
5 mm
G
S
G
SiO2
Si
x
y
z
y
(c)
LiTaO3
Au
(d)
(e)
ER = 29.6 dB
Vπ = 1.3 V
-6 dB
-3 dB
FIG. 1. Near-IR thin-film lithium tantalate Mach-Zehnder electro-optic modulator (a) Optical micrograph of the fabricated modulator. Crystal
axes and electrode details are labeled. (b) Cross-section of the material stack of the modulator with crystal axes indicated. (c) Scanning electron
micro-graph showing a grating coupler used to couple light to and from our modulator. (d) Measured transfer function of the modulator with
slowly varying applied voltage yields an extinction ratio (ER) of 29.6 dB and half-wave voltage (Vπ) of 1.3 V. (c) Measured electro-optic
frequency response (S21) is limited by the detector used. Reflected RF power (S11) of the modulator transmission line indicates efficient power
delivery to the electrodes.
1
3
5
7
9
11
13
15
(a)
(b)
FIG. 2. Measured electro-optic relaxation of a thin-film lithium tantalate Mach-Zehnder modulator using (a) an applied 0.1 Hz square wave
signal and (b) a voltage step. An equivalent counterpart thin-film lithium niobate modulator relaxes faster under the same conditions.
ent electrode metals.
ACKNOWLEDGMENTS
The authors thank M. Yeh and D. Barton for dis-
cussions.
We acknowledge funding from NSF EEC-
1941583,
AFOSR
FA9550-20-1-01015,
NSF
2138068,
NASA 80NSSC22K0262, MagiQ Technology/Naval Air War-
fare Center N6833522C0413, and Amazon Web Services.
This work was performed in part at the Harvard University
Center for Nanoscale Systems (CNS); a member of the Na-
tional Nanotechnology Coordinated Infrastructure Network
(NNCI), which is supported by the National Science Foun-
dation under NSF award no. ECCS-2025158.

<!-- page 4 -->
4
2.7 pm
637.9
638
638.1
638.2
638.3
638.4
638.5
638.6
638.7
638.8
638.9
Wavelength (nm)
0.02
0.04
0.06
0.08
0.1
0.12
0.14
0.16
0.18
Intensity (A.U.)
(a)
(b)
FIG. 3. (a) Measured transmission spectrum of a thin-film lithium tantalate micro-ring resonator at wavelengths around 638 nm. (b) Resonance
linewidth of the micro-ring with Lorentzian fit reveals a FWHM linewidth of 2.7 pm. Inset: scanning electron microwscope image of the
fabricated micro-ring resonator and bus waveguide.
AUTHOR DECLARATIONS
Conflict of Interest
K.P., N.S., and M.L. are involved in developing lithium tan-
talate technologies at Lumina Corporation. D.R. and M.L. are
involved in developing lithium niobate technologies at Hyper-
Light Corporation.
DATA AVAILABILITY STATEMENT
Data available on request from the authors.
1Y. Hu, D. Zhu, S. Lu, X. Zhu, Y. Song, D. Renaud, D. Assumpcao,
R. Cheng, C. Xin, M. Yeh, et al., “Integrated electro-optics on thin-film
lithium niobate,” Nature Reviews Physics , 1–18 (2025).
2A. Boes, L. Chang, C. Langrock, M. Yu, M. Zhang, Q. Lin, M. Lonˇcar,
M. Fejer, J. Bowers, and A. Mitchell, “Lithium niobate photonics: Un-
locking the electromagnetic spectrum,” Science 379 (2023).
3D. Zhu, L. Shao, M. Yu, R. Cheng, B. Desiatov, C. Xin, Y. Hu, J. Holz-
grafe, S. Ghosh, A. Shams-Ansari, et al., “Integrated photonics on thin-film
lithium niobate,” Advances in Optics and Photonics 13, 242–352 (2021).
4M. Xu, M. He, H. Zhang, J. Jian, Y. Pan, X. Liu, L. Chen, X. Meng,
H. Chen, Z. Li, et al., “High-performance coherent optical modulators
based on thin-film lithium niobate platform,” Nature communications 11,
3911 (2020).
5B. Desiatov, A. Shams-Ansari, M. Zhang, C. Wang, and M. Lonˇcar, “Ultra-
low-loss integrated visible photonics using thin-film lithium niobate,” Op-
tica 6, 380–384 (2019).
6K. Powell, X. Li, D. Assumpcao, L. Magalhães, N. Sinclair, and M. Lonˇcar,
“Dc-stable electro-optic modulators using thin-film lithium tantalate,” Op-
tics Express 32, 44115–44122 (2024).
7C. Wang, Z. Li, J. Riemensberger, G. Lihachev, M. Churaev, W. Kao, X. Ji,
J. Zhang, T. Blesin, A. Davydova, Y. Chen, K. Huang, X. Wang, X. Ou, and
T. J. Kippenberg, “Lithium tantalate photonic integrated circuits for volume
manufacturing,” Nature 629, 784–790 (2024).
8J. Yu,
Z. Ruan,
Y. Xue,
H. Wang,
R. Gan,
T. Gao,
C. Guo,
K.
Chen,
X.
Ou,
and
L.
Liu,
“Tunable
and
stable
micro-
ring
resonator
based
on
thin-film
lithium
tantalate,”
APL
Pho-
tonics
9,
036115
(2024),
https://pubs.aip.org/aip/app/article-
pdf/doi/10.1063/5.0187996/19844291/036115_1_5.0187996.pdf.
9J. Shen, Y. Zhang, C. Feng, Z. Xu, L. Zhang, and Y. Su, “Hybrid lithium
tantalite-silicon integrated photonics platform for electro-optic modula-
tion,” Opt. Lett. 48, 6176–6179 (2023).
10J. Yu,
Z. Ruan,
Y. Xue,
H. Wang,
R. Gan,
T. Gao,
C. Guo,
K.
Chen,
X.
Ou,
and
L.
Liu,
“Tunable
and
stable
micro-
ring
resonator
based
on
thin-film
lithium
tantalate,”
APL
Pho-
tonics
9,
036115
(2024),
https://pubs.aip.org/aip/app/article-
pdf/doi/10.1063/5.0187996/19844291/036115_1_5.0187996.pdf.
11H. Nishi, T. Tsuchizawa, T. Segawa, and S. Matsuo, “Low-loss lithium
tantalate on insulator waveguide towards on-chip nonlinear photonics,” in
2022 27th OptoElectronics and Communications Conference (OECC) and
2022 International Conference on Photonics in Switching and Computing
(PSC) (2022) pp. 1–3.
12C. Wang, D. Fang, J. Zhang, A. Kotz, G. Lihachev, M. Churaev, Z. Li,
A. Schwarzenberger, X. Ou, C. Koos, et al., “Ultrabroadband thin-film
lithium tantalate modulator for high-speed communications,” Optica 11,
1614–1620 (2024).
13J. Zhang, C. Wang, C. Denney, J. Riemensberger, G. Lihachev, J. Hu,
W. Kao, T. Blésin, N. Kuznetsov, Z. Li, et al., “Ultrabroadband integrated
electro-optic frequency comb in lithium tantalate,” Nature , 1–8 (2025).
14J. L. Casson, K. T. Gahagan, D. A. Scrymgeour, R. K. Jain, J. M. Robin-
son, V. Gopalan, and R. K. Sander, “Electro-optic coefficients of lithium
tantalate at near-infrared wavelengths,” J. Opt. Soc. Am. B 21, 1948–1952
(2004).
15S. Çabuk and A. Mamedov, “Urbach rule and optical properties of the
linbo3 and litao3,” Journal of Optics A: Pure and Applied Optics 1, 424
(1999).
16M. Jacob, J. Hartnett, J. Mazierska, V. Giordano, J. Krupka, and M. Tobar,
“Temperature dependence of permittivity and loss tangent of lithium tanta-
late at microwave frequencies,” IEEE Transactions on Microwave Theory
and Techniques 52, 536–541 (2004).
17F.
Holtmann,
J.
Imbrock,
C.
Bäumer,
H.
Hesse,
E.
Krätzig,
and
D.
Kip,
“Photorefractive
properties
of
undoped
lithium
tan-
talate
crystals
for
various
composition,”
Journal
of
Applied
Physics
96,
7455–7459
(2004),
https://pubs.aip.org/aip/jap/article-
pdf/96/12/7455/18720753/7455_1_online.pdf.
18O. Althoff and E. E. Kraetzig, “Strong light-induced refractive index
changes in LiNbO3,” in Nonlinear Optical Materials III, Vol. 1273, edited
by P. Guenter, International Society for Optics and Photonics (SPIE, 1990)
pp. 12 – 19.

<!-- page 5 -->
5
19X. Yan, Y. Liu, L. Ge, B. Zhu, J. Wu, Y. Chen, and X. Chen, “High optical
damage threshold on-chip lithium tantalate microdisk resonator,” Opt. Lett.
45, 4100–4103 (2020).
20Y. Kong, F. Bo, W. Wang, D. Zheng, H. Liu, G. Zhang, R. Rupp, and
J. Xu, “Recent progress in lithium niobate: Optical damage, defect sim-
ulation, and on-chip devices,” Advanced Materials 32, 1806452 (2020),
https://onlinelibrary.wiley.com/doi/pdf/10.1002/adma.201806452.
21R.-Y.
Yang,
Y.-K.
Su,
M.-H.
Weng,
C.-Y.
Hung,
and
H.-W.
Wu,
“Characteristics
of
coplanar
waveguide
on
lithium
nio-
bate
crystals
as
a
microwave
substrate,”
Journal
of
Applied
Physics
101,
014101
(2007),
https://pubs.aip.org/aip/jap/article-
pdf/doi/10.1063/1.2402978/13341406/014101_1_online.pdf.
22J. Holzgrafe, E. Puma, R. Cheng, H. Warner, A. Shams-Ansari, R. Shankar,
and M. Lonˇcar, “Relaxation of the electro-optic response in thin-film
lithium niobate modulators,” Opt. Express 32, 3619–3631 (2024).
23M. A. Tran, C. Zhang, T. J. Morin, L. Chang, S. Barik, Z. Yuan, W. Lee,
G. Kim, A. Malik, Z. Zhang, et al., “Extending the spectrum of fully
integrated photonics to submicrometre wavelengths,” Nature 610, 54–60
(2022).
24F. Wang, Y. Zhong, O. Bruns, Y. Liang, and H. Dai, “In vivo nir-ii fluores-
cence imaging for biology and medicine,” Nature Photonics 18, 535–547
(2024).
25Z. L. Newman, V. Maurice, T. Drake, J. R. Stone, T. C. Briles, D. T.
Spencer, C. Fredrick, Q. Li, D. Westly, B. R. Ilic, et al., “Architecture for the
photonic integration of an optical atomic clock,” Optica 6, 680–685 (2019).
26Q. Cheng, M. Bahadori, M. Glick, S. Rumley, and K. Bergman, “Recent
advances in optical technologies for data centers: a review,” Optica 5, 1354–
1370 (2018).
27T. J. Morin, L. Chang, W. Jin, C. Li, J. Guo, H. Park, M. A. Tran, T. Koml-
jenovic, and J. E. Bowers, “Cmos-foundry-based blue and violet photon-
ics,” Optica 8, 755–756 (2021).
28M.-G. Suh, X. Yi, Y.-H. Lai, S. Leifer, I. S. Grudinin, G. Vasisht, E. C.
Martin, M. P. Fitzgerald, G. Doppmann, J. Wang, et al., “Searching for
exoplanets using a microresonator astrocomb,” Nature photonics 13, 25–30
(2019).
29C. Bradac, W. Gao, J. Forneris, M. E. Trusheim,
and I. Aharonovich,
“Quantum nanophotonics with group iv defects in diamond,” Nature com-
munications 10, 5625 (2019).
30D. Renaud, D. R. Assumpcao, G. Joe, A. Shams-Ansari, D. Zhu, Y. Hu,
N. Sinclair,
and M. Loncar, “Sub-1 volt and high-bandwidth visible to
near-infrared electro-optic modulators,” Nature Communications 14, 1496
(2023).
31S. Xue, Z. Shi, J. Ling, Z. Gao, Q. Hu, K. Zhang, G. Valentine, X. Wu,
J. Staffa, U. A. Javid, and Q. Lin, “Full-spectrum visible electro-optic mod-
ulator,” Optica 10, 125–126 (2023).
32A. Sabatti, J. Kellner, F. Kaufmann, R. J. Chapman, G. Finco, T. Kuttner,
A. Maeder, and R. Grange, “Extremely high extinction ratio electro-optic
modulator via frequency upconversion to visible wavelengths,” Opt. Lett.
49, 3870–3873 (2024).
33O. T. Celik, N. Y. Ammar, T. Park, H. S. Stokowski, K. K. S. Multani, A. Y.
Hwang, S. Gyger, Y. Guo, M. M. Fejer, and A. H. Safavi-Naeini, “Roles
of temperature, materials, and domain inversion in high-performance, low-
bias-drift thin film lithium niobate blue light modulators,” Opt. Express 32,
36160–36170 (2024).
34D. Assumpcao, D. Renaud, A. Baradari, B. Zeng, C. De-Eknamkul, C. Xin,
A. Shams-Ansari, D. Barton, B. Machielse, and M. Loncar, “A thin film
lithium niobate near-infrared platform for multiplexing quantum nodes,”
Nature communications 15, 1–9 (2024).
35S. Huband, D. Keeble, N. Zhang, A. Glazer, A. Bartasyte,
and P. A.
Thomas, “Relationship between the structure and optical properties of
lithium tantalate at the zero-birefringence point,” Journal of Applied
Physics 121 (2017).
36M. Leidinger, S. Fieberg, N. Waasem, F. Kühnemann, K. Buse, and I. Bre-
unig, “Comparative study on three highly sensitive absorption measurement
techniques characterizing lithium niobate over its entire transparent spectral
range,” Optics express 23, 21690–21705 (2015).
37A. M. Glazer, N. Zhang, A. Bartasyte, D. S. Keeble, S. Huband, and P. A.
Thomas, “Observation of unusual temperature-dependent stripes in litao3
and litaxnb1- xo3 crystals with near-zero birefringence,” Journal of Applied
Crystallography 43, 1305–1313 (2010).
38R. A. McCracken, J. M. Charsley, and D. T. Reid, “A decade of astro-
combs: recent advances in frequency combs for astronomy,” Optics express
25, 15058–15078 (2017).
39M. Wang, J. Li, H. Yao, X. Li, J. Wu, K. S. Chiang, and K. Chen, “Thin-film
lithium-niobate modulator with a combined passive bias and thermo-optic
bias,” Opt. Express 30, 39706–39715 (2022).
40D. Zhu, L. Shao, M. Yu, R. Cheng, B. Desiatov, C. Xin, Y. Hu, J. Holz-
grafe, S. Ghosh, A. Shams-Ansari, et al., “Integrated photonics on thin-film
lithium niobate,” Advances in Optics and Photonics 13, 242–352 (2021).
41X. Zhu, Y. Hu, S. Lu, H. K. Warner, X. Li, Y. Song, L. M. aes, A. Shams-
Ansari, A. Cordaro, N. Sinclair, and M. Lonˇcar, “Twenty-nine million in-
trinsic q-factor monolithic microresonators on thin-film lithium niobate,”
Photon. Res. 12, A63–A68 (2024).
42D.
A.
Bryan,
R.
Gerson,
and
H.
E.
Tomaschke,
“Increased
optical
damage
resistance
in
lithium
niobate,”
Applied
Physics
Letters
44,
847–849
(1984),
https://pubs.aip.org/aip/apl/article-
pdf/44/9/847/18451047/847_1_online.pdf.
43A. Ashkin, G. D. Boyd, J. M. Dziedzic, R. G. Smith, A. A. Ballman,
J. J. Levinstein,
and K. Nassau, “OPTICALLY-INDUCED REFRAC-
TIVE INDEX INHOMOGENEITIES IN LiNbO3 AND LiTaO3,” Ap-
plied Physics Letters 9, 72–74 (1966), https://pubs.aip.org/aip/apl/article-
pdf/9/1/72/18419070/72_1_online.pdf.
44L. Wang, S. Liu, Y. Kong, S. Chen, Z. Huang, L. Wu, R. Rupp, and J. Xu,
“Increased optical-damage resistance in tin-doped lithium niobate,” Opt.
Lett. 35, 883–885 (2010).
45Y. Kong, S. Liu, Y. Zhao, H. Liu, S. Chen, and J. Xu, “Highly optical
damage resistant crystal: Zirconium-oxide-doped lithium niobate,” Applied
Physics Letters 91, 081908 (2007), https://pubs.aip.org/aip/apl/article-
pdf/doi/10.1063/1.2773742/13977509/081908_1_online.pdf.
46J. P. Salvestrini, L. Guilbert, M. Fontana, M. Abarkan, and S. Gille, “Anal-
ysis and control of the dc drift in linbo3based mach–zehnder modulators,”
Journal of Lightwave Technology 29, 1522–1534 (2011).
47H. Iwasaki, T. Yamada, N. Niizeki, H. Toyoda, and H. Kubota, “Refractive
indices of litao3 at high temperatures,” Japanese Journal of Applied Physics
7, 185 (1968).
48R. Takigawa, T. Tomimatsu, E. Higurashi,
and T. Asano, “Resid-
ual stress in lithium niobate film layer of lnoi/si hybrid wafer fabri-
cated using low-temperature bonding method,” Micromachines 10 (2019),
10.3390/mi10020136.
49K. D. Baklanova, A. V. Solnyshkin, I. L. Kislova, S. I. Gudkov, A. N.
Belov, V. I. Shevyakov, R. N. Zhukov, D. A. Kiselev,
and M. D. Ma-
linkovich, “Pyroelectric properties and local piezoelectric response of
lithium niobate thin films,” physica status solidi (a) 215, 1700690 (2018),
https://onlinelibrary.wiley.com/doi/pdf/10.1002/pssa.201700690.
50I. G. Wood, P. Daniels, R. H. Brown, and A. M. Glazer, “Optical birefrin-
gence study of the ferroelectric phase transition in lithium niobate tantalate
mixed crystals,” Journal of Physics: Condensed Matter 20, 235237 (2008).
51Y. Yan, K. Huang, H. Zhou, X. Zhao, W. Li, Z. Li, A. Yi, H. Huang, J. Lin,
S. Zhang, et al., “Wafer-scale fabrication of 42° rotated y-cut litao3-on-
insulator (ltoi) substrate for a saw resonator,” ACS Applied Electronic Ma-
terials 1, 1660–1666 (2019).
52D. B. Maring, R. F. Tavlykaev, R. V. Ramaswamy, and S. M. Kostritskii,
“Waveguide instability in litao3,” J. Opt. Soc. Am. B 19, 1575–1581 (2002).
53J. P. Salvestrini, L. Guilbert, M. Fontana, M. Abarkan, and S. Gille, “Anal-
ysis and control of the dc drift in linbo _{3}-based mach–zehnder modula-
tors,” Journal of lightwave technology 29, 1522–1534 (2011).
54P. Kharel, C. Reimer, K. Luke, L. He, and M. Zhang, “Breaking voltage–
bandwidth limits in integrated lithium niobate modulators using micro-
structured electrodes,” Optica 8, 357–363 (2021).
55H. Wang, X. Xing, Z. Ruan, J. Yu, K. Chen, X. Ou, and L. Liu, “Optical
switch with an ultralow dc drift based on thin-film lithium tantalate,” Opt.
Lett. 49, 5019–5022 (2024).

