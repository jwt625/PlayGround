---
paper_id: wang2025
source_url: https://doi.org/10.1021/acsphotonics.5c00159
doi: 10.1021/acsphotonics.5c00159
license: https://doi.org/10.15223/policy-029;https://doi.org/10.15223/policy-037;https://doi.org/10.15223/policy-045
sha256: 08e508b9ea4c2207543e955a75ec34a45152988c904d16603c15a6e85c0b304e
pages: 7
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Thin-Film Lithium Tantalate Modulator Operating at High Optical
Power
Haohua Wang, Ao Cui, Bin Chen, Ziliang Ruan, Changjian Guo,* Kaixuan Chen,* and Liu Liu*
Cite This: https://doi.org/10.1021/acsphotonics.5c00159
Read Online
ACCESS
Metrics & More
Article Recommendations
ABSTRACT: Handling high optical power in electro-optic modulators is
critical for applications in nonlinear optics, microwave photonics, and THz
communications. While thin-film lithium niobate (TFLN) modulators
provide ultralarge bandwidth, their performance under high optical power
is constrained by photorefractive (PR) effects. Thin-film lithium tantalate
(TFLT) addresses this challenge with its significantly higher optical
damage threshold and reduced PR effects, making it a promising material
for high-power applications. In this work, we designed and experimentally
demonstrated a high-performance TFLT modulator capable of stable
operation at optical powers up to 28 dBm (630.9 mW). Compared to
TFLN-based modulators, the TFLT device achieves enhanced stability and
exhibits consistent high efficiency, low insertion loss, and large electro-
optic bandwidth across various power levels. These results highlight the
potential of TFLT modulators for next-generation high-power photonic systems.
KEYWORDS: lithium tantalate, electro-optic modulator, high optical power, photorefractive, optical damage, lithium niobate
■INTRODUCTION
Thin-film lithium niobate (TFLN) electro-optic (EO)
modulators have rapidly advanced in recent years, owing to
their high modulation efficiency, large bandwidth, compact
waveguides, and low loss.1−3 Several high-performance
modulators have been reported, featuring EO bandwidths
exceeding 110 GHz,4 half-wave voltages (Vπ) below 1 V,5 or
insertion losses (IL) under 0.2 dB.6 These devices are expected
to be widely used in next-generation optical interconnects for
data centers, coherent optical communications, and artificial
intelligence computing.7 In these scenarios, high optical power
is typically unnecessary, and modulators generally operate
under relatively low power levels. However, certain applica-
tionssuch as long-haul communications,8 nonlinear optics,9
microwave/mm-wave generation,10,11 THz communications,12
and high-linearity analog photonic links12,13demand not
only large bandwidth and high modulation efficiency but also
the ability to handle high optical power. This capability is
crucial for enhancing signal gain,10,14 mitigating third-order
intermodulation,13 and reducing noise in RF photonic links.10
Unfortunately, undoped lithium niobate (LN), when exposed
to high optical power, often suffers from severe photorefractive
(PR) effects, leading to unstable operating points and even
irreversible damage, such as deformation of the optical mode
field distribution.15−19 This is reflected in its relatively low
optical damage threshold (∼1 kW/cm2 @532 nm).20 In a
review paper, Zhang et al. predicted that TFLN modulators
could potentially handle watt-level optical power.21 However,
this estimate was based solely on calculations from microring
resonator waveguides, and actual power limits for modulators
have yet to be theoretically modeled or experimentally
demonstrated. Valdez et al. reported a Si-LN hybrid integrated
EO modulator capable of handling optical power exceeding
110 mW, by mitigating the two-photon absorption limitations
of silicon waveguides.22 However, their experiment was
restricted to pulsed light operation, leaving the performance
under higher continuous-wave optical power unclear. Doping
LN with ZrO2 or MgO23−25 has been shown to mitigate PR
effects and increase the optical damage threshold. High-
performance MgO-doped TFLN modulators have been
demonstrated within the visible to near-infrared range.26,27
Nonetheless, there has been limited discussion or investigation
into the power-handling capabilities of MgO-doped TFLN
modulators, particularly in the C-band.
Alternatively, thin-film lithium tantalate (TFLT), a promis-
ing candidate in integrated photonics,28 offers mechanical and
chemical stability comparable to TFLN, along with a similar
Received:
January 18, 2025
Revised:
June 20, 2025
Accepted:
June 23, 2025
Article
pubs.acs.org/journal/apchd5
© XXXX American Chemical Society
A
https://doi.org/10.1021/acsphotonics.5c00159
ACS Photonics XXXX, XXX, XXX−XXX
Downloaded via STANFORD UNIV on June 30, 2025 at 01:12:59 (UTC).
See https://pubs.acs.org/sharingguidelines for options on how to legitimately share published articles.

<!-- page 2 -->
linear electro-optic coefficient (r33 = 30.5 pm/V), enabling
high modulation performance.29 Crucially, TFLT has an
optical damage threshold (2000 kW/cm2@532 nm) that is 3
orders of magnitude higher than that of LN.20 This property
has attracted attention in high-power nonlinear optics
applications.20,30−32 In recent studies, different research groups
observed a weak PR effect and minimal direct-current (DC)
drift in TFLT-based microrings,33 optical switches,34 and
modulator,35 confirming the superior stability of TFLT
photonic devices over TFLN.
In this paper, we designed a high-performance silicon-
substrate TFLT modulator and experimentally demonstrated
its stable operation at high optical power (up to 28 dBm, 630.9
mW). Compared to TFLN modulators, the TFLT modulator
exhibited significantly improved stability, highlighting its
higher optical damage threshold and reduced PR effect.
These results underscore TFLT’s potential for high-power
electro-optic modulator applications.
■RESULTS
Design. Figure 1 shows the three-dimensional structure of
the proposed EO modulator, which employs periodic
capacitively loaded traveling-wave (CLTW) electrodes on an
acoustic-grade TFLT on insulator (LTOI) wafer (Novel Si
Integration Technology, China). The wafer consists of a 400
nm thick TFLT layer with an X-112°Y crystal cut, a 4.7 μm
thick buried oxide (BOX) layer, and a silicon substrate. The
modulator’s ridge waveguides have a height of 200 nm, a width
of 1.5 μm, and sidewall angles of 60°. The 3 dB beam splitters/
combiners are designed as multimode interference (MMI)
structures. The modulation electrodes follow a CLTW design
based on a classical coplanar line structure, incorporating with
wide unloaded signal electrodes and periodically loaded T-
segments. This design has been proven effective in achieving
both large EO bandwidth and low Vπ in TFLN modu-
lators.6,36,37 The cross-sectional view of the modulation region
is shown in Figure 1c, featuring an unloaded signal electrode
width of Ws = 80 μm and a gap j = 1.8 μm between the
opposing loaded T-segment electrodes. A 900 nm thick SiO2
cladding layer is applied between the LT layer and metal
electrodes to minimize absorption losses from the metal. The
metal thicknesses for the loaded T-segment electrodes and the
wide unloaded electrodes are set to l = 200 nm and k = 1.1 μm,
respectively. To achieve index matching between the micro-
wave and optical signals, the microwave refractive index is
reduced through backside holes (r = 90 μm) in the silicon
substratea robust solution for high-performance modulators
on silicon substrates.37 The remaining CLTW electrode
parameters are optimized for impedance matching, with
dimensions (a, b, c, d, e) = (15.3, 3, 2, 47, 5) μm, as shown
in Figure 1a.
We simulated the optical and electrical modes of the
modulator using a multiphysics field solver based on the finite
element method at a wavelength of 1550 nm and a frequency
of 100 GHz. The simulated modulation efficiency, represented
by the half-wave voltage-length product VπL = 2.41 V·cm, was
derived for a waveguide oriented along the Y axis of the LT
crystal and for the fundamental transverse-electric (TE) mode,
as shown in Figure 2a. The microwave field is concentrated in
the gap between the opposing loaded T-segment electrodes
(Figure 2b), resulting in strong overlap between the electrical
and optical fields, which contributes to the low VπL value. The
simulated characteristic impedance Zc, RF effective index nm,
and RF loss αm are shown in Figure 2c-e, respectively. The RF
loss curves follow the relation αm = α0f1/2, where α0 represents
the frequency-independent characteristic RF loss, extracted as
0.194 dB·cm−1 GHz−1/2, which is comparable to that of
advanced TFLN modulators.37 The backside hole design
effectively reduces the microwave refractive index, aligning it
closely with the optical mode group index (ng = 2.19). The
characteristic impedance Zc is designed to be 50 Ω. Due to the
low RF loss, index matching, and impedance matching, the
modulator achieves a large EO bandwidth. As shown in Figure
2f, the bandwidth roll-off is less than 1 dB at 100 GHz for a 7
mm-long device.
The designed TFLT modulator was fabricated using
standard microfabrication processes (see Methods). By
adjusting the width of backside hole, the RF effective index
nm could be fine-tuned to closely match the optical mode
group index. The final fabricated modulator has a modulation
length of L = 0.7 cm. It is noteworthy that the waveguide is
oriented along the 112° Y axis of the LT crystal, with the
electric field along the 22° Z axis. Therefore, the EO efficiency
is about 0.83 times that of the highest EO coefficient in the LT
crystal’s optimal orientation.38 Figure 3a−e show some optical
microscope images of the fabricated modulator, including
views of the chip’s backside.
Measurement and Analysis. A tunable laser combined
with a preamplifier (erbium-doped fiber amplifier, EDFA) was
used as the light source, followed by a polarization controller
to ensure operation in the TE mode. Light was coupled into
and out of the modulator using a pair of grating couplers
(GCs), each with a coupling loss of 6 dB at the operating
wavelength. To monitor the output power, 5% of the light was
Figure 1. Proposed TFLT-based EO modulator on a Si substrate using CLTW electrodes with backside holes: (a) Detailed top-view schematic of
the T-segment electrode structures. (b) Three-dimensional schematic of the EO modulator. (c) Cross-sectional schematic of the modulation
section.
ACS Photonics
pubs.acs.org/journal/apchd5
Article
https://doi.org/10.1021/acsphotonics.5c00159
ACS Photonics XXXX, XXX, XXX−XXX
B

<!-- page 3 -->
directed to a power meter, while the remaining 95% was sent
into another EDFA to maintain consistent optical power
before reaching the high-speed photodiode (PD). RF probes
were employed to deliver electrical and RF signals to the
modulator. The electrical electrical (EE) responses of the
modulator were measured using a vector network analyzer, as
shown in Figure 3f. The EE reflection S11 remained below −20
dB across the entire frequency range, indicating good
impedance matching. The EE transmission S21 exhibited a
5.7 dB roll-off at 67 GHz. Extracted values for nm and Zc are
plotted in Figure 3g,h, respectively. The significant roll-off in
S21 compared to simulations was primarily attributed to
nonideal metal deposition and residual silicon beneath the
electrodes, as indicated by the dark, slightly blurred shadows
near the edge of the backside hole in Figure 3a. The blurring
occurs because the microscope was focused on the electrode
surface rather than the residual silicon. Deviations in the
calculated nm from the simulated values were likely due to
inaccuracies in the backside hole dimensions and the slope of
the positive sidewall caused by deep silicon etching, which may
have narrowed the silicon opening beneath the LT waveguide.
To measure the Vπ of the modulator, a sawtooth wave with a
peak-to-peak voltage of 20 V and a frequency of 100 kHz was
applied to the electrodes. The measured Vπ values at 0 dBm
and 35 dBm optical input power (equivalent to −7 dBm and
28 dBm within the modulator, respectively) are shown in
Figure 4a,b. The 35 dBm optical power was generated using a
tunable laser combined with an erbium-doped fiber amplifier.
The calculated VπL was 3.22 V·cm for both input power levels.
If the modulated light propagated strictly along the Y axis of
the LT crystal, the equivalent VπL would be 2.67 V·cm, closely
matching the simulated result. The insets in Figure 4a,b
indicate a static extinction ratio (ER) and IL of approximately
22 and 2 dB, respectively. This loss is partially attributed to the
alignment error between the electrodes and the waveguide, as
the device was fabricated using a contact lithography system
(EVG620). Further reduction in IL is expected with more
precise alignment techniques. The small-signal EO response
S21 was measured with a 70 GHz photodetector, as shown in
Figure 4c. The 3 dB EO bandwidth was around 55 GHz for
both optical power levels, primarily limited by RF losses, as
shown in Figure 3f. Consistent VπL, ER, loss, and EO
bandwidth results were observed for both low and high optical
power inputs, confirming that the high power (28 dBm in the
waveguide) is well below the material’s optical damage
threshold.
The modulator’s performance for high-speed data trans-
mission at different optical power levels was also evaluated (see
Methods). Figure 4d−g displays EO eye diagrams for on−off
key (OOK) modulation at 80 Gb/s and 100 Gb/s, and four-
level pulse-amplitude modulation (PAM-4) format at 80 Gb/s
(40 Gbaud) and 112 Gb/s (56 Gbaud) under 0 dBm optical
input power (−7 dBm within the modulator). Similar tests
conducted with 35 dBm optical input power are shown in
Figure 4h−k. The eye diagrams remained clear and well-
opened, with slight improvement at higher optical power, likely
due to enhanced signal-to-noise ratio (SNR).
To further demonstrate the high optical power operating
stability of the modulator, EO eye diagram measurements for
the OOK modulation format at 80 Gb/s were performed at
various optical power levels within the modulator, recording
dynamic ER and SNR values. For comparison, a modulator
with a similar T-segment structure, fabricated on X-cut TFLN,
with a coupling loss of 7 dB per grating, was tested under
identical conditions. It is worth noting that a 1 dB front-end
loss was introduced by the polarization controller and
transmission fibers prior to the modulator input.
As shown in Figure 5a, the TFLT modulator exhibits stable
ER and SNR across the entire tested optical power range. In
contrast, while the TFLN modulator initially achieves slightly
higher SNR at low optical powers, both its ER and SNR
progressively degrade as the optical power increases, as shown
in Figure 5b. The slightly lower SNR of the TFLT modulator
compared to the TFLN modulator at low optical powers is
likely due to its relatively lower EE bandwidth, which limits the
rise/fall times and slightly reduces eye diagram clarity. The
insets on the right of Figure 5 show the eye diagrams for both
modulators at each power level, captured with identical
sampling parameters. At powers below 21 dBm, both
Figure 2. (a) Simulated TE optical mode profile and static electrical
field distribution. (b) Simulated RF mode profile at a microwave
frequency of f = 100 GHz. Simulation results for (c) characteristic
impedance Zc, (d) RF effective index nm, (e) RF loss am and (f) EO
bandwidth. The three red dashed lines in (c), (d), and (f) represent
the source/load impedance 50 Ω, optical group index ng, and the −3
dB indicator line, respectively.
ACS Photonics
pubs.acs.org/journal/apchd5
Article
https://doi.org/10.1021/acsphotonics.5c00159
ACS Photonics XXXX, XXX, XXX−XXX
C

<!-- page 4 -->
modulators produce well-defined eye diagrams. However, at 27
dBm, the TFLN modulator exhibits significant eye quality
degradation, including unstable crossing levels. Notably, even
after resetting the operating point via wavelength tuning, the
eye rapidly deterioratesindicating an inherent instability. In
contrast, the TFLT modulator maintains high-quality and
consistent eye diagrams even under the highest tested optical
powers.
This behavior is further reflected in the bit-error rate (BER)
performance. As shown in Figure 6a, the BER of both
modulators initially decreases with increasing optical power,
falling below the KP-4 forward error correction threshold.
With further power increase, the BER reaches a plateau.
However, for the TFLN modulator, when the input exceeds
∼28 dBm, the average output power no longer increases and
becomes unstable (as seen in the inset of Figure 6a and in
Figure 3. Backside view of the fabricated modulator, with a microscope-focused view of (a) the electrode and (b) the chip’s backside surface. (c)
Top view of the entire fabricated device. Detailed top views of the (d) 1 × 2 MMI splitter and (e) grating coupler. (f) Measured electrical reflection
S11 and transmission S21 of the fabricated EO modulator. Deduced (g) RF effective index nm, and (h) characteristic impedance Zc. The three red
dashed lines in (f), (g), and (h) represent the −6.4 dB indicator line, the optical group index ng, and the source/load impedance 50 Ω, respectively.
Figure 4. Measured EO response of the fabricated modulator using a 100 kHz sawtooth wave with a peak-to-peak voltage of 20 V at (a) 0 dBm and
(b) 35 dBm optical input power (equivalent to −7 dBm and 28 dBm within the modulator, respectively). The Vπ values are indicated. Insets display
the measured normalized transmissions, showing the ILs and static ERs of the modulator for both optical input powers. (c) Measured EO
bandwidth of the modulator for both optical input powers. (d−k) Measured EO eye diagrams: the first to fourth columns correspond to OOK
80Gb/s, OOK 100Gb/s, PAM-4 80 Gb/s, and PAM-4 112 Gb/s, respectively. The first row shows results for 0 dBm optical input power, while the
second row shows results for 35 dBm optical input power.
ACS Photonics
pubs.acs.org/journal/apchd5
Article
https://doi.org/10.1021/acsphotonics.5c00159
ACS Photonics XXXX, XXX, XXX−XXX
D

<!-- page 5 -->
Figure 6b), accompanied by a noticeable rise in BER. This
degradation aligns with the eye diagram deterioration
discussed in Figure 5. In contrast, the TFLT modulator
exhibits a linear relationship between input and output power
and maintains consistently low BER even at high input powers.
As shown in Figure 6c,d, the TFLT modulator demonstrates
excellent temporal stability in both output power and BER
over a 30 min period at input powers of 0 dBm and 35 dBm,
confirming its robustness for high-power operation.
In our experiments, no DC bias was applied; instead, the
operating point was adjusted by fine-tuning the wavelength,
leveraging the inherent arm asymmetry of the Mach−Zehnder
interferometer (MZI). This confirms that the observed
instability in the TFLN modulator originates from optical
effects rather than DC drift. At high optical power, PR effects
in the LN waveguide induce space-charge redistribution,
leading to local refractive index variations. These distort the
guided mode and can excite high-order modes,19 ultimately
causing ER degradation, increased IL, and operating point
instability due to index shifts in the phase shifter. In contrast,
the LT waveguide exhibits significantly weaker PR effects and a
higher optical damage threshold, likely due to its higher
photon energy resistance,28 stronger Ta−O bonds,39 and lower
density of vacancies and defect states.40 Notably, stronger PR
effects correspond to a lower optical damage threshold, as less
optical power is required to induce substantial space-charge
accumulation and refractive index modulation. The maximum
input power tested for the TFLT modulator in this work was
limited only by the available optical source output.
■CONCLUSIONS
This paper presents a 7 mm EO modulator based on TFLT
with an X-112°Y crystal cut, utilizing CLTW electrodes with
backside holes. The fabricated modulator demonstrates a
consistent EO bandwidth exceeding 55 GHz, a large static ER
of over 20 dB, and Vπ of 4.6 V under both 0 dBm and 35 dBm
optical input powers (equivalent to −7 dBm and 28 dBm
within the modulator, respectively). Successful data trans-
missions are achieved using OOK modulation at 100 Gb/s and
PAM-4 at 112 Gb/s, under both input power levels. Compared
to TFLN modulators, the TFLT modulator exhibits signifi-
cantly improved stability, as evidenced by consistent ER, SNR,
and BER performance under increasing optical power. Long-
term stability of output power and BER measurements at high
input power further confirm the device’s robustness. This
enhanced stability is attributed to TFLT’s higher optical
damage threshold and reduced PR effect, ensuring stable
operation point near quadrature. These findings highlight the
strong potential of TFLT modulators for next-generation high-
power photonic applications, including nonlinear optics,
microwave photonics, and THz communications.
■METHODS
Fabrication. First, the MZI patterns were defined via
electron beam lithography on a Ma-N 2403 resist, followed by
inductively coupled plasma (ICP) dry etching using pure Ar+
gas. A series of chemical cleaning steps were employed to
remove redeposition and residual resist. Next, a 900 nm thick
SiO2 cladding layer was deposited using plasma-enhanced
chemical vapor deposition (PECVD). The loaded T-segment
electrodes and unloaded electrodes were then deposited via
electron-beam evaporation and a lift-off process. The chip was
then thinned to approximately 250 μm using grinding and
polishing equipment. Backside holes were patterned, and the
silicon substrate beneath the modulation section was removed
using a double-side mask aligner and anisotropic deep reactive
ion etching (RIE) of silicon until the BOX layer was reached.
Measurements for Eye Diagrams and BERs. An
arbitrary waveform generator (Micram DAC10004) combined
with a clock source (Agilent E8257D) was used to generate
OOK and PAM-4 signals. The signal was amplified by an RF
amplifier (SHF 807c) and fed to the device through an RF
probe (GGB 67A). Another probe was used to provide an off-
chip 50 Ω termination for the electrode. The modulated
Figure 5. Dynamic ER and SNR values at various optical power levels
within the modulator, along with the corresponding eye diagrams for
each power point for (a) the TFLT-based modulator and (b) the
TFLN-based modulator.
Figure 6. (a) Measured BER as a function of received optical power
for 80 Gb/s OOK signals. The inset shows the correlation between
the input optical power and the average modulated optical power
detected at the receiver. Stability of the modulated average output
optical power over time for (b) the TFLN-based and (c) the TFLT-
based modulators, where a small fraction (5%) of the output is tapped
for monitoring. (d) BER stability of TFLT-based modulators over
time under input optical powers of 0 dBm and 35 dBm.
ACS Photonics
pubs.acs.org/journal/apchd5
Article
https://doi.org/10.1021/acsphotonics.5c00159
ACS Photonics XXXX, XXX, XXX−XXX
E

<!-- page 6 -->
optical signal was detected using a preamplified receiver,
consisting of a variable optical attenuator, an erbium-doped
fiber amplifier (EDFA), and a high-speed PD (Finisar
BPDV3320R). The output signal was recorded using either a
digital sampling oscilloscope (Agilent 81600D) for eye
diagrams or a real-time oscilloscope (Tektronix 73304D) for
offline digital signal processing (DSP). The offline DSP
includes down-sampling, timing phase synchronization, and a
25-tap least-mean-square equalizer.
■AUTHOR INFORMATION
Corresponding Authors
Changjian Guo −Guangdong Provincial Key Laboratory of
Optical Information Materials and Technology, South China
Academy of Advanced Optoelectronics, South China Normal
University, Guangzhou 510006, China; National Center for
International Research on Green Optoelectronics, South
China Normal University, Guangzhou 510006, China;
Email: changjian.guo@coer-scnu.org
Kaixuan Chen −Guangdong Provincial Key Laboratory of
Optical Information Materials and Technology, South China
Academy of Advanced Optoelectronics, South China Normal
University, Guangzhou 510006, China; National Center for
International Research on Green Optoelectronics, South
China Normal University, Guangzhou 510006, China;
orcid.org/0000-0002-2538-6793; Email: chenkaixuan@
m.scnu.edu.cn
Liu Liu −State Key Laboratory of Extreme Photonics and
Instrumentation, College of Optical Science and Engineering,
International Research Center for Advanced Photonics,
Zhejiang University, Hangzhou 310058, China;
orcid.org/0000-0002-3651-544X; Email: liuliuopt@
zju.edu.cn
Authors
Haohua Wang −Guangdong Provincial Key Laboratory of
Optical Information Materials and Technology, South China
Academy of Advanced Optoelectronics, South China Normal
University, Guangzhou 510006, China
Ao Cui −Guangdong Provincial Key Laboratory of Optical
Information Materials and Technology, South China
Academy of Advanced Optoelectronics, South China Normal
University, Guangzhou 510006, China
Bin Chen −State Key Laboratory of Extreme Photonics and
Instrumentation, College of Optical Science and Engineering,
International Research Center for Advanced Photonics,
Zhejiang University, Hangzhou 310058, China
Ziliang Ruan −State Key Laboratory of Extreme Photonics
and Instrumentation, College of Optical Science and
Engineering, International Research Center for Advanced
Photonics, Zhejiang University, Hangzhou 310058, China
Complete contact information is available at:
https://pubs.acs.org/10.1021/acsphotonics.5c00159
Funding
National Natural Science Foundation of China (62135012),
Leading Innovative and Entrepreneur Team Introduction
Program of Zhejiang (2021R01001), Basic and Applied Basic
Research
Foundation
of
Guangdong
Province
(2024A1515011710), and Guangdong Provincial Key Labo-
ratory of Optical Information Materials and Technology
(2023B1212060065).
Notes
The authors declare no competing financial interest.
■REFERENCES
(1) Zhu, D.; Shao, L.; Yu, M.; Cheng, R.; Desiatov, B.; Xin, C. J.;
Hu, Y.; Holzgrafe, J.; Ghosh, S.; Shams-Ansari, A.; Puma, E.; Sinclair,
N.; Reimer, C.; Zhang, M.; Lončar, M. Integrated photonics on thin-
film lithium niobate. Adv. Opt. Photon. 2021, 13 (2), 242−352.
(2) Li, Y.; Sun, M.; Miao, T.; Chen, J. Towards High-Performance
Pockels Effect-Based Modulators: Review and Projections. Micro-
machines 2024, 15 (7), 865.
(3) Wang, C.; Zhang, M.; Stern, B.; Lipson, M.; Lončar, M.
Nanophotonic lithium niobate electro-optic modulators. Opt. Express
2018, 26 (2), 1547−1555.
(4) Yang, F.; Fang, X.; Chen, X.; Zhu, L.; Zhang, F.; Chen, Z.; Li, Y.
Monolithic thin film lithium niobate electro-optic modulator with
over 110 GHz bandwidth. Chin. Opt. Lett. 2022, 20 (2), 022502.
(5) Xu, M.; Zhu, Y.; Pittalá, F.; Tang, J.; He, M.; Ng, W. C.; Wang,
J.; Ruan, Z.; Tang, X.; Kuschnerov, M.; Liu, L.; Yu, S.; Zheng, B.; Cai,
X. Dual-polarization thin-film lithium niobate in-phase quadrature
modulators for terabit-per-second transmission. Optica 2022, 9 (1),
61−62.
(6) Chen, G.; Chen, K.; Gan, R.; Ruan, Z.; Wang, Z.; Huang, P.; Lu,
C.; Lau, A. P. T.; Dai, D.; Guo, C.; Liu, L. High performance thin-film
lithium niobate modulator on a silicon substrate using periodic
capacitively loaded traveling-wave electrode. APL Photonics 2022, 7
(2), 026103.
(7) Boes, A.; Chang, L.; Langrock, C.; Yu, M.; Zhang, M.; Lin, Q.;
Lončar, M.; Fejer, M.; Bowers, J.; Mitchell, A. Lithium niobate
photonics: Unlocking the electromagnetic spectrum. Science 2023,
379 (6627), No. eabj4396.
(8) Ivankovski, Y.; Mendlovic, D. High-rate−long-distance fiber-
optic communication based on advanced modulation techniques.
Appl. Opt. 1999, 38 (26), 5533−5540.
(9) Sun, D.; Zhang, Y.; Wang, D.; Song, W.; Liu, X.; Pang, J.; Geng,
D.; Sang, Y.; Liu, H. Microstructure and domain engineering of
lithium niobate crystal films for integrated photonic applications.
Light: sci. Appl. 2020, 9 (1), 197.
(10) Marpaung, D.; Yao, J.; Capmany, J. Integrated microwave
photonics. Nat. Photonics 2019, 13 (2), 80−90.
(11) Feng, H.; Ge, T.; Guo, X.; Wang, B.; Zhang, Y.; Chen, Z.; Zhu,
S.; Zhang, K.; Sun, W.; Huang, C.; Yuan, Y.; Wang, C. Integrated
lithium niobate microwave photonic processing engine. Nature 2024,
627 (8002), 80−87.
(12) Burla, M.; Hoessbacher, C.; Heni, W.; Haffner, C.; Fedoryshyn,
Y.; Werner, D.; Watanabe, T.; Massler, H.; Elder, D. L.; Dalton, L. R.;
Leuthold, J. 500 GHz plasmonic Mach-Zehnder modulator enabling
sub-THz microwave photonics. APL Photonics 2019, 4 (5), 056106.
(13) Feng, H.; Zhang, K.; Sun, W.; Ren, Y.; Zhang, Y.; Zhang, W.;
Wang, C. Ultra-high-linearity integrated lithium niobate electro-optic
modulators. Photon. Res. 2022, 10 (10), 2366−2373.
(14) Cox, C. H., Analog Optical Links: Theory And Practice;
Cambridge University Press, 2006.
(15) Glass, A. The Photorefractive Effect. Opt. Eng. 1978, 17 (5),
175470.
(16) Xu, Y.; Shen, M.; Lu, J.; Surya, J. B.; Sayem, A. A.; Tang, H. X.
Mitigating photorefractive effect in thin-film lithium niobate
microring resonators. Opt. Express 2021, 29 (4), 5497−5504.
(17) Celik, O. T.; Yousry Ammar, N.; Park, T.; Stokowski, H. S.;
Multani, K. K. S.; Hwang, A. Y.; Gyger, S.; Guo, Y.; Fejer, M. M.;
Safavi-Naeini, A. H. Roles of temperature, materials, and domain
inversion in high-performance, low-bias-drift thin film lithium niobate
blue light modulators. Opt. Express 2024, 32 (21), 36160−36170.
(18) Kong, Y.; Bo, F.; Wang, W.; Zheng, D.; Liu, H.; Zhang, G.;
Rupp, R.; Xu, J. Recent Progress in Lithium Niobate: Optical
Damage, Defect Simulation, and On-Chip Devices. Adv. Mater. 2020,
32 (3), 1806452.
(19) Villarroel, J.; Carnicero, J.; Luedtke, F.; Carrascosa, M.; García-
Cabañes, A.; Cabrera, J. M.; Alcazar, A.; Ramiro, B. Analysis of
ACS Photonics
pubs.acs.org/journal/apchd5
Article
https://doi.org/10.1021/acsphotonics.5c00159
ACS Photonics XXXX, XXX, XXX−XXX
F

<!-- page 7 -->
photorefractive optical damage in lithium niobate: application to
planar waveguides. Opt. Express 2010, 18 (20), 20852−20861.
(20) Xue, M.; Yan, X.; Wu, J.; Ge, R.; Yuan, T.; Chen, Y.; Chen, X.
On-chip ultraviolet second-harmonic generation in lithium-tantalate
thin film microdisk. Chin. Opt. Lett. 2023, 21 (6), 061902.
(21) Zhang, M.; Wang, C.; Kharel, P.; Zhu, D.; Lončar, M.
Integrated lithium niobate electro-optic modulators: when perform-
ance meets scalability. Optica 2021, 8 (5), 652−667.
(22) Valdez, F.; Mere, V.; Wang, X.; Boynton, N.; Friedmann, T. A.;
Arterburn, S.; Dallo, C.; Pomerene, A. T.; Starbuck, A. L.; Trotter, D.
C.; Lentine, A. L.; Mookherjea, S. 110 GHz, 110 mW hybrid silicon-
lithium niobate Mach-Zehnder modulator. Sci. Rep. 2022, 12 (1),
18611.
(23) Kong, Y.; Liu, S.; Xu, J. Recent Advances in the Photorefraction
of Doped Lithium Niobate Crystals. Materials 2012, 5 (10), 1954−
1971.
(24) Schwesyg, J. R.; Falk, M.; Phillips, C. R.; Jundt, D. H.; Buse, K.;
Fejer, M. M. Pyroelectrically induced photorefractive damage in
magnesium-doped lithium niobate crystals. J. Opt. Soc. Am. B 2011, 28
(8), 1973−1987.
(25) Furukawa, Y.; Kitamura, K.; Takekawa, S.; Miyamoto, A.;
Terao, M.; Suda, N. Photorefraction in LiNbO3 as a function of [Li]/
[Nb] and MgO concentrations. Appl. Phys. Lett. 2000, 77 (16),
2494−2496.
(26) Valdez, F.; Mere, V.; Mookherjea, S. 100 GHz bandwidth, 1 V
integrated electro-optic Mach-Zehnder modulator at near-IR wave-
lengths. Optica 2023, 10 (5), 578−584.
(27) Celik, O. T.; Ammar, N. Y.; Stokowski, H.; Park, T.; Safavi-
Naeini, A.In Visible Electro-optic Modulator Array in Thin-Film
Lithium Niobate for Atomic Qubit Control. In Frontiers In Optics +
Laser Science 2023 (Fio, LS); Optica Publishing Group: Tacoma, WA,
2023. .
(28) Wang, C.; Li, Z.; Riemensberger, J.; Lihachev, G.; Churaev, M.;
Kao, W.; Ji, X.; Zhang, J.; Blesin, T.; Davydova, A.; Chen, Y.; Huang,
K.; Wang, X.; Ou, X.; Kippenberg, T. J. Lithium tantalate photonic
integrated circuits for volume manufacturing. Nature 2024, 629
(8013), 784−790.
(29) Wang, C.; Fang, D.; Zhang, J.; Kotz, A.; Lihachev, G.; Churaev,
M.; Li, Z.; Schwarzenberger, A.; Ou, X.; Koos, C.; Kippenberg, T. J.
Ultrabroadband thin-film lithium tantalate modulator for high-speed
communications. Optica 2024, 11 (12), 1614−1620.
(30) Yan, X.; Liu, Y. A.; Ge, L.; Zhu, B.; Wu, J.; Chen, Y.; Chen, X.
High optical damage threshold on-chip lithium tantalate microdisk
resonator. Opt. Lett. 2020, 45 (15), 4100−4103.
(31) Nishi, H.; Tsuchizawa, T.; Segawa, T.; Matsuo, S.In Low-loss
lithium tantalate on insulator waveguide towards on-chip nonlinear
photonics, 2022. In 2022 International Conference on Photonics in
Switching and Computing (PSC); IEEE: 2022, pp 1−3.
(32) Senouci, K.; Mansour, D.; Abderrahmane, A. Electro-optic
properties of one-dimensional (1D) nonlinear perfect photonic
crystals based on Lithium tantalate layer. Optik 2022, 265, 169537.
(33) Yu, J.; Ruan, Z.; Xue, Y.; Wang, H.; Gan, R.; Gao, T.; Guo, C.;
Chen, K.; Ou, X.; Liu, L. Tunable and stable micro-ring resonator
based on thin-film lithium tantalate. APL Photonics 2024, 9 (3),
036115.
(34) Wang, H.; Xing, X.; Ruan, Z.; Yu, J.; Chen, K.; Ou, X.; Liu, L.
Optical switch with an ultralow DC drift based on thin-film lithium
tantalate. Opt. Lett. 2024, 49 (18), 5019−5022.
(35) Powell, K.; Li, X.; Assumpcao, D.; Magalhães, L.; Sinclair, N.;
Lončar, M. DC-stable electro-optic modulators using thin-film lithium
tantalate. Opt. Express 2024, 32 (25), 44115−44122.
(36) Wang, Z.; Chen, G.; Ruan, Z.; Gan, R.; Huang, P.; Zheng, Z.;
Lu, L.; Li, J.; Guo, C.; Chen, K.; Liu, L. Silicon−Lithium Niobate
Hybrid Intensity and Coherent Modulators Using a Periodic
Capacitively Loaded Traveling-Wave Electrode. ACS Photonics
2022, 9 (8), 2668−2675.
(37) Wang, M.; Qi, L.; Wang, H.; Ruan, Z.; Chen, G.; Chen, B.;
Gong, S.; Chen, K.; Liu, L. Robust thin-film lithium niobate
modulator on a silicon substrate with backside holes. Chin. Opt.
Lett. 2024, 22 (5), 050601.
(38) Bahadori, M.; Goddard, L. L.; Gong, S. Fundamental electro-
optic limitations of thin-film lithium niobate microring modulators.
Opt. Express 2020, 28 (9), 13731−13749.
(39) Gruber, M.; Konetschnik, R.; Popov, M.; Spitaler, J.; Supancic,
P.; Kiener, D.; Bermejo, R. Atomistic origins of the differences in
anisotropic fracture behaviour of LiTaO3 and LiNbO3 single crystals.
Acta Mater. 2018, 150, 373−380.
(40) Grachev, V. G.; Malovichko, G. I. Structures of Impurity
Defects in Lithium Niobate and Tantalate Derived from Electron
Paramagnetic and Electron Nuclear Double Resonance Data. Crystals
2021, 11 (4), 339.
ACS Photonics
pubs.acs.org/journal/apchd5
Article
https://doi.org/10.1021/acsphotonics.5c00159
ACS Photonics XXXX, XXX, XXX−XXX
G

