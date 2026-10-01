---
paper_id: zheng2026
source_url: https://arxiv.org/abs/2605.28971
doi: 
license: CC-BY-NC-SA-4.0
sha256: c171be52e020a7a9c30ec438b89ec79a08f9c153688dfa232e497259b24c51d4
pages: 15
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Micro-Transfer Printing of Lithium Niobate on 200 mm Silicon Photonics: A
High-Speed Heterogeneous Wafer-Scale Platform
Xiujun Zheng,1, 2, a) Suzanne Bisschop,1, 2 Arno Moerman,1, 2 Margot Niels,1, 2 Ewoud
Vissers,1, 2 Athina Papadopoulou,1, 2 Philip Ekkels,1, 2 Patrick Nenezic,1, 2 Simone
Atzeni,1, 2 Elif Ozceri,1, 2 Tiernan McCaughery,1, 2 Ali Uzun,1, 2 Ye Chen,1, 2 Laurens
Bogaert,1, 2 Nishant Singh,3, 2 Sandeep Seema Saseendran,2 Sofie Janssen,2 Natarajan
Rajasekaran,2 Sadhishkumar Balakrishnan,2 Philippe Absil,2 Günther Roelkens,1, 2 Bart
Kuyken,1, 2 Sarah Uvin,1, 2 and Maximilien Billet1, 2
1)Department of Information Technology (INTEC) - Photonics Research Group,
Ghent University–imec, Technologiepark Zwijnaarde 126, 9052 Ghent,
Belgium
2)imec, Kapeldreef 75, 3001 Leuven, Belgium
3)Department of Information Technology (INTEC) - IDLab,
Ghent University–imec, Technologiepark Zwijnaarde 126, 9052 Ghent,
Belgium
(Dated: 16 June 2026)
The rapid growth of artificial intelligence (AI) and other data-center applications is driving
the demand for photonic interconnects that combine high-speed with low energy consump-
tion, making scalability a critical requirement. Micro-transfer printing (MTP) has emerged
as a promising technique for the wafer-scale heterogeneous integration of thin-film lithium
niobate (TFLN) onto silicon photonics (SiPho) platforms. Here, we demonstrate hetero-
geneous SiPho–TFLN integration across four full 200 mm wafers with a 3σ placement
accuracy down to 420 nm and a printing yield of >95%. Low insertion loss <2 dB over 600
phase modulators (forming 300 amplitude modulators) is achieved. A half-wave voltage
of 4 V in push-pull configuration, and high-speed modulation with a bandwidth >70 GHz
are demonstrated on a subset of tested devices.
a)xiujun.zheng@ugent.be
1
arXiv:2605.28971v2  [physics.optics]  15 Jun 2026

<!-- page 2 -->
I.
INTRODUCTION
The growing demand for low energy consumption, dense integration and fast components in
data centers is pushing the performance requirements for photonic integrated circuits, and es-
pecially for optical modulators that enable optical data-communication links1. Emerging AI and
data-center interconnect roadmaps are pushing per-lane data rates toward hundreds of Gbit/s, plac-
ing increasing pressure on modulator bandwidth, drive voltage, linearity and energy efficiency.
Conventional depletion-based silicon (Si) modulators face trade-offs in these metrics, motivat-
ing the exploration of new solutions. As an alternative, several technologies are considered to
reach this milestone, such as the use of BTO2, plasmonics3–5, graphene,6,7, electro-absorption
modulators8 and electro-optics materials. TFLN has emerged as a promising platform for high-
performance modulators due to its intrinsically low optical loss, large electro-optic coefficient,
and capability to support high-speed operations9. It is therefore seen as a solution to surpass
the performance limitations of conventional SiPho platforms10. However, TFLN is not compati-
ble with CMOS fabrication, primarily because of lithium contamination concerns within silicon
foundries, which limits circuit complexity and the production volume of TFLN-based fabrication
processes11.
To overcome these limitations, heterogeneous integration offers a viable approach for incorpo-
rating TFLN onto SiPho, combining the advantages of both materials on the same platform. Wafer-
to-wafer and die-to-wafer bonding have produced high-performance SiPho-TFLN platforms and
remain important integration routes12–15. This method, demonstrated in the literature, provides
platforms that can reach to the standards of the optical data-communication16. However, blanket
bonding consumes TFLN over the full bonded area, offers limited selectivity, and generally im-
poses strong constraints on target-wafer planarity and process sequencing. To address this weak-
ness, micro-transfer printing (MTP) has been evaluated17–19 and provides a complementary route
in which processed TFLN coupons are placed only at the required locations on a pre-fabricated
silicon photonics wafer. Furthermore, the integration in local cladding apertures offers the possi-
bility of the use of complex advanced silicon stacks In this work, we demonstrate the wafer-scale
integration of pre-patterned TFLN modulators on full 200-mm silicon photonics wafers using
MTP. A total of 4 wafers with more than 600 phase modulators, corresponding to 300 amplitude
modulators, are successfully transferred and integrated across the SiPho wafers. The fabricated
heterogeneous devices are 7-mm-long push-pull Mach–Zehnder modulators (MZMs) in ground-
2

<!-- page 3 -->
signal-signal-ground (GSSG) electrode configuration. This demonstration highlights the scalabil-
ity and the potential for volume production of high-speed SiPho-TFLN heterogeneous modulators
fabricated with MTP.
FIG. 1: Comparison between previously reported designs and the optimised design presented in
this work. a) The transfer printed TFLN device consists in a material slab, not allowing for
optimised confinement of the light. b) The transfer printed devices in this work are patterned
prior to the integration, enabling an optimised mode confinement. c) Cross-section of the design
from a). d) Microscope picture of a chip made following the design from a). e) Cross-section of
the design from b). f) Microscope picture of a chip made following the design from b).
First, a statistical analysis of wafer-scale (200-mm) integration of patterned TFLN devices (in-
cluding a TFLN waveguide), demonstrating high transfer yield and low optical insertion loss, are
presented. As a proof of concept, back-end processing of metal electrodes is performed on a subset
of the transferred devices, enabling reproducible fabrication of electro-optic modulators with low
Vπ. The radio-frequency (RF) performance of representative devices are also characterised.
II.
HETEROGENEOUS THIN-FILM LITHIUM NIOBATE ON SILICON PHOTONICS
USING MICRO-TRANSFER PRINTING
MTP is an emerging method for heterogeneous integration. The technology, licensed from
X-Celeprint Ltd., combines die-level assembly with wafer-scale processing. Thin-film devices
(“coupons”) are first fabricated in dense arrays on a source wafer. They are released by selec-
3

<!-- page 4 -->
tively etching away a sacrificial layer. Once suspended, an elastomeric stamp, generally made
out of polydimethylsiloxane (PDMS), is used to retrieve the devices. The coupons are printed
onto a target wafer and bonded using adhesive or direct bonding. MTP offers high material ef-
ficiency, known-good-die integration, and excellent scalability through parallel printing, enabling
high throughput at sub-micron alignment accuracy. Its greatest strength lies in its broad mate-
rial and process compatibility, allowing heterogeneous integration of multiple material systems
without compromising individual fabrication flows. The integration is performed at the back-end
of SiPho processing, supporting both efficient optical coupling and electrical interconnection via
redistribution layers.
Despite these advantages, MTP remains at an early stage of commercialisation, with challenges
related to pritning yield, long-term reliability, and supply-chain maturity. Nonetheless, extensive
research demonstrations including the integration of III-V lasers, amplifiers, modulators, photo-
diodes, and thin-film electro-optic devices highlight MTP’s strong potential for enabling complex,
high-performance heterogeneous photonic integrated circuits20.
III.
DEMONSTRATION OF INTEGRATION OF TFLN ON 200-MM SILICON
PHOTONICS WAFERS
The modulator demonstrated in this work is a hybrid SiN/TFLN unbalanced Mach–Zehnder
modulator (MZM), where silicon waveguides are used for routing and SiN/TFLN hybrid waveg-
uides form the electro-optic modulating arms. The MZM consists of TFLN electro-optically active
arms combined with a passive silicon photonic circuit. The passive components, including grating
couplers, Si and SiN waveguides and multi-mode interferometer (MMI) splitters, are designed us-
ing a standard process design kit (PDK) and fully fabricated prior to active integration. The TFLN
devices are prepared and suspended before being integrated onto the silicon photonics wafer. This
integration is done using micro-transfer printing with a 50-nm intermediate bonding layer. During
transfer printing, the LN crystal orientation in the two modulator arms is intentionally flipped,
enabling push–pull modulation. Optical coupling from Si waveguides into the hybrid SiN/TFLN
modes is achieved through a bilayer Si–SiN adiabatic transition, rendering a low-loss transition.
Electro-optic modulation is provided by the Pockels effect in TFLN, while the SiN layer ensures
low-loss optical propagation. Finally, metal electrodes are defined in a post-processing step. Fig.1
compares previously reported device designs21 and the upgraded design. In previous designs, as
4

<!-- page 5 -->
illustrated in Fig.1 a) the TFLN is a slab and the mode is a hybrid SiN/TFLN mode. In contrast,
the current design incorporates a TFLN taper and an etched waveguide, as seen in Fig.1 b). The
optical mode confinement is stronger, allowing for closer electrode spacing. This enables for a
lower Vπ, better optical transition and improved mode confinement along the propagation direc-
tion. Furthermore, this concept is compatible with designs asking for full coupling in a TFLN
waveguide and is not limited to SiN/TFLN hybrid modes, even if this was not demonstrated in this
work. Details on the cross-sections of both designs and the corresponding fabricated modulators
are shown in Fig.1(c-f).
FIG. 2: Overview of the integration of TFLN on a 200-mm SiPho wafer using MTP. a)
Wafer-scale printing tool compatible with target wafers of 200 mm and 300 mm. b) Overview of
a 200-mm SiPho wafer, which is completely populated with TFLN and zoomed images of a
reticle and a TFLN printed device. c) Statistical printing alignment data, extracted from four
wafers. d) Distribution wafer map of Y misalignment data across a wafer. The distribution of the
alignment focuses on the vertical (y) direction since this is the relevant metric for low loss optical
coupling e) Gaussian Kernel Density Estimation (KDE) distribution data for one wafer
5

<!-- page 6 -->
The approach presented in this work is compatible with the integration of TFLN on 200-mm
(and 300-mm) wafers using a commercial MTP tool, the ASMPT Amicra NANO, as depicted in
Fig.2 a). The used approach enables accurate and repeatable wafer-scale transfer of TFLN from
a source wafer (providing the TFLN suspended coupons) to a target wafer (the SiPho platform).
The entire process is enabled by a fully automated tool that integrates coupon tracking, stamp
identification, printing location control, and misalignment pattern recognition. Each full cycle
duration is around 1 min, for single coupon printing operation as well as for the transfer of arrays
of multiple devices, including the coupon recognition, location to print identification, picking,
alignment, printing, post-printing measurement and stamp cleaning. Demonstrations up to 28
coupons of 1-mm21 and 16 coupons of 7-mm have been performed, where the printing duration
and alignment level were confirmed. Here we provide data by single coupon printing on four 200-
mm SiPho wafers. An example of a chip containing four MZMs after TFLN integration on the
wafer is presented in Fig.2 b). The SEM picture highlights the patterned TFLN waveguide. All
back-end of line processing following the CMOS wafer production has been implemented in the
imec-Gent University pilot line "TRANSVERSE"22.
A dedicated full-wafer calibration is required to capture systematic tool offsets and stabilize the
tool, thereby allowing global optimization of the tool settings. Wafer No. 1 was used as a calibra-
tion wafer to set picking and printing recipes and evaluate alignment accuracy across all coupons.
The alignment 3σ value was obtained from automated post-printing measurements provided by
the tool. Additional microscope inspection was carried out on randomly chosen devices to validate
the results. The readout alignment of the commercial transfer-printing tool was not optimised and
Wafer No. 1 has been used as a printing test platform (3σ ≈770 nm) to confirm the stability of
the printing operation. This wafer also demonstrates the minimum effort to set a new printing
procedure. In total, four successive wafers are printed using a similar printing recipe architecture.
As shown in Fig.2 c), the misalignment distributions along both the x and y axes follow Gaussian
profiles centred at zero. Table I summarises the extracted σ and 3σ values for the four processed
wafers. Progressive improvement in the σ values is observed as the printing operation is refined
by playing with the pattern recognition parameters, to finally reach the representative value given
by wafer No. 4 exhibiting a value of 3σ below 500 nm. Fig.2 d) shows the spatial distribution
of alignment error for the y-direction (the most critical for the coupling) across a representative
wafer, demonstrating uniform printing performance over the full wafer area and Fig.2 e) shows
the Gaussian distribution. In total, 600 pre-patterned coupons, incorporating TFLN waveguides,
6

<!-- page 7 -->
TABLE I: 3σ data in y axis for 4 wafers populated with TFLN with MTP
Wafer
Printing success/printing trials
Printing yield
σ in Y (nm)
3σ in Y (nm)
Wafer 1
260/272
96%
259
770
Wafer 2
108/114
95%
324
630
Wafer 3
145/152
95%
289
610
Wafer 4
234/240
98%
165
420
were printed with a high transfer yield of >95% across all wafers, with the yield defined as the
number of successfully printed coupons without any breakage. The failures can be explained by
remaining inhomogeneities in the source and target preparation. The printing process could reach
a higher yield by the implementation of automatic inspection in all preparation steps, hence util-
ising the know-good-die concept attributed to MTP. These results demonstrate stable wafer-scale
coupon preparation and printing. Implementation in industrial fabrication environments would
enable larger statistical datasets and further validation of the long-term process stability.
IV.
OPTICAL CHARACTERISATION OF THE DEVICES
The optical insertion loss (IL) of each device across the wafers must be characterised to evaluate
process uniformity and device yield. The ILs are excluding the silicon routing from the CMOS
platform as described in section III. As photonic integration scales to larger wafer formats and
higher device densities, device variability becomes an increasingly important consideration. High-
throughput measurement capabilities are therefore required to generate large datasets that provide
statistical insight into device performance and process uniformity.
A wafer-scale characterisation setup, based on an MPI TS2000-IFE fully automated probe
station, was used, where the optical characterisation was performed using an EXFO CTP10 testing
platform in combination with an EXFO T200S tunable laser source and an optical power meter
module (1936-R Newport). A schematic representation of the setup is provided in Fig. 3 a).
The measurement setup enables fast and repeatable characterisation of photonic devices across
an entire wafer, mandatory for wafer level characterisation. An example of a wavelength sweep
around 1310 nm performed on a representative MZM prior to metallisation, is shown in Fig. 3 b).
The measured transmission spectrum is normalised and excludes the grating coupler and device
7

<!-- page 8 -->
ILs which are not contributing to the integrated phase shifters. The extinction ratio exceeding 30
dB for this device demonstrated the quality of the MZM interferometer.
Fiber array and pigtails
Thermal chuck
Automated wafer probing
Soware
Laser
Powermeter
a)
c)
b)
FIG. 3: a) Characterisation setup for 200-mm wafers, based on an automated wafer prober. b)
Example of a normalised MZM transmission spectrum. c) Statistical insertion loss of all four
wafers.
The ILs are measured for all printed modulators on the four wafers and are presented in Fig. 3
c). To obtain the ILs, a reference waveguide with a given length is first measured, after which the
corresponding MZMs are characterised. The observed spread in the mean value of the ILs for each
wafer is correlated with the misalignment in table I. An increase in the printing alignment accuracy
(lower 3σ) provides lower ILs. The spread of the IL values within a given wafer is attributed to
multiple factors, primarily coupon misalignment and measurement uncertainty. This uncertainty
is partly explained by the difficulty in extracting very low ILs from short devices (7-mm long). All
the devices presented here provide decent ILs, with a mean value below 2 dB. By using separated
test structures based on ring resonators, the precision of the measurements could be improved.
8

<!-- page 9 -->
FIG. 4: Electro-optics characterisation. a) Picture of a 200-mm wafer after RDL back-end
processing. b-c) Zoomed pictures on a device and on a chip. d) RF and photodiode signal as a
function of time. e) Vπ data analysis. f) Vπ as a function of the electrode gaps. g-h) EOE
measurement of four MZMs on a chip for 500 nm and 1 µm thick Al electrodes
V.
ELECTRO-OPTIC CHARACTERISATION OF THE DEVICES
In addition to passive optical characterisation, wafer-level Vπ measurements were also per-
formed to evaluate the electro-optic modulation efficiency of the transferred TFLN devices. The
back-end redistribution layer (RDL) processing was carried out on a full 200-mm wafer using UV-
lithography, metal deposition and lift-off. This study is limited to one wafer because of the facility
constraints. The value of the electrode gap design varies between 4 and 6 µm and aluminium (Al)
electrodes with a thickness of 500 nm are used. A thin layer of 100 nm oxide is placed in between
the RDL and the TFLN as a spacer. Fig. 4 a) depicts the wafer after the fabrication of the elec-
trodes, while Fig. 4 b-c) presents an SEM image and an optical microscope picture of the devices.
9

<!-- page 10 -->
The devices are driven by an RF signal generator, providing a triangular wave at a frequency of
100 kHz. The optical signal is modulated via the EO effect and recorded by a photodetector.
The raw signals of four MZMs from the same chip are presented in Fig. 4 d), where the signal
of both the RF signal generator and the photodetector are presented, as a function of time. The
four devices show very similar behavior. In order to extract the Vπ of the amplitude modulators,
the photodiode signal is plotted as a function of the RF signal for a full modulation period. The
analysis of the data from Fig. 4 d) is presented in Fig. 4 e). In this case, the Vπ is around 4 V and
corresponds to an electrode gap of 4 µm. Results of the Vπ value as a function of the electrode gap
are also presented in Fig. 4 f). As expected from their linear relationship, a wider electrode gap
results in a higher Vπ, confirming the linear dependence between these two parameters. However,
with the current technology used to process the RDLs, the electrode misalignment is around 1 µm,
making it difficult to stay at low ILs after metalisation. On average, an extra 2 dB IL is observed at
this step. The implementation of alternative RDL processing approaches (use of a stepper, print-
ing TFLN devices with pre-defined electrodes or using buried electrodes23) could help reduce the
additional IL while further decreasing the electrode gap and the Vπ. Nevertheless, this proof of
concept shows the possibility of making active EO devices based on MTP at a wafer level.
In this platform, the substrate is made with a high resistivity silicon, allowing for very high
speed operations (>100 GHz) and this property is evaluated on a subset of devices. For this ex-
periment, the RDLs are fabricated on two extra chips. The deposited Al thickness is 500 nm on
the first one and 1 µm on the second. While the Vπ is negligibly affected by the value of the
metal thickness, the electric to optic to electric (EOE) bandwidth (BW) differs. As observed in
Fig. 4 g-h), the bandwidth is limited to 20-30 GHz using a thin 500 nm thick RDL, while the BW
is extended to 70 GHz+ for the option with 1 µm Al. The exact BW is expected to be around
90 GHz, but the current experiment is limited by the measurement tools. Further design of the
device architecture would allow for improvement in the performance, as presented in the work of
P. Nenezic et al.24, taking into account realistic process variability stemming from experimental
statistical analysis, extracted from the presented results.
VI.
CONCLUSION
This work demonstrates the scalability of micro-transfer printing for wafer-scale integration
of TFLN modulators on 200-mm silicon photonics platforms. A printing yield of >95% and a
10

<!-- page 11 -->
3-sigma alignment accuracy below 500 nm enable efficient optical coupling, as evidenced by an
insertion loss lower than 2 dB, demonstrated over 4 wafers. Wafer-scale measurements show good
uniformity across the wafers, confirming the robustness of the integration process. Wafer-level
half-wave voltage characterisation further validates the electro-optic performance and compati-
bility with automated testing. The EO effect provided by TFLN is not affected by the transfer
printing process, and devices with 70 GHz+ BW, made from a subset of fabricated wafers, are
demonstrated. Overall, the results highlight micro-transfer printing as a viable approach for high-
volume integration of high-performance TFLN high-speed modulators within CMOS-compatible
silicon photonics platforms. Recent demonstrations of the integration of CMOS electronics cir-
cuits (EIC) using MTP25 and co-integration of heterogeneous MZMs with electronic drivers and
transimpedance amplifiers16, are paving the way for the next generation of optical interconnects
using the presented technology. Furthermore, several demonstrations of heterogeneous integra-
tion based on the use of lithium tantalate as an alternative material have been demonstrated26 .
The proposed methodology based on MTP is also compatible with this material27,28, allowing for
high-power applications or working with short wavelengths down to the UV-range29.
ACKNOWLEDGMENTS
We would like to thank ASMPT Amicra team for their valuable input for the tool utilisation.
The authors would like to also thank the imec-Leuven teams providing the silicon and silicon
nitride photonic waveguide circuits for our pilot line. The research has been made possible by
FWO and F.R.S.-FNRS under the Excellence of Science (EOS) program (40007560). The work
has been supported by The Dutch National Growth Fund PhotonDelta, INTERREG Vlaanderen-
Nederland project LIGHTUP, and CHIPS-JU PhotonixFAB (101111896).
REFERENCES
1L. Torrijos-Morán and D. Pérez-López, “Industry insight: photonics to scale AI data centers,”
npj Nanophotonics 3, 8 (2026).
2D. Chelladurai, M. Kohli, J. Winiger, D. Moor, A. Messner, Y. Fedoryshyn, M. Eleraky, Y. Liu,
H. Wang, and J. Leuthold, “Barium titanate and lithium niobate permittivity and pockels coef-
ficients from megahertz to sub-terahertz frequencies,” Nature Materials 24, 868–875 (2025).
11

<!-- page 12 -->
3M. Kohli, D. Chelladurai, L. Kulmer, T. Blatter, Y. Horst, K. Keller, M. Doderer, J. Winiger,
D. Moor, A. Messner, T. Buriakova, C. Convertino, F. Eltes, Y. Fedoryshyn, U. Koch, and
J. Leuthold, “The plasmonic BTO-on-SiN platform – beyond 200 GBd modulation for optical
communications,” Light: Science & Applications 14, 399 (2025).
4B. Baeuerle, W. Heni, C. Hoessbacher, Y. Fedoryshyn, U. Koch, A. Josten, T. Watanabe, C. Uhl,
H. Hettrich, D. L. Elder, L. R. Dalton, M. Möller, and J. Leuthold, “120 GBd plasmonic Mach-
Zehnder modulator with a novel differential electrode design operated at a peak-to-peak drive
voltage of 178 mV,” Optics Express 27, 16823–16832 (2019), publisher: Optica Publishing
Group.
5M. Eppenberger, A. Messner, B. I. Bitachon, W. Heni, T. Blatter, P. Habegger, M. Destraz,
E. De Leo, N. Meier, N. Del Medico, C. Hoessbacher, B. Baeuerle, and J. Leuthold, “Resonant
plasmonic micro-racetrack modulators with high bandwidth and high temperature tolerance,”
Nature Photonics 17, 360–367 (2023), publisher: Nature Publishing Group.
6C. Wu, T. Reep, S. Brems, D. Yudistira, J. Van Campenhout, I. Asselberghs, C. Huyghebaert,
M. Pantouvaki, Z. Wang, and D. Van Thourhout, “Graphene-Based Silicon Photonic Electro-
Absorption Modulators and Phase Modulators,” IEEE Journal of Selected Topics in Quantum
Electronics 30, 1–11 (2024), conference Name: IEEE Journal of Selected Topics in Quantum
Electronics.
7M. Rahimi and M. Noori, “Graphene-based waveguide modulator for independent control of
amplitude and phase,” Scientific Reports (2026), 10.1038/s41598-026-47013-8.
8D. Steckler, S. Lischke, Y. Yamamoto, W.-C. Wen, A. Peczek, J. Beyer, A. Kroh, O. Fursenko,
F. Bärwolf, S. Marschmeyer, P. Kulse, D. Wolansky, and L. Zimmermann, “Monolithic electro-
optic platform on silicon with bandwidth of 100 GHz and beyond,” Nature Communications 16,
10789 (2025).
9M. Zhang, C. Wang, P. Kharel, D. Zhu, and M. Lonˇcar, “Integrated lithium niobate electro-optic
modulators: when performance meets scalability,” Optica 8, 652–667 (2021).
10X. Zhou, D. Yi, D. W. U. Chan, and H. K. Tsang, “Silicon photonics for high-speed communi-
cations and photonic signal processing,” npj Nanophotonics 1, 27 (2024).
11A. F. Wandesleben, D. Truffier-Boutry, F. Glowacki, A. Royer, M. Lederer, B. Lilienthal-Uhlig,
and C. Vogt, “Influences and Diffusion Effects of Lithium Contamination during the Thermal
Oxidation Process of Silicon,” Advanced Engineering Materials 26, 2400396 (2024).
12

<!-- page 13 -->
12S. Ghosh, S. Yegnanarayanan, D. Kharas, M. Ricci, J. J. Plant, and P. W. Juodawlkis, “Wafer-
scale heterogeneous integration of thin film lithium niobate on silicon-nitride photonic integrated
circuits with low loss bonding interfaces,” Optics Express 31, 12005–12015 (2023).
13M. Churaev, R. N. Wang, A. Riedhauser, V. Snigirev, T. Blésin, C. Möhl, M. H. Anderson,
A. Siddharth, Y. Popoff, U. Drechsler, D. Caimi, S. Hönl, J. Riemensberger, J. Liu, P. Seidler,
and T. J. Kippenberg, “A heterogeneously integrated lithium niobate-on-silicon nitride photonic
platform,” Nature Communications 14, 3499 (2023).
14A. Rahman, F. Valdez, V. Mere, C. O. d. Beeck, P. Wuytens, and S. Mookherjea, “Integration of
Hybrid Thin-Film Lithium Niobate Electro-Optic Modulators on a Wafer-Scale Silicon Nitride
Photonics Platform,” Journal of Lightwave Technology 44, 1822–1831 (2026).
15L. Wu, Z. Zhou, W. Ma, H. Wang, Z. Ruan, C. Guo, S. Gao, Z. Huang, L. Qi, J. Liu, J. Feng,
D. Liu, K. Chen, and L. Liu, “Heterogeneous back-end-of-line integration of thin-film lithium
niobate on active silicon photonics for single-chip optical transceivers,” (2025).
16J. Declercq, S. Niu, M. Niels, A. Shahin, J. van Kerrebrouck, M. Billet, M. Berclano, J. Lam-
brech, B. Moeneclaey, T. Vanackere, E. Vissers, C. Coughlan, G. Roelkens, A. Moerman,
O. Caytan, S. Lemey, M. Chakrabarti, H. Kobbi, M. Kim, D. Yudistra, R. Loo, S. Bipul, F. Fer-
raro, Y. Ban, P. De Heyn, G. Torfs, J. Bauwelinck, D. Velenis, P. Ossieur, P. Absil, B. Kuyken,
J. van Campenhout, N. Singh, C. Bruynsteen, and X. Yin, “320 Gb/s Unamplified Transmission
Using 1 00 GHz Ge PD and TFLN MZM on a Foundry-Compatible SiPh Platform Co-Packaged
with Traveling-Wave Drivers and TIAs,” in 2025 European Conference on Optical Communica-
tions (ECOC) (2025) pp. 1–4.
17G. Roelkens, J. Zhang, L. Bogaert, E. Soltanian, M. Billet, A. Uzun, B. Pan, Y. Liu, E. Delli,
D. Wang, V. B. Oliva, L. T. Ngoc Tran, X. Guo, H. Li, S. Qin, K. Akritidis, Y. Chen, Y. Xue,
M. Niels, D. Maes, M. Kiewiet, T. Reep, T. Vanackere, T. Vandekerckhove, I. L. Lufungula,
J. De Witte, L. Reis, S. Poelman, Y. Tan, H. Deng, W. Bogaerts, G. Morthier, D. Van Thourhout,
and B. Kuyken, “Present and future of micro-transfer printing for heterogeneous photonic inte-
grated circuits,” APL Photonics 9, 010901 (2024).
18C. Yu, M. Zhang, L. Liang, L. Qin, Y. Chen, Y. Lei, Y. Wang, Y. Song, C. Qiu, P. Jia, D. Li,
and L. Wang, “Advancements in transfer printing techniques and their applications in photonic
integrated circuits,” Light: Science & Applications 14, 396 (2025).
19M. Niels, T. Vandekerckhove, L. De Jaeger, M. Billet, and B. Kuyken, “Advances in Micro-
Transfer Printing of Lithium Niobate Thin-Films for Silicon Photonic Devices,” Nanophotonics
13

<!-- page 14 -->
15, e70042 (2026), _eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/nap2.70042.
20Y. Chen, K. Akritidis, K.-W. Chen, L. De Jaeger, J. De Witte, A. F. Gamez, M. F. C. Garrido,
M. Kiewiet, H. Li, C. Lin, et al., “Micro-transfer printing on silicon photonics: Tutorial, recent
progress and outlook,” Journal of Lightwave Technology (2026).
21M. Niels, E. Vissers, T. Vanackere, A. Moerman, X. Guo, P. Geerinck, E. Soltanian, J. Zhang,
S. Janssen, P. Verheyen, et al., “Demonstration of lithium niobate integration on a 200-mm
silicon photonics wafer using transfer printing,” Optics Letters 50, 4678–4681 (2025).
22“Transverse,” https://transverse.technology/, (Accessed:2026-05-29).
23N. Boynton, H. Cai, M. Gehl, S. Arterburn, C. Dallo, A. Pomerene, A. Starbuck, D. Hood,
D. C. Trotter, T. Friedmann, C. T. DeRose, and A. Lentine, “A heterogeneously integrated
silicon photonic/lithium niobate travelling wave electro-optic modulator,” Optics Express 28,
1868–1884 (2020).
24P. Nenezic, E. Vissers, A. Moerman, L. Bogaert, S. Atzeni, X. Zheng, T. Vanackere, M. Niels,
A. Papadopoulou, P. De Heyn, et al., “A variability-aware simulation and design work-
flow for wafer-scale, heterogeneously integrated lithium niobate modulators,” arXiv preprint
arXiv:2605.28765 (2026).
25H. Li, Y. Gu, T. Pannier, S. Niu, P. Heise, C. Mai, P. Ramaswamy, A. Farrell, A. Fe-
cioru, A. Trindade, R. Loi, N. Singh, S. Qin, B. Pan, J. Zhang, J. Rimböck, K. Dhaenens,
T. Baere, G. Steenberge, D. Bode, D. Velenis, G. Lepage, N. Singh, J. V. Campenhout, X. Yin,
G. Roelkens, and P. Ossieur, “A 3D-integrated BiCMOS-silicon photonics high-speed receiver
realized using micro-transfer printing,” (2026), iSSN: 2693-5015.
26J. Cai, A. Kotz, H. Larocque, C. Wang, X. Ji, J. Zhang, D. Drayss, J. Sun, S. Zheng, X. Ou,
C. Koos, and T. J. Kippenberg, “Heterogeneously integrated lithium tantalate-on-silicon nitride
modulators for high-speed communications,” Nature Communications 17, 3314 (2026).
27M. Niels, T. Vanackere, E. Vissers, T. Zhai, P. Nenezic, J. Declercq, C. Bruynsteen, S. Niu,
A. Moerman, O. Caytan, N. Singh, S. Lemey, X. Yin, S. Janssen, P. Verheyen, N. Singh, D. Bode,
M. Davi, F. Ferraro, P. Absil, S. Balakrishnan, J. Van Campenhout, G. Roelkens, B. Kuyken, and
M. Billet, “A high-speed heterogeneous lithium tantalate silicon photonics platform,” Nature
Photonics 20, 225–231 (2026).
28J. Su, Y. Dai, A. Sun, S. Ran, Y. Yuan, L. Lu, C. Zeng, J. Zhang, Y. Li, J. Xia, N. Chi, J. Chen,
and L. Zhou, “Low-Loss and High-Speed Heterogeneous Lithium Tantalate-on-Si3N4 Modu-
lator via Micro-Transfer Printing,” in CLEO 2025 (2025), paper PD104_4 (Optica Publishing
14

<!-- page 15 -->
Group, 2025) p. PD104_4.
29C. Lin, P. Nenezic, A. Moerman, K. Akritidis, T. Vanackere, S. Atzeni, M. Niels, H. Li, V. B.
Oliva, M. Billet, and B. Kuyken, “Thin-film lithium tantalate for ultraviolet integrated electro-
optic modulator,” (2026), arXiv:2605.02758 [physics.optics].
15

