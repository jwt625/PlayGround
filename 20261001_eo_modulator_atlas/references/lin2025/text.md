---
paper_id: lin2025
source_url: https://doi.org/10.1038/s41467-026-69588-6
doi: 10.1038/s41467-026-69588-6
license: https://creativecommons.org/licenses/by-nc-nd/4.0
sha256: 6dff0e22d007449f2eb8137600e9b9431e6e1b52f92f08e47865163e06267452
pages: 26
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
https://doi.org/10.1038/s41467-026-69588-6
Received: 26 August 2025
Accepted: 29 January 2026
Cite this article as: Lin, M., Li, Z., Kotz,
A. et al. Copper damascene process-
based high-performance thin-film
lithium tantalate modulators. Nat
Commun (2026). https://doi.org/
10.1038/s41467-026-69588-6
Mengxin Lin, Zihan Li, Alexander Kotz, Hugo Larocque, Nikolai Kuznetsov, Jiale Sun,
Yichi Zhang, Shuhang Zheng, Johann Riemensberger, Christian Koos & Tobias J.
Kippenberg
We are providing an unedited version of this manuscript to give early access to its
findings. Before final publication, the manuscript will undergo further editing. Please
note there may be errors present which affect the content, and all legal disclaimers
apply.
If this paper is publishing under a Transparent Peer Review model then Peer
Review reports will publish with the final article.
© The Author(s) 2026. Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit
to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not
have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this
article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the
article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain
permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.
Nature Communications
Article in Press
Copper damascene process-based high-
performance thin-film lithium tantalate
modulators

<!-- page 2 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
Copper Damascene Process-Based High-Performance
Thin-Film Lithium Tantalate Modulators
Mengxin Lin,1,2 Zihan Li,1,2 Alexander Kotz,3 Hugo Larocque,1,2 Nikolai Kuznetsov,1,2 Jiale Sun,1,2
Yichi Zhang,1,2 Shuhang Zheng,1,2 Johann Riemensberger,1,2 Christian Koos,3 Tobias J. Kippenberg,1,2,†
1Institute of Physics, Swiss Federal Institute of Technology Lausanne (EPFL), CH-1015 Lausanne,
Switzerland
2Institute of Electrical and Micro Engineering, EPFL, CH-1015 Lausanne, Switzerland
3Institute of Photonics and Quantum Electronics (IPQ), Karlsruhe Institute of Technology (KIT),
76131 Karlsruhe, Germany
Email: † tobias.kippenberg@epfl.ch
The conversion between electrical and optical signals underpins modern optical communi-
cation systems and increasingly requires tight co-integration with electronics at short length
scales. Thin-film lithium tantalate has emerged as a promising electro-optic platform due
to its large Pockels coefficient, low bias drift, and high power handling, yet its integration
with standardized microelectronic processes remains limited. Here we show that incorpo-
rating the copper Damascene process into thin-film lithium tantalate modulators enables
a scalable, electronics-compatible fabrication approach. The resulting devices exhibit ap-
proximately 10% lower microwave loss than conventional gold-electrode designs, while si-
multaneously supporting watt-level on-chip optical power handling and maintaining a stable

<!-- page 3 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
quasi-static half-wave voltage from 1 Hz to 1 MHz, with a bias point drift of only 0.4 dB over
a 15-hour period when operated at 1.75 mW on-chip optical power. High-speed transmission
experiments demonstrate line rates of 416 Gbit/s (PAM4) and 540 Gbit/s (PAM8) below the
25% soft-decision forward-error-correction threshold, together with watt-level on-chip opti-
cal power handling. These results establish a practical route toward scalable chip-on-wafer
integration of electro-optic modulators with microelectronic circuits.
Integrated electro-optic (EO) modulators provide a scalable platform for converting information
between electrical and optical domains, serving as key building blocks for high-capacity opti-
cal communications 1–3 and emerging chip-to-chip interconnects 4,5. Recent progress in thin-
film lithium niobate and thin-film lithium tantalate photonics has transformed traditional bulk
devices 6,7. The adoption of a layered “on-insulator” structure, combined with improved etch-
ing techniques, enables strong optical confinement and allows electrodes to be placed closer to the
waveguide, thereby enhancing modulation efficiency. This structure also facilitates velocity match-
ing between optical and microwave signals, resulting in higher modulation bandwidths. However,
as the electrode gap is reduced to achieve tighter optical confinement and stronger modulation, the
currents in the signal line and ground plane become increasingly concentrated near the gap due
to the electromagnetic proximity effect 8. This enhanced current concentration increases the per-
unit-length resistance of the transmission line, leading to higher microwave loss. The resulting loss
limits further improvements in voltage–bandwidth performance. Employing electrodes with peri-
odic microstructured “T-rails” mitigates the proximity effect by introducing a larger effective gap
while preserving the electric-field distribution required for efficient modulation 9–12. An alternative

<!-- page 4 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
approach to reducing microwave loss replaces conventional gold electrodes with low-resistivity
metals, such as copper which has a resistivity approximately 24% lower than gold 13.
While continuous efforts have been made to extend the voltage–bandwidth performance of thin-
film lithium niobate and lithium tantalate modulators 14,15, most reported devices remain stand-
alone prototypes, requiring bulky electrical probes and external control electronics. Meanwhile,
AI-based applications are driving an ever-growing demand for higher data throughput, lower la-
tency, and improved energy efficiency in modern computing systems 16. To meet these require-
ments, co-packaged optics (CPO) has emerged as a transformative technology that tightly inte-
grates photonics and electronics 17. Such integration reduces high-frequency signal degradation,
reduces energy consumption, and enables large-scale system scalability. Among various integra-
tion approaches, three-dimensional (3D) integration, as illustrated in Figure 1 (a), offers the short-
est metal interconnects by vertically stacking photonic and electronic chips, thereby minimizing
signal loss and improving overall efficiency 18. While this stacking is typically achieved through
flip-chip bonding using microbumps 19, the industry is transitioning toward hybrid copper-copper
bonding 20. In this approach, copper surfaces are directly joined through thermally activated dif-
fusion, eliminating solder interfaces and thereby reducing parasitic effects. A key enabler of this
transition is planarized copper metallization achieved via the copper Damascene process, already
widely adopted in both microelectronics 21 and silicon photonics 22. However, ferroelectric pho-
tonic platforms still rely on conventional gold electrodes, highlighting the need for further innova-
tion to achieve fully integrated, high-performance photonic–electronic systems.

<!-- page 5 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
Here, we address this challenge by incorporating the copper Damascene process into the fabri-
cation of lithium tantalate photonic integrated circuits (PICs). This approach leverages a mature,
low-cost, and widely adopted method used for metallization in microelectronics and silicon pho-
tonics. Using this process, we demonstrate integrated electro-optic modulators with microwave
losses approximately 10% lower than those employing conventional gold electrodes, while pre-
serving the intrinsic benefits of thin-film lithium tantalate, including low optical loss, high mod-
ulation efficiency, and broad modulation bandwidth. The resulting planar electrode surface also
supports direct chip-on-chip and chip-on-wafer bonding with driver electronics.
Fabrication. The devices were fabricated from a commercially available X-cut thin-film lithium
tantalate (TFLT) wafer (NANOLN), which consists of 600 nm of lithium tantalate (LiTaO3, LT),
4.7 µm of SiO2, and a 525 µm-thick high-resistivity silicon substrate. The TFLT PICs were manu-
factured using an etching technique based on a diamond-like carbon hard mask, which can reliably
produce both lithium niobate and lithium tantalate PICs 23,24. The etch depth of the LT was 320 nm,
leaving a 280 nm-thick slab for efficient electro-optic modulation and proper phase matching be-
tween microwave and optical signals.
Next, the copper Damascene process, outlined in Figure 1 (b), was introduced to fabricate traveling-
wave electrodes. This Damascene process differs from silicon nitride Damascene methods 25,
which reduce optical propagation losses in waveguides whereas we apply it to copper electrodes
to minimize microwave loss. In this process, dielectric trenches are patterned and filled by elec-
troplated copper followed by chemical–mechanical polishing (CMP). This process produces well-

<!-- page 6 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
defined electrodes with a smooth surface potentially suitable for chip-on-wafer bonding. Further
information on the fabrication is available in the Methods. Figure 1 (c) shows two fabricated Cu-
TFLT MZMs extending over an effective modulation length of 6 mm. Figure 1 (d) illustrates the
fabricated 4-inch wafer with hundreds of Cu-TFLT MZMs, highlighting its potential for scalable
integration of next-generation photonic systems. To improve the wafer’s uniformity following the
CMP process, inactive regions feature a filler copper pattern, which is visible in Figure 1 (c-e). As
respectively shown in Figure 1 (e-g), each modulator consists of 50:50 1×2 multimode interference
(MMI) couplers, EO phase shifters operating in a push-pull configuration, and double-layer tapers
for enhanced PIC-to-fiber coupling efficiency 26 (Figure S1). As illustrated in Figure 1 (f), the tight
confinement of the modulator’s microwave and optical modes ensures strong overlap between the
two fields and thus enhanced modulation efficiency.
Electrical Transmission. To assess the potential of copper as an electrode material for high-
performance modulators, we characterized the electrical properties of fabricated copper Dama-
scene CPWs, including resistivity and microwave performance, and compared them with their
conventional gold-based counterparts. Figure 2 (a) shows a photograph of the test structures. For
context, Figure 2 (b) presents the bulk resistivity of the seven most conductive elemental metals
commonly used in microelectronics and photonics. Although thin films are typically employed in
these applications, their resistivity depends on thickness and fabrication process; thus, bulk values
are used here for a general comparison. Silver (1.59 µΩcm) exhibits the lowest resistivity, fol-
lowed by copper (1.68 µΩcm), which is only 5.6% higher, and gold (2.21 µΩcm), which is 39%
higher 27. We focus on copper and gold due to the impracticality of using silver (see Discussion).

<!-- page 7 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
Figure 2 (c) compares the measured resistivity of thin-film gold and copper and its evolution with
time. The gold thin film (796 nm) was deposited by electron beam evaporation, and the copper thin
film (1170 nm) by electroplating. Both were formed on silicon carrier wafers with a 2 µm silicon
dioxide layer and characterized using a four-point probing and mechanical profilometry. The elec-
troplated copper exhibited a 24 % decrease in resistivity within 48 hours at room temperature due
to self-annealing 28, stabilizing at 2.04 µΩcm - approximately 20% lower than that of gold. Studies
suggest that the resistivity of electroplated copper could be further reduced to 1.77 µΩcm 29 via
process optimization. Using these measured values, we modeled the microwave behavior of the
fabricated CPWs through full-wave electromagnetic simulations (Figure S2). Figure 2 (d) shows
the simulated microwave loss of a 16 mm-long copper Damascene CPW with a 6 µm gap and a
27 µm signal line, alongside simulated CPWs with varying resistivities but identical geometries.
The measured microwave loss agrees well with the simulation, validating the model. The data
indicate that copper electrodes yield approximately 10% lower microwave loss than gold, while
the propagation index and characteristic impedance change by less than 0.3%. The dependence of
the microwave effective index on design parameters is provided in Figure S3.
Electro-optic Modulation. A direct approach to assess the modulators’ performance is to mea-
sure their EO response, including the quasi-static half-wave voltage (Vπ) and the 3 dB EO band-
width. Finite-element simulations (COMSOL Multiphysics) were performed to analyze the VπL
product and optical loss for various modulator geometries. A design with 6 µm electrode gap and
a 4 µm-wide waveguide was selected. This relatively wide waveguide maintains a similar VπL
and optical loss while allowing a larger electrode gap that reduces microwave loss (Figure S4) 30.

<!-- page 8 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
A 100 µm-long adiabatic linear taper connects the modulation region to the single-mode rout-
ing waveguide, ensuring single-mode operation. The quasi-static Vπ was measured by applying
a low-frequency triangular voltage signal to the CPW electrode and recording the optical output
simultaneously. Figure 3 (a) shows the normalized transmission of a modulator as a function of
the applied voltage, linearly swept at 100 Hz, exhibiting a near-sinusoidal response with a Vπ of
1.7 V. For a device length of 16 mm, this corresponds to a VπL of 2.7 V·cm, closely matching the
simulated value of 2.64 V·cm. The inset dB-scale plot shows an extinction ratio of 20 dB, suitable
for advanced modulation formats such as PAM4 and PAM8. The dependence of the Vπ on the
modulation frequency was also examined. As shown in Figure 3 (b), Vπ remains stable across a
wide frequency range from 1 Hz to 1 MHz. More detailed results are shown in Figure S5. Similar
measurements were performed over a range of on-chip optical powers up to 1.17 W, and the results
are shown in Figure S6. This flat frequency response contrasts with the pronounced low-frequency
instability previously reported for lithium niobate modulators with gold CPWs 31. A small fluc-
tuation (< 0.1 V) in Vπ between 1 kHz and 100 kHz is attributed to dielectric relaxation intrinsic
to the Cu–LiTaO3 stack. Such effects arise from slow polarization dynamics or charge trapping
in the ferroelectric layer when the modulation frequency approaches the characteristic relaxation
rate. The magnitude and frequency of this fluctuation remain unchanged as the on-chip optical
power increases from 1 mW to 1.17 W, confirming that is not due to optical intensity-dependent
photorefractive effects. As indicated by the stable quasi-static Vπ, the modulator exhibits excel-
lent long-term stability, showing only a 0.4 dB bias drift over 15 hours at 1.75 mW on-chip power
(Figure 3 (c)). This stability removes the need for active thermal biasing and its associated power-

<!-- page 9 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
hungry heaters 30. More detailed bias-drift measurement results are provided in Figure S7. Fig-
ure 3 (d) presents the measured EO response up to 110 GHz. The 3 dB EO bandwidth reaches
40 GHz for a 16 mm-long device and 100 GHz for a 6 mm-long device, both with a 6 µm electrode
gap. The simulated EO response is derived from the measured electrical response and agrees well
with the experimental data, validating the measurement. Simulations further indicate that the 3-dB
EO bandwidth is approximately 10% higher than that of comparable gold-electrode devices (Fig-
ure S8). These results demonstrate that the copper-electrode-based modulators combine high EO
bandwidth with low drive voltage, enabling high-speed optical communication applications.
Optical Communications. To demonstrate the exceptional performance of our Cu-TFLT MZM,
we performed a high-speed intensity-modulation and direct-detection (IMDD) signaling experi-
ment. The experimental apparatus for this task is illustrated in Figure 4 (a) and further described in
the Methods. We generated and received PAM4 and PAM8 data signals with symbol rates between
144 GBd and 208 GBd. Figure 4 (b) displays the measured bit error ratios (BER) of the various
PAM signals as a function of symbol rate. The horizontal dashed lines indicate the thresholds
for typical soft-decision (SD) forward-error correction (FEC) with 25 % and 15 % overhead, and
for hard-decision (HD) FEC with 7 % overhead. The results show that we can transmit 180 GBd
PAM8 signals and 208 GBd PAM4 signals while the respective BER of 3.76×10−2 and 3.56×10−2
are still below the 25 % SD-FEC limit. In Figure 4 (c), we further calculate the generalized mu-
tual information (GMI) of our measurements based on log-likelihood ratios by using an additive
white Gaussian noise channel model 32. The dashed curves depict the achievable information rate
(AIR), which is the product of the GMI and the symbol rate. The solid curves in Figure 4 (c) show

<!-- page 10 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
the net data rates for measurements with BER below the 25 % SD-FEC limit, where the net data
rates are determined by multiplying the transmitted line rates with the code rate that is associated
with the normalized GMI threshold, extracted from 33. The results indicate that the highest AIR of
449 Gbit/s is achieved by using PAM8 signals at a symbol rate of 176 GBd. This corresponds to a
net data rate of 423 Gbit/s, which is on par with results demonstrated for high-bandwidth thin-film
lithium niobate MZMs 34. Comparison of these data rates and other metrics achieved in this work
along with those from other state-of-the-art lithium niobate and lithium tantalate modulators are
available in Table S1. It should be noted that some of the results such as the PAM4 symbol rate
were limited by the bandwidth of our driver electronics, not by the MZM itself.
Discussion. From a materials perspective, metals with lower resistivity reduce microwave loss.
Gold electrodes are conventionally used in bulk lithium niobate modulators due to their relatively
low resistivity, chemical stability, and ease of fabrication by evaporation and lift-off or electroplat-
ing followed by etching 35,36. With the emergence of thin-film platforms and growing demands
for high-speed data connectivity, replacing conventional gold electrodes with copper Damascene
electrodes has become strategically important. Copper offers lower resistivity, improved voltage-
bandwidth performance, and compatibility with scalable electronic integration 21,22. Although
silver has slightly lower resistivity than copper, it suffers from diffusion and sulfidation issues that
cannot be addressed as effectively as in the case of copper 37,38. Moreover, silver’s electroplating
and CMP processes are less mature. Consequently, despite its marginal electrical advantage, silver
is less reliable and less cost-effective for large-scale integration.

<!-- page 11 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
To assess the stability of our copper Damascene electrodes, we measured their microwave per-
formance over an extended period and conducted thermal cycling tests. The results show that
the electrodes maintained their loss and propagation effective index over six months of ambient
storage (Figure S9). These properties also remained unchanged under a 45-minute-per-cycle ther-
mal cycling test between room temperature and 80 °C (Figure S10). We further characterized the
microwave metrics and quasi-static Vπ values on devices from nine fields across a 4-inch wafer
(Figure S11). The results show that the device-to-device variation of the electrical S21 parameter
is below 0.1 dB at 60 GHz (less than than 3% in linear scale), the microwave group indices are all
well matched to the optical group index of 2.22 within a tolerance of ± 0.03, and the quasi-static
Vπ varies by less than 5%. These results confirm high process reproducibility and support the
scalability of our approach.
Further improvements in the voltage–bandwidth performance could be achieved by adopting a
microstructured electrode design 9–12. Test structures fabricated with the copper Damascene pro-
cess demonstrate the potential to reduce microwave loss to 2–4 dB/cm, while preserving velocity
and impedance matching through proper design parameters (Figure S12). In addition, the cop-
per Damascene process can reliably produce submicrometer-scale features, which is difficult to
achieve using the conventional lift-off process. To accommodate the higher microwave effective
index of these designs, the thickness of the buried oxide layer may need further optimization 39.
We also envision a full co-packaging of Pockels modulators with their driving circuits using emerg-
ing copper-copper hybrid bonding techniques 20. Although a full bonding demonstration is beyond

<!-- page 12 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
the scope of this study, we have shown morphological results that strongly support the feasibility
of chip-on-wafer integration using our process. Specifically, the CMP-treated copper electrodes
exhibit an RMS roughness below 11 nm and a maximum step height of approximately 37 nm at
the Cu-SiO2 interface (Figure S13), values that fall within accepted limits for hybrid bonding 40.
In summary, our results directly demonstrate the advantages of implementing a copper Dama-
scene process for electro-optic modulators targeting high-bandwidth telecommunications, while
also showing strong potential for co-integration with high-speed microelectronics.
Methods
Copper Damascene Fabrication Flow. To begin with, the TFLT PICs were cladded with a 2.5 µm-
thick silicon dioxide layer by plasma-enhanced chemical vapor deposition (PECVD). This cladding
layer will embed the subsequently formed copper electrodes. The layout for these electrodes was
defined with a DUV stepper lithography (ASML PAS 5500/350C) and transferred as preforms into
the cladding layer through fluorine-based dry etching in a reactive ion etcher. These preforms
were coated with 10 nm of titanium (Ti) and 100 nm of copper (Cu) through sputtering (Alliance-
Concept DP 650). The titanium layer serves as a barrier to prevent copper from diffusing into
the surrounding dielectric 41. It also provides adhesion to a subsequently sputtered copper layer
acting as a seed to facilitate copper electroplating. The electroplating was performed in a Silicet
Electroplating unit and contributed to a 2.8 µm thick copper layer with low resistivity (2.04 µΩcm).
Chemical mechanical planarization (CMP) was employed to remove the excess copper, leaving

<!-- page 13 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
copper only in the etched preforms with a final thickness of 1.8 µm. This relatively thick metal
layer minimizes microwave losses in the CPW electrodes across frequencies ranging from tens of
GHz to 110 GHz (Figure S14). Since the electrodes are embedded within a SiO2 cladding layer and
share a common top surface, their thickness can be indirectly monitored by measuring the thickness
of nearby SiO2 windows using spectroscopic reflectometry, as shown in Figure 1 (c). Subsequently,
a 100 nm-thick silicon nitride passivation layer was deposited with PECVD to prevent copper ox-
idation. The electrode pads were then opened and capped with gold (Au) for efficient and stable
electrical probing. Finally, chip singulation was achieved through a combination of dry etching -
standard reactive ion etching for LT and silicon dioxide, deep reactive ion etching for silicon - and
backside grinding. This process ensures smooth facets for edge coupling to single-mode fibers.
Data Transmission Apparatus. An external-cavity laser (ECL, 17.8 dBm at 1550 nm) provides
the optical carrier. A high-speed arbitrary-waveform generator (AWG, M8199B, Keysight) is used
to generate the electrical drive signal, which is fed to the CPW of the MZM via a 20 cm-long
RF cable, a broadband RF amplifier, and a 110 GHz RF probe. We synthesize various pulse-
amplitude modulation (PAM) signals based on pseudo-random bit sequences and apply root-raised
cosine pulse-shaping filters with a roll-off of β = 0.05. We account for the frequency-dependent
RF loss up to the input of the feeding probe by applying a linear minimum-mean-square-error
(MMSE) predistortion. The CPW is terminated by a 50 Ωresistor via a second 110 GHz RF probe.
As previously demonstrated in Figure 3 (c), the EO stability of the Cu-TFLT modulator allows
reliable DC biasing at the quadrature point for intensity modulation with a bias-T attached to the
second probe. The optical power at the output fiber of the MZM operated at the quadrature point

<!-- page 14 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
is 5.6 dBm, which exceeds the power requirements in typical specifications for high-speed optical
ethernet transceivers 42. Still, an additional erbium-doped fiber amplifier (EDFA) was needed in
the experiment to reach sufficient power levels for the high-speed photodiode (8.5 dBm) at the
receiver. Note that a practical transceiver implementation could rely on a sufficiently broadband
amplifier after the photodiode, thereby rendering the EDFA unnecessary. In our experiments, the
out-of-band amplified spontaneous-emission (ASE) noise of the EDFA is suppressed by a tunable
bandpass filter (BPF), and a variable optical attenuator (VOA) is used to adjust a power level of
8.5 dBm at the input of the photodiode. The electrical signal at the photodiode output is digitized by
a real-time oscilloscope (RTO, UXR 1004A, Keysight) with an analogue bandwidth of 105 GHz
and a sampling rate of 256 GSa/s. The data is finally extracted by offline receiver DSP (Rx-
DSP), which contains resampling to 2 Sa/symbol, timing recovery, linear Sato equalization, and an
additional decision-directed least-mean-square (DD-LMS) equalizer.
Data Availability
The experimental datasets and scripts used to produce the plots in this paper are available at Zenodo
(https://doi.org/10.5281/zenodo.18148021)

<!-- page 15 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
References
1. Koenig, S. et al. Wireless sub-thz communication system with high data rate. Nature photonics
7, 977–981 (2013).
2. Marin-Palomo, P. et al. Microresonator-based solitons for massively parallel coherent optical
communications. Nature 546, 274–279 (2017).
3. Rizzo, A. et al. Massively scalable kerr comb-driven silicon photonic link. Nat. Photonics 17,
781–790 (2023).
4. Sun, C. et al. Single-chip microprocessor that communicates directly using light. Nature 528,
534–538 (2015).
5. Atabaki, A. H. et al. Integrating photonics with silicon nanoelectronics for the next generation
of systems on a chip. Nature 556, 349–354 (2018).
6. Wang, C. et al.
Integrated lithium niobate electro-optic modulators operating at CMOS-
compatible voltages. Nature 562, 101–104 (2018).
7. He, M. et al. High-performance hybrid silicon and lithium niobate Mach–Zehnder modulators
for 100 Gbit s-1 and beyond. Nat. Photonics 13, 359–364 (2019).
8. Mei, S. & Ismail, Y. I. Modeling skin and proximity effects with reduced realizable rl circuits.
IEEE Transactions on very large scale integration (VLSI) systems 12, 437–447 (2004).
9. Bennion, I. & Walker, T. Guided-wave devices and circuits. Physics World 3, 47 (1990).

<!-- page 16 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
10. Spickermann, R. & Dagli, N. Experimental analysis of millimeter wave coplanar waveguide
slow wave structures on gaas. IEEE Trans. Microw. Theory Techn. 42, 1918–1924 (1994).
11. Kharel, P., Reimer, C., Luke, K., He, L. & Zhang, M. Breaking voltage–bandwidth limits in
integrated lithium niobate modulators using micro-structured electrodes. Optica 8, 357–363
(2021).
12. Chen, G. et al. High performance thin-film lithium niobate modulator on a silicon substrate
using periodic capacitively loaded traveling-wave electrode. APL photonics 7 (2022).
13. Zhang, Y. et al. Systematic investigation of millimeter-wave optic modulation performance in
thin-film lithium niobate. Photon. Res. 10, 2380–2387 (2022).
14. Xu, M. et al. Dual-polarization thin-film lithium niobate in-phase quadrature modulators for
terabit-per-second transmission. Optica 9, 61–62 (2022).
15. Xu, M. et al. Attojoule/bit folded thin film lithium niobate coherent modulators using air-
bridge structures. Apl Photonics 8 (2023).
16. Goodfellow, I. et al. Generative adversarial networks. Communications of the ACM 63, 139–
144 (2020).
17. Huang, J. Gtc 2025 keynote address. NVIDIA (2025). Accessed: November 7, 2025.
18. Xiang, C. & Bowers, J. E. Building 3d integrated circuits with electronics and photonics.
Nature Electronics 7, 422–424 (2024).

<!-- page 17 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
19. Daudlin, S. et al.
Three-dimensional photonic integration for ultra-low-energy, high-
bandwidth interchip data links. Nature Photonics 1–8 (2025).
20. Moore, S. K. The copper connection: Hybrid bonding is the 3d-chip tech that’s saving moore’s
law. IEEE Spectr. 61, 34–39 (2024).
21. Andricacos, P. C., Uzoh, C., Dukovic, J. O., Horkans, J. & Deligianni, H. Damascene copper
electroplating for chip interconnections. IBM Journal of Research and Development 42, 567–
574 (1998).
22. Fahrenkopf, N. M. et al. The aim photonics mpw: A highly accessible cutting edge technology
for rapid prototyping of photonic integrated circuits. IEEE J. Sel. Top. Quantum Electron. 25,
1–6 (2019).
23. Li, Z. et al. High density lithium niobate photonic integrated circuits. Nat. Commun. 14, 4856
(2023).
24. Wang, C. et al. Lithium tantalate photonic integrated circuits for volume manufacturing. Na-
ture 629, 784–790 (2024).
25. Pfeiffer, M. H. P. et al. Ultra-smooth silicon nitride waveguides based on the damascene reflow
process: fabrication and loss origins. Optica 5, 884–892 (2018).
26. He, L. et al. Low-loss fiber-to-chip interface for lithium niobate photonic integrated circuits.
Opt. Lett. 44, 2314–2317 (2019).
27. Gall, D. Electron mean free path in elemental metals. Journal of applied physics 119 (2016).

<!-- page 18 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
28. Harper, J. M. E. et al. Mechanisms for microstructure evolution in electroplated copper thin
films near room temperature. J. Appl. Phys. 86, 2516–2525 (1999).
29. Kang, M. S., Kim, S.-K. & Kim, J. J. A novel process to control the surface roughness and
resistivity of electroplated cu using thiourea. Japanese journal of applied physics 44, 8107
(2005).
30. Xu, M. et al. High-performance coherent optical modulators based on thin-film lithium niobate
platform. Nat. Commun. 11, 3911 (2020).
31. Holzgrafe, J. et al. Relaxation of the electro-optic response in thin-film lithium niobate mod-
ulators. Opt. Express 32, 3619–3631 (2024).
32. Ivanov, M. et al. On the information loss of the max-log approximation in bicm systems. IEEE
Trans. Inf. Theory 62, 3011–3025 (2016).
33. Hu, Q. et al. Ultrahigh-net-bitrate 363 gbit/s pam-8 and 279 gbit/s polybinary optical trans-
mission using plasmonic mach-zehnder modulator. J. Light. Technol. 40, 3338–3346 (2022).
34. Berikaa, E. et al. Tfln mzms and next-gen dacs: Enabling beyond 400 gbps imdd o-band and
c-band transmission. IEEE Photonics Technol. Lett. 35, 850–853 (2023).
35. Wooten, E. L. et al. A review of lithium niobate modulators for fiber-optic communications
systems. IEEE Journal of selected topics in Quantum Electronics 6, 69–82 (2000).
36. Noguchi, K., Mitomi, O. & Miyazawa, H. Millimeter-wave ti: Linbo3 optical modulators.
Journal of Lightwave Technology 16, 615 (1998).

<!-- page 19 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
37. Wang, Y. & Alford, T. Formation of aluminum oxynitride diffusion barriers for ag metalliza-
tion. Applied physics letters 74, 52–54 (1999).
38. Gao, L. et al. Thermal stability of titanium nitride diffusion barrier films for advanced silver
interconnects. Microelectronic engineering 76, 76–81 (2004).
39. Tang, Y. et al. High performance thin-film lithium niobate modulator on silicon substrate with
a thick silica buffer layer. Optics Express 33, 20334–20344 (2025).
40. Ohyama, M. et al. Evaluation of hybrid bonding technology of single-micron pitch with planar
structure for 3d interconnection. Microelectronics Reliability 59, 134–139 (2016).
41. Edelstein, D. et al. A high performance liner for copper damascene interconnects. In Proceed-
ings of the IEEE 2001 International Interconnect Technology Conference (Cat. No.01EX461),
9–11 (2001).
42. IEEE standard for ethernet - amendment 10: Media access control parameters, physical layers,
and management parameters for 200 gb/s and 400 gb/s operation. IEEE Std 802.3bs-201
(2017).
Acknowledgments
We thank Mohammad Bereyhi for helpful discussions. The samples were fabricated in the EPFL
Center of MicroNano Technology (CMi) and the Institute of Physics (IPHYS) cleanroom. This
project has received funding from the Horizon Europe EIC transition programme under grant No.

<!-- page 20 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
101113260 (HDLN), and this work was further supported by the Swiss State Secretariat for Edu-
cation, Research and Innovation (SERI). This material is also based upon work supported by the
Air Force Office of Scientific Research under award no. FA8655-24-1-7007.
Author contributions
J.R. conceived the concept. M.L. and Z.L. developed the copper Damascene fabrication process.
M.L. designed and fabricated the devices with help from Z.L. and Y.Z. M.L., A.K., H.L., and N.K.
performed the measurements and analyzed the data. J.S. and S.Z. carried out device packaging.
M.L., A.K., H.L., and T.J.K. wrote the manuscript with input from all authors. C.K. and T.J.K.
supervised the project.
Competing interests
TJK is co-founder of LUXTELLIGENCE SA, offering electro-optical photonic integrated circuits.
The other authors declare no competing interests.

<!-- page 21 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
Figure 1: Copper Damascene process-based thin-film lithium tantalate Mach-Zehnder modulators. a, Cross-
section diagram illustrating the level of wiring enabled by the copper Damascene process in both electronic and
photonic integrated circuits. The boxed region highlights the modulator architecture introduced in this work. b,
Simplified process flow for fabricating copper electrodes on patterned lithium tantalate wafers using a Damascene
process. c, Microscope image of two Mach-Zehnder modulators, each comprising of two 1x2 multimode interference
couplers, an unbalanced arm, and a pair of push-pull phase shifters. The yellow color indicates the gold-capped pads
for stable and efficient electrode probing. d, Photograph of a manufactured 100-mm wafer hosting hundreds of copper
Damascene lithium tantalate modulators. e, Microscope image of a multimode interference coupler surrounded by
honeycomb copper fillers, which are used to ensure uniform planarization. f, Colored scanning electron microscopy
image of the cross-section of the lithium tantalate waveguide (green) and copper electrodes (orange). g, Colored
scanning electron microscope image of a double-layer taper for efficient and broadband edge coupling between the
chip and single-mode fibers. h, Numerically simulated microwave and optical field distributions in the cross-section
of the lithium tantalate modulator.
Figure 2: Electrical characterization of copper thin-film lithium tantalate modulators. a, A photograph of copper
Damascene coplanar waveguide electrode test structures fabricated on silicon. b, Electrical resistivity of the seven
most conductive elemental metals, sorted by decreasing bulk room-temperature resistivity. c, Measured resistivity of
thin-film gold and copper, and their evolution with time. The resistivity of the electroplated copper undergoes a 24 %
decrease within 48 hours at room temperature due to its self-annealing effect, and is stabilized at 2.04 µΩcm, which
is 20 % lower than the measured resistivity of the thin-film gold. d, Measured microwave losses for the fabricated
copper coplanar waveguide electrodes along with numerically simulated losses for electrodes composed of materials
defined by different resistivities yet sharing the geometry of the fabricated device. Inset: microwave propagation loss
testing apparatus and main dimensions used in the material stack of the measured device as previously defined in
Figure 1(a). Side panels: Corresponding percent loss, microwave index, and characteristic impedance (Z0) variations
of the simulated coplanar waveguide electrodes compared to thin-film gold.
Figure 3: Electro-optic characterization of copper Damascene thin-film lithium tantalate Mach-Zehnder mod-
ulators. a, Normalized optical transmission of a modulator as a function of the applied voltage, yielding a Vπ value
of 1.7 V. The device features a 16 mm modulation region with a 4 µm-wide waveguide width, and a 6 µm electrode
gap. Inset: transmission plotted on a dB scale, showing a 20 dB extinction ratio. b, Measured Vπ as a function of
the sweeping frequency of the applied voltage signal. c, Long-term bias-point stability over 15 hours for a device
with a 6 mm modulation length, measured with an on-chip power of 1.75 mW. d, Small-signal electro-optic response
of devices with a modulation length of 16 mm and 6 mm, respectively. Inset: testing apparatus for the electro-optic
response measurement.

<!-- page 22 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
Figure 4: Data transmission experiment using intensity-modulation direct detection with a copper Damascene
thin-film lithium tantalate modulator. a, Experimental apparatus: an external cavity laser (ECL) is used as the light
source, and a fiber polarization controller (FPC) adjusts the polarization state of the light. Optical input and output
coupling to the lithium tantalate modulator is achieved using a pair of lensed fibers. The drive signals are synthe-
sized by transmitter digital signal processing (Tx-DSP) and generated by an arbitrary waveform generator (AWG).
The modulated optical signal is amplified using an erbium-doped fiber amplifier (EDFA), and out-of-band amplified
spontaneous emission (ASE) noise is suppressed by a tunable bandpass filter (BPF). The amplified signal then passes
through a variable optical attenuator (VOA) before being detected by a photodiode (PD). A high-speed real-time os-
cilloscope (RTO) samples the resulting signal, which is processed offline by receiver DSP (Rx-DSP). b, Measured
bit error ratios (BER) as a function of symbol rate for PAM8 (green) and PAM4 (blue) signals. The horizontal black
dashed lines denote the thresholds for 25 % and 15 % soft-decision forward error correction (SD-FEC), as well as 7 %
hard-decision forward error correction (HD-FEC). c, Extracted available information rates (AIR, dashed lines) and
corresponding net data rates (NDR, solid lines) for measurements with BER values below the 25 % SD-FEC thresh-
old. The maximum NDR of 423 Gbit/s is obtained by using a PAM8 signal at a symbol rate of 176 GBd. d, e, Eye
diagrams and corresponding histograms, taken at the center of the symbol slot (indicated by the vertical dashed line),
for the circled data points marked in b.

<!-- page 23 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
1. SiO2 Cladding & 
     Preform Etching
2. Ti Diﬀusion Barrier &
    Cu Seed Layer 
    
3. Cu Electroplating
a
Si3N4
Cu
Ti
LT
SiO2
Si
4. Cu Planarization &
     Si3N4 Passivation & ...
800 μm
Air
Cu
SiO2
SiO2
LT
Cu
1 μm
b
c
d
e
x
z
4 μm
70 μm
f
g
Electric Field (V/m)
2 μm
h
Electronic
Integrated Circuit
Photonic
Integrated Circuit
copper-copper hybrid bonding

<!-- page 24 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
10
20
30
40
50
60
70
80
90
100
110
Frequency (GHz)
-8
-6
-4
-2
0
Electro-Optic S21 (dB)
16 mm (Meas.)
6 mm (Meas.)
16 mm (Sim.)
6 mm (Sim.)
V L = 2.7 V·cm
a
b
c
d
10
-2
10
0
10
2
10
4
10
6
Frequency (Hz)
1
1.2
1.4
1.6
1.8
2
Vπ (V)
-1
0
1
Voltage (V)
0
0.2
0.4
0.6
0.8
1
Norm. Transmission
-1
0
1
-20
-10
0
NT (dB)
V   = 1.7 V
WG 4 μm, Gap 6 μm
Modulation length 16mm
0.4 dB
Time (h)
Variation (dB)
Change in bias point
VNA
ECL
FPC
PD
50Ω
0
3
6
9
12
15
-1
-0.5
0
0.5
1

<!-- page 25 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
a
b
c
Loss Var. (%)
-30
-20
-10
0
Index Var. (%)
-1
-0.5
0
Frequency (GHz)
Z0 Var. (%)
0
20
40
60
-1
-0.5
0
Pt
Ni
W
Al Au Cu Ag
1
2
3
5
10
Resistivity (µΩ·cm)
Bulk
Day 1
Day 2
Day 3
Day 4
1.5
2.0
2.5
20% lower
Resistivity (µΩ·cm)
Thin Film
Bulk Au: 2.21
Bulk Cu: 1.68
Cu
Au
0
10
20
30
40
50
60
Frequency (GHz)
0
1
2
3
4
5
6
Microwave Loss (dB/cm)
1.68 Cu Bulk
2.04 Cu Thin Film (Sim.)
2.21 Au Bulk
2.56 Au Thin Film
2.04 Cu Thin Film (Meas.)
Resistivity (µΩ·cm)
1.8 µm
280 nm
4.7 µm
6 µm
6 µm
27 µm
d
VNA

<!-- page 26 -->
ARTICLE IN PRESS
ARTICLE IN PRESS
Counts (a.u.)
Time (2 ps/div)
Amplitude (a.u.)
180 GBd PAM8 (540 Gbit/s)
Amplitude (a.u.)
208 GBd PAM4 (416 Gbit/s)
Counts (a.u.)
Time (2 ps/div)
208
192
176
160
144
Symbol Rate (GBd)
AIR / NDR in GBit/s
300
350
400
450
PAM4
PAM8
Net: 423 Gbit/s
208
192
176
160
144
Symbol Rate (GBd)
10-2
10-3
10-4
10-5
10-6
BER
25% FEC
15% FEC
7% FEC
PAM8
PAM4
EDFA
AWG
RTO
256 GSa/s
Tx-DSP
ECL
Rx-DSP
256 GSa/s
Cu-LiTaO3 MZM
FPC
BPF
VOA
PD
b
c
d
e
a

