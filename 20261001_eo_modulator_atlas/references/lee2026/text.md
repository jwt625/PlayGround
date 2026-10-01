---
paper_id: lee2026
source_url: https://arxiv.org/abs/2601.17385
doi: 10.1038/s41467-026-77674-y
license: arXiv-nonexclusive
sha256: 5de3d227fb2b4deab762c473b67a3fb4607c2b651b638662b5a8459d3c583105
pages: 15
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Suspended thin-film lithium niobate modulator for broadband
mid-infrared light modulation and frequency comb generation
Chun-Ho Lee∗1,2, Xinyi Ren∗1,2, Xinzhou Su∗2, Wonho Lee3, Zile Jiang2, Yue Yu1,2,6, Huibin Zhou2, Yue Zuo2,
Shaoyuan Ou2, Reshma Kopparapu1,2, Adam T. Heiniger4, Moshe Tur5, Alan E. Willner2, Zaijun Chen1,2, and
Mengjie Yu†1,2,6
1Department of Electrical Engineering and Computer Sciences, University of California, Berkeley, CA, USA
2Ming Hsieh Department of Electrical and Computer Engineering, University of Southern California, Los Angeles, CA, USA
3PHY research lab, Intel lab, Hillsboro, OR, USA
4TOPTICA Photonics Inc., Pittsford, NY 14534, USA
5School of Electrical Engineering, Tel Aviv University, Ramat Aviv 69978, ISRAEL
6Materials Sciences Division, Lawrence Berkeley National Laboratory, Berkeley, California 94720, USA
Abstract
The mid-infrared (MIR) spectral regime is central to applications including remote sensing, precision spec-
troscopy, higher harmonic generation, and free-space optical communication. However, coherent and broadband
MIR modulation remains challenging owing to high optical loss, limited bandwidth, and large drive voltages
in existing platforms.
Here, we overcome the challenges by deploying a suspended thin-film lithium-niobate
(TFLN) based electro-optic (EO) platform co-designed with high-performance traveling-wave microwave (MW)
electrodes. We demonstrate a record-low Vπ,DC of 2.3 to 4.3 V over a broadband MIR bandwidth from 2.4 to 3.6
µm, and a 2.7-dB EO bandwidth of 40 GHz (extracted 3-dB bandwidth of 50 GHz), yielding a figure-of-merit of
17.4 GHz/V—more than an order of magnitude higher than the state-of-the-art. We demonstrate, for the first
time, high frequency Vπ,MW of 4.5-6.5 V in the 25-35 GHz range, and frequency-agile MIR EO frequency comb
generation with a 10-dB optical bandwidth over 0.8 THz using a suspended phase modulator of 4-cm active
modulation length. We further validate the platform in a free-space optical communication link. Our results
establish a monolithic MIR photonic platform capable of powerful EO modulation and spectral synthesis, and
present a significant step towards reconfigurable MIR sensing and communication systems on chip.
1
Main text
Rapidly growing amount of data being transmitted across networks is creating a significant demand for new wire-
less communication technologies to support higher data rate, lower latency, higher connection density, and global
coverage [1]. Free space optical communication is an emerging technology for data transfer to remote assets through
atmospheric communication channels, addressing the broadband connectivity bottlenecks for space and terrestrial
applications [2, 3] while offering tremendous advantages over radio-frequency carriers due to its high bandwidth,
immunity to electromagnetic interferences, low latency, and multiplexing features.
The MIR spectral range from 3 to 5 µm and long-wave infrared (LIR) from 8 to 12 µm are actively being
investigated as free space optical carriers since it has the lower absorption in atmosphere, the higher tolerance to
adverse weather conditions (such as dust, haze, and low-altitude clouds), and less phase-front distortion by turbu-
lence effects compared to the near-infrared (NIR), mm-wave, and THz-waves [4–14]. There are recent developments
of the optoelectronic devices based on subsequent intersubband transitions and stark effect at LIR, such as quantum
cascade laser (QCL), quantum-well infrared detector (QWIP) and quantum cascade detector, however the efficiency
of unipolar device is significantly worse in the MIR region [15–19]. Existing approaches to modulate the MIR light
include direct current modulation of a QCL at a limited bandwidth of a few GHz at room temperature [16,20–24],
and nonlinear parametric conversion of the near-infrared light which suffers from limited optical conversion band-
width and efficiency as well as requires external power-hungry pump lasers [25]. Alternatively, EO material could
be used to modulate the refractive index or optical absorption via external voltages. The MIR EO modulators have
∗These authors contributed equally.
†Corresponding author: mengjie.yu@berkeley.edu
1
arXiv:2601.17385v1  [physics.optics]  24 Jan 2026

<!-- page 2 -->
been demonstrated in silicon [26], silicon on lithium niobate (LN) [8,27], titanium dioxide on LN [28], ion diffused
waveguide on LN [29], black phosphorous on silicon [30,31], germanium on silicon [32,33], and barium titanate [34].
However, none of the material platform offers beyond a few GHz modulation speed and CMOS-compatible half-wave
voltages. In addition, traditional silicon-based modulators have inevitable high optical losses due to the free carrier
absorption at longer optical wavelength. Therefore, a compact, high-speed, efficient and low-loss EO modulator in
the MIR is still missing.
Here, we present a monolithic optoelectronic platform in the MIR based on a suspended TFLN platform, capable
of both high-speed amplitude and phase modulation (Fig. 1a). The air suspended TFLN is used to overcome the
limitation of the absorption loss in the MIR from the underlying silicon dioxide layer as well as to support a tight
optical confinement in the waveguide and reduce the MW propagation loss, both of which are critical to achieve
a low switching voltage [10]. Co-designed with the optical waveguide, the segmented coplanar MW electrodes are
deployed to further reduce the ohmic loss and achieve velocity matching between the MW and optical signals,
supporting high speed EO response. The demonstrated traveling-wave-based Mach-Zehnder amplitude modulator
(AM) and double-pass phase modulator (PM) are based on (non-resonant) waveguide structures which can operate
in a broad MIR wavelength range and be compatible with integrated MIR light sources. Unlike electro-absorption
or carriers based modulators [32, 33], our modulator platform based on the Pockel’s effect not only allows for
high-fidelity data links for free space optical communication via independent amplitude and phase modulation, but
also can be used to generate broadband frequency-agile optical frequency comb [35] for spectroscopy and sensing
applications (Fig. 1b&c) [15]. Until now, none of the existing platforms have achieved broadband tunable MIR EO
frequency comb generation [11,36,37].
Figure 1d presents an optical microscope image of the integrated MIR modulator chip with a footprint of
25 × 7 mm2. The chip is fabricated on an 800-nm X-cut LN on 4.7-µm oxide on Si substrate wafer. The waveguide
has a top width of 4 µm and an etch depth of 500 nm, patterned via electron-beam lithography and ion milling (see
Methods). Air holes are then defined and etched on the remaining 300-nm LN slab via a second step of lithography
and etching, followed by the metal-electrode patterning, deposition and lift-off process. The air holes are placed in
between the metal segments and used to selectively release the bottom oxide only around the photonic waveguide via
a wet etching process, providing both good mechanical and thermal support for the electrodes. Scanning electron
microscopy (SEM) image in Fig. 1e shows the structures of the air holes, suspended optical waveguide, and the
segmented microwave electrodes.
We optimize the operation for a fundamental transverse electric (TE) mode.
Here, the smallest electrode gap of 6.5 µm along with the waveguide width of 4 µm is chosen to maximize the
EO response as well as to minimize the metal induced optical loss. The cross-section of the EO waveguide and
microwave transmission line are illustrated in Fig. 1f along with the optimized device geometry parameters after
co-design and the simulated optical mode profile at 3 µm. The chip is cleaved for edge coupling at both input and
output facets.
We first characterize the near-DC half-wave voltage, namely Vπ, of our AM device at a low modulation frequency
of 100 kHz. The suspended AM shown in Fig. 2a includes two Y-shape splitters, two asymmetric arms of a 21-
µm length difference and an active modulation length of 2 cm in a push-pull configuration. The continuous-wave
tunable optical parametric oscillator (Toptica, TOPO) is used to inject the device via a free space objective lens
at a MIR wavelength range from 2.4 to 3.6 µm. We record the modulated optical output using an AC-coupled
mercury cadmium telluride (MCT) detector while applying a saw-tooth voltage waveform to the electrodes at 100
kHz. Figure 2b plots the optical transmission as a function of applied voltage at various MIR wavelengths, featuring
the Vπ of 2.3 V at 2.4 µm to 4.3 V at 3.6 µm. The corresponding half-wave voltage length product (Vπ · L) are 4.6
V · cm at 2.4 µm and 8.6 V · cm at 3.6 µm wavelength, respectively. The Vπ scales near linearly with the optical
wavelength λ, considering that the accumulated phase shift ∆ϕ ∝Γ/λ where Γ is the mode overlap (see Methods).
As the suspended optical waveguide supports guided mode over an ultrabroad bandwidth, we tested the Vπ at the
NIR wavelength of 1.55 µm to be 1.8 V, demonstrating our integrated EO modulator operation across a 1.2-octave
span. The extinction ratio is measured to be 7 dB in the MIR using a lock-in amplifier and can be improved by
deploying a single mode waveguide at the Y splitter to suppress the scattering into undesired optical modes. We
compare the near-DC Vπ with the state-of-art reported in other MIR modulators at similar wavelength range in
Fig. 2c. This work presents the lowest Vπ ever reported in the MIR, reaching CMOS-compatible voltages, and the
broadest MIR optical bandwidth, both of which are critical for in-parallel power-efficient modulation.
Next, we demonstrate high performance of the suspended MIR modulator at microwave frequencies, including
the EO response and the MW Vπ · L. We first measure the microwave attenuation loss, reflection loss and phase
index of the segmented electrodes via a vector network analyzer in a frequency span from 10 MHz to 40 GHz,
shown in Fig. 3a-c. The MW loss is extracted from S21 and measured to be 2 dB/cm at 10 GHz and 2.75 dB/cm
at 40 GHz, respectively. The fitted loss slope coefficient is 0.26 dB/cm/GHz0.5 after air suspension (Fig. 3a). We
2

<!-- page 3 -->
discover a higher MW loss slope of 0.31 dB/cm/GHz0.5 and an increased loss of 3.75 dB/cm at 40 GHz before air
suspension of the waveguide (Fig. 3a), which suggests a potential loss channel induced by the bottom oxide layer.
In conclusion, the record MW loss is a result of a thick Au metal thickness of 800 nm, optimization of the slow-wave
T-shaped segment design, and removal of oxide layer. Secondly, Fig.3b also plots the reflection S11 below -25 dB
across 2 to 40 GHz range, indicating that the characteristic impedance of the transmission line is well matched to
50 Ωas designed. At last, Fig. 3c shows that the MW phase index, obtained by nphase,MW = (c0×group delay)/
(electrode length), matches well with the optical group index (ngrp,optical) of 2.29. The achieved high performance
electrodes contribute to a lower MW Vπ as the electrode length increases. The MW Vπ is extracted from the optical
spectrum of the driven modulator output measured by a Fourier-transform infrared spectrometer. Figure 3d plots
the measured MW Vπ of our 2-cm-long AM to be 4.5-6.5 V at 25-35 GHz frequency, which agrees well with our
simulation based on the device parameters. The EO response can be simulated from the measured MW properties
and scales inversely with the square of the microwave Vπ, which matches well with the experimental data (Fig. 3d).
The EO response drops 2.7 dB at 40 GHz MW frequency which is the frequency limit of our signal generator while
the fitted 3-dB EO BW is 50 GHz (dashed line). To the best knowledge, this is the first time that the Vπ is ever
measured at tens of GHz frequency range and presents the broadest EO bandwidth demonstrated so far in any
MIR modulators. To note, another well-adopted figure-of-merit (FOM) is 3-dB EO bandwidth divided by near-DC
Vπ, which amortizes the electrode length dependence. In our work, this FOM is achieved at 17.4 GHz/V, orders of
magnitude larger than the reported value (Fig. 3e).
In addition to the amplitude modulation, we demonstrate a suspended 2-cm-long TFLN PM and generate
frequency agile EO frequency combs in the MIR. Figure 4a shows the optical image of the fabricated PM where
we adopt a double-pass photonic structure where the waveguide is rerouted into the same transmission line after a
waveguide crossing. Therefore, the total active modulation length is achieved at 4 cm. Further leveraging the low
MW Vπ and large EO bandwidth of our platform, we generate an integrated MIR EO frequency comb with a 10-dB
optical bandwidth of 0.8 THz (20 nm) and a total comb line number of 33 at a center wavelength of 2.7 µm by
driving the PM with a single-frequency MW signal at 29.2 GHz (Fig. 4b). In addition, by driving the same device,
we achieve an EO comb spectrum at 2.36 µm with a 27.2-GHz line spacing and a similar 10-dB span of 0.8 THz
(15 nm), which shows the frequency agility of the non-resonant waveguide-based MIR EO comb generator in both
center wavelengths and comb line spacings. The total modulation index (β) we achieve are 4.2 π and 4 π at 27.2
GHz and 29.2 GHz, respectively. This is the first demonstration of integrated EO comb generation in the MIR with
tens of GHz repetition rates, without any need of parametric conversion. Complementary to the semiconductor
based combs [1, 38, 39] and Kerr microcombs in the MIR [40, 41], the EO comb sources offer unique advantages
including tunable spectral resolution, wavelength multiplexing and robust mode-locking operation for sensing and
communication applications [9].
Finally, we evaluate a free space communication link based on the suspended TFLN AM as a proof of principle
demonstration. We experimentally demonstrate an intensity modulation/direct detection (IM/DD) MIR link at
2.7 µm. Figure 5a illustrates the experimental setup (see Methods). The modulated MIR output propagates 0.5
m in the free space and is detected by the MCT detector with a 3-dB bandwidth of > 1 GHz. We first measure
the BER performance of OOK signals at various input power levels to the AM with a baud rate of 1.5 Gbaud,
as shown in Fig. 5b. The reference case (0 dB) corresponds to the maximum transmitted power. In this setup,
the system performance is primarily limited by the detector bandwidth and total loss resulting in a BER decrease
from 10−2 to 10−5 within a 2.5-dB power range. Additionally, we demonstrate higher-order intensity modulation
formats at a maximum transmitted power, including 4-level and 8-level pulse amplitude modulation (PAM-4 and
PAM-8). Eye diagrams, BER values, and Q-factors for different modulation formats and baud rates are presented
in Fig. 5(c–f) for comparison. All configurations achieved BER values below the 7% hard-decision forward error
correction (HD-FEC) threshold. The phase modulator demonstrated on the same platform further highlights its
capability for phase-modulated formats and more complex data modulation.
In conclusion, the suspended TFLN EO devices demonstrated here are prototypical building blocks for a new
generation of MIR integrated photonic architectures, expanding functionality beyond amplitude modulation to in-
clude phase modulation and frequency-comb generation. We achieved record-low Vπ at both near-DC and previously
unexplored tens-of-GHz microwave frequencies range, a broad EO bandwidth exceeding 40 GHz, and an ultra-wide
optical window covering 1.55 µm and 2.4 to 3.6 µm in the MIR. This work marks the first time that an MIR EO
modulator exceeds the centimeter-scale interaction length while achieving a modulation index exceeding 4π and a
FOM of 17.4 GHz/V for the EO bandwidth per drive voltage, more than an order of magnitude improvement over
other MIR platforms (Table 1). Leveraging these unique characteristics, we demonstrated the first, to our knowl-
edge, direct EO frequency comb generation in the MIR with a 0.8-THz span and over 33 comb lines. These results
establish TFLN as a compelling monolithic platform for scalable MIR EO systems. The suspended platform could
3

<!-- page 4 -->
be extended to operate beyond 3.6 µm to explore the full LN transparency window (up to 5 µm) by increasing the
electrode gap and applying the same co-design principle. The optical loss can be improved via further optimizing
the facet design and fabrication process with thermal and chemical treatment [42–44]. A straightforward yet crucial
next step will be to integrate in-phase/quadrature (IQ) modulator configurations for coherent MIR communication
and vector signal processing. We highlight the versatility of the TFLN platform where periodically poled TFLN
waveguides would further be integrated to down-convert the high speed MIR signal to the NIR as the receiver of
an FSOC link. Further integration of the AM and PM on the same chip will also enable MIR pulse generation and
on-chip waveform synthesis [35]. The demonstrated comb bandwidth of 0.8 THz would lead to a 1.7-ps pulse train
at 2.7 µm at a 30-GHz repetition rate via EO modelocking. On-chip EO modulation offers a path to stabilized MIR
comb sources and synchronized pulse trains [35, 45], bridging the gap between semiconductor lasers and electro-
optic frequency synthesis. With continued advancement of QCLs and chip-based parametric oscillators [11,46], we
envision a hybrid passive and active nonlinear integrated system where frequency conversion, EO modulation and
comb generation could coexist. Such systems would not only enable spectrally tailored dual-comb spectroscopy
for molecular sensing and metrology [47] but also provide multi-wavelength coherent sources for free-space optical
communication [3, 23, 48] and astronomical spectrograph calibration [49, 50]. The ability to combine low-loss EO
modulation, broadband MIR transparency, and scalable chip-level integration marks an important step toward fully
reconfigurable, multifunctional MIR photonic systems.
Figure 1:
Monolithic mid-infrared optoelectronic platform on TFLN. (a) Air-suspended MIR traveling-
wave-based amplitude and phase electro-optic modulators on the TFLN. Inset shows the cross-sectional view of
the optical layer which consists of a ridge TFLN waveguide sitting on oxide-on-Si substrate. The oxide layer is
removed underneath the waveguide area through air holes on the TFLN slab layer etched via a second etching
process. High performance optoelectronic interface in the MIR would enable high-bandwidth data links via free
space communication (b) and frequency-agile EO frequency combs for molecular spectroscopy (c). (d) Photographic
image of the fabricated MIR modulator chip on 800-nm X-cut TFLN. (e) Scanning-electron-microscopy (SEM) image
of suspended photonic waveguides and coplanar microwave transmission line. Segmented slow-wave electrode design
(zoom in, right) is applied to reduce the microwave loss and achieve impedance matching and velocity matching
with optical field while compatible with a low optical propagation loss. Air hole arrays are optimized and placed
between the metal segments for releasing the adjacent photonic waveguides. (f) Cross section of the suspended
device where the metal gap (g), waveguide height (hLN), slab thickness (hslab), metal thickness (hmetal), waveguide
width (w0), the air gap under the waveguide (hair) and the segmented electrode parameters (s , t, r , l ) are 6.5,
0.8, 0.25, 0.8, 4, 4.7 µm, and (0.5, 6.5, 0.5, 45) µm, respectively. The SEM image of the waveguide facet is shown.
The simulated transverse-electric optical mode profile at 3 µm is plotted.
4

<!-- page 5 -->
Figure 2:
Characterization of near-DC half-wave voltage Vπ of a 2-cm-long air-suspended amplitude
modulator. (a) Optical microscope images. Two optical path in the AM have a different length of 21 µm to enable
bias point tuning via varying the optical wavelength. (b) Normalized optical transmission as a function of applied
voltage at the different MIR wavelengths as well as at the 1.55 µm in the NIR. The measured half-wave voltage
(Vπ) at 100 kHz is 2.3 - 4.26 V across the optical span from 2.4 µm to 3.6 µm, which indicates approximately linear
dependence of the optical operational wavelength. In addition, the suspended AM device is measured across more
than an octave optical bandwidth with a half-wave voltage of 1.8 V at 1.55 µm. (c) Comparison of Vπ with other
MIR EO platforms. The LN amplitude modulator shows the lowest Vπ as well as Vπ·L of 4.6 V·cm at 2.4 µm and
8.6 V·cm at 3.6 µm.
5

<!-- page 6 -->
Figure 3: Characterization of the MIR modulator up to 40-GHz microwave frequencies. (a) Measured
microwave loss on the segmented co-planar electrodes before and after air suspension.
Segmented travel-wave
electrodes combined with air suspension lead to a reduced MW propagation loss of 2.75 dB/cm at 40 GHz and
an ohmic-loss-limited slope of 0.26 dB/cm/GHz0.5 , which is comparable to the loss slope of the best NIR LN
modulators reported [51]. (b) Measured electrical transmission S21 and reflection S11 spectrum. The measured
reflection is below 25 dB from 2 - 40 GHz range indicating a well-matched impedance to 50 Ω. (c) Measured
MW phase index, which matches well with the optical group index based on the air-suspended TFLN waveguide
(dashed line). (d) Microwave Vπ and the corresponding electro-optic response, referenced to the performance at
2 GHz. Microwave Vπ are measured to be 5 V at 27 GHz and 5.26 V at 34 GHz for a 2-cm-long EO AM at the
optical wavelength of 2.7 µm. The microwave Vπ is extracted via the optical spectrum from the AM under MW
driving. The measurement results match with the simulation based on the measured S parameters in (a). Electro-
optic response scales inversely with the square of the microwave Vπ and drops 2.7 dB at 40 GHz MW frequency
which is the frequency limit of our signal generator while the extracted 3-dB EO BW is 50 GHz (dashed line). (e)
Comparison of the modulator figure of merit, defined as the ratio between the 3-dB EO bandwidth (BW) and the
near-DC Vπ. The AM in our work demonstrates significantly higher BW/Vπ values of 17.4 GHz/V, as compared to
the literature values (the dashed lines indicate constant BW/Vπ values).
6

<!-- page 7 -->
Figure 4: Frequency-agile MIR electro-optic frequency combs based on an integrated recycled phase
modulator. (a) Microscopic image of the suspended double passing phase modulator with the total modulation
length of 4 cm. (b) Broadband EO frequency comb generation using the same PM device at two different MIR
wavelengths of 2.36 µm and 2.7 µm. Total modulation indexes of 4.2 π and 4 π at MW driving frequencies of 27.2
and 29.2 GHz were measured at 2.36 µm and 2.7 µm pump wavelength, respectively. This corresponds to a 10-dB
optical bandwidth of 0.8 THz.
Figure 5: Experimental MIR communication link using an air-suspended TFLN intensity modulator.
(a) Experimental setup for measuring the communication link at 2.7 µm wavelength. OPO: optical parametric
oscillator, HWP: half-wave plate, AWG: arbitrary waveform generator, MCT detector: mercury cadmium telluride
detector, DSO: digital storage oscilloscope. (b) Measured BER curve versus transmitted optical power for 1.5-Gbaud
on-off keying (OOK) signal. (c-f) Experimental results of eye diagram, BER values, and Q-factors of (c) 1.5-Gbaud
OOK, (d) 0.5-Gbaud Pulse Amplitude Modulation (PAM)-4, (e) 1.5-Gbaud PAM-4, (f) 0.5-Gbaud PAM-8. The
achievable data rate is limited by the 3-dB bandwidth of our MIR detector (1 GHz).
7

<!-- page 8 -->
Table 1: Comparison of mid-infrared electro-optic modulators.
Wavelength
(µm)
Platform
design
Scheme Function
L
(cm) Vπ,DC(V) Vπ,MW (V)
Optical
loss
(dB/cm)
MW loss
(dB/cm)
EO
BW
EO BW
per Vπ,DC
(GHz/V)
Application
Ref.
2.4
–3.6
Air-suspended
TFLN, MZI
Pockels
AM,
PM,
combs
2
2.3-4.2
4.5-6.5b
2.2-2.8
2.75
@40GHz
40 GHzc
50 GHzd
17.4
EO comb 0.8-THz span
Mod index of 4π
OOK@1.5 Gbaud
PAM8@1.5 Gbaud
This
work
2.5
TiO2-on-LN,
WG
Pockels
AM
NA
NA
NA
NA
NA
NA
NA
NA
[28]
2.6
BTO-on-oxide,
WG
Pockels
AM
0.2
55
NA
2.3
NA
NA
NA
NA
[34]
3.39
Si-on-LN,
MZI
Pockels
AM
0.5
52
NA
2.5
NA
>23 kHza 4.4×10−7
NA
[8]
3.39
Ti-diffused LN,
MZI
Pockels
AM
0.8
31
NA
NA
NA
1.8 GHz
0.058
NA
[29]
3.78
Si-on-LN,
MZI
Pockels
AM
0.6
20.8
NA
4.5
NA
>5 MHza
0.00024
NA
[27]
3.8
Si-on-SiO2,
MZI
TO
AM
NA
NA
NA
3.5
NA
23.8 kHz
NA
NA
[26]
3.85
–4.1
BP-Si-on-SiO2,
WG
EA
AM
NA
NA
NA
NA
NA
>60 kHz
NA
NA
[30]
3.8
Ge-on-Si,
MZI/EAM
Cl/CA
AM
0.1
4.7
NA
NA
NA
>60 MHz
0.013
OOK@60 MHz
[33]
3.72
–3.8
Si-on-oxide,
MZI/EAM
Cl/CA
AM
NA
NA
NA
11.5
NA
NA
NA
OOK@125 Mbit/s
[31]
3.95
–4.3
TFLN-on-Al2O3,
MZI
Pockels
AM
0.8
28
NA
2.2
NA
20 GHz
0.71
OOK@10 Gbit/s
[36]
3.3
–3.8
Air-clad TFLN,
cavity
Pockels
AM
0.15
51
NA
4
NA
NA
NA
NA
[37]
ER: extinction ratio; EO: electro-optic; TO: thermo-optic; EA: electro-absorption; BP: black phosphorus; EAM: electro-absorption
modulator; CI: free carrier injection; CA: free carrier absorption. Mod: modulation
a) The measured bandwidths are limited by the PD speed.
b) Vπ measured at 2.7 µm wavelength and 25-35 GHz MW frequency.
c) 2.7 dB EO BW. Limited by the VNA performance.
d) Extrapolated.
Note: During the preparation of this manuscript, a related work was posted online [36] which performance are cited in this table.
2
Methods
2.1
Device Fabrication
To support guided modes in the MIR spectrum, we patterned waveguides using electron beam (e-beam) lithography
with a thick hydrogen silsesquioxane (HSQ) resist and dry etched 500 nm of lithium niobate (LN) using reactive
ion etching (RIE). Air holes were patterned near the LN waveguides using direct laser writing and etched with
RIE through a photoresist mask. Subsequently, 800 nm-thick electrodes were patterned with e-beam lithography,
followed by metal deposition using a thermal evaporator and a lift-off process. The underlying SiO2 layer was then
etched using a buffered oxide etch (BOE) solution.
2.2
Wavelength dependent near-DC Vπ
∆βTE = 2ne ∆ne
2βTE
k2
0 = −n4
er33Γmo
2βTE
k2
0
(1)
Vπ · L =
π
∆βTE
× 100 (V · cm)
(2)
8

<!-- page 9 -->
Definitions:
• ∆βTE: total phase change per meter per volt for TE guided mode.
• k0: wavevector in vacuum.
• βTE: propagation constant of the guided mode.
• ne: extraordinary refractive index of LiNbO3 (LN).
• r33: electro-optic (EO) coefficient.
• Γmo: mode overlap factor.
Dispersive DC Vπ is studied using COMSOL finite element method numerical simulation using equation (1) and
(2). Extended Data Figure 1a shows comparison between measurement and simulation results. Trenching effect
at the bottom of waveguide sidewall is included in the simulation. Measured DC Vπ values matches well to the
simulation in all wavelength ranges. We note that a higher Vπ is expected at a larger optical wavelength due to the
smaller wave-vector as well as a smaller mode overlap between optical and microwave fields (Extended Data Fig.
1b).
Extended Data Fig. 1: Analysis for wavelength-dependent DC Vπ.
(a) DC Vπ measurement and simulation for the wavelength range of 2.4–3.6 µm. (b) Simulated mode overlap
factor and other parameters used for ∆βTE calculation.
2.3
Extinction ratio analysis
The extinction ratio is measured using a chopper, a MIR photodetector (PVI-4TE-8), and a lock-in amplifier as the
AWG sweeps the applied voltage on the AM. After the sinusoidal fitting, the extinction ratio is 7.1 dB based on
the maximum and minimum transmission values (Extended Data Fig. 2). The extinction ratio is primarily limited
by the higher-order mode coupling to the slab at the second Y-splitter/merger, which can be improved by using a
single-mode waveguide width of 0.9 µm instead of 2 µm in our case.
2.4
Communication Setup
A TOPTICA continuous-wave (CW) optical parametric oscillator (OPO) is utilized in our system as a mid-infrared
(MIR) source. Using a 1064 nm CW pump laser, this OPO can generate tunable ∼1 W MIR power within the range
of 2.5–4 µm with a laser linewidth of approximately 500 kHz. The free-space output of the OPO is coupled into the
integrated Mach–Zehnder amplitude modulator (AM) using a MIR collimating lens.
An arbitrary waveform generator (AWG) with a sampling rate of 20 GSa/s, followed by an electrical amplifier
with 26 dB gain, is used to apply various intensity modulation signals to the AM. The electrical signal is delivered to
9

<!-- page 10 -->
Extended Data Fig. 2: Extinction measurement of the air-suspended amplitude modulator at 2.7 µm wavelength.
the device electrode through a 67 GHz-bandwidth RF probe. After free-space propagation of approximately 0.5 m,
the output of the modulator is focused onto a MIR mercury cadmium telluride (MCT) detector for detection. This
MCT detector has a bandwidth of about 1 GHz.
The detector output is recorded by a digital storage oscilloscope (DSO) for data recovery. Specifically, offline
digital signal processing (DSP) is applied to compensate for system degradation in multi-level pulse amplitude
modulation (PAM) signals. A 20-tap adaptive filter based on recursive least-squares (RLS) equalization with a
training sequence is used to improve signal quality. No additional DSP equalization is applied to the on–off keying
(OOK) signal in this measurement.
2.5
Insertion Loss Analysis
We use two free-space objective lenses (Thorlabs, C036TME-E) to couple the MIR light into and out of the devices
over a wavelength range of 2.4–3.6 µm, with an off-chip optical power ranging from 60 to 700 mW at the chip
facet. Extended Data Table 1 summarizes the optical losses measured from both the suspended device used in
the main manuscript and non-suspended reference devices (air clad) of identical dimensions.
The passive and
active propagation losses were extracted from waveguides, amplitude modulators and recycled phase modulators
with different waveguide lengths. As shown in the Extended Data Table 1, the optical loss at 2.4 µm remains
comparable before and after suspension, while the suspended devices exhibit noticeably lower loss at 2.7, 2.8 and
3.6 µm compared to the non-suspended counterpart. This result agrees well with the spectral absorption feature
of SiO2 in the mid-infrared where the absorption has a peak around 2.7-2.8 µm and increases starting at 3.1 µm.
Even though our device is designed and measured up to 3.6µm, the suspended TFLN approach is expected to
continuously reduce the propagation loss beyond 3.6µm up to the end of the transmission window of LN at 4.5-5
µm.
However, the active waveguides, which include nearby metal electrodes for electro-optic modulation, show higher
loss than the passive sections. The measured active propagation losses at 2.4 µm and 3.6 µm are 6.4 dB/cm and
10.5 dB/cm, respectively. The exact cause of this additional attenuation is still under investigation as the simulated
metal induced loss is 0.2 and 4.8 dB/cm respectively.
One of the possible reason is the high energy electron
injection during the ebeam lithography of the metal electrodes nearby the waveguide which induces higher free
carriers loss at higher optical wavelength, which could be improved in the future via controlled annealing condition
and photolithography.
3
Data availability
The data that support the findings of this study are available from the corresponding author upon reasonable
request.
10

<!-- page 11 -->
Extended Data Table 1. Measured propagation loss of the fabricated devices.
Wavelength (µm)
Propagation loss
passive (dB/cm)
Suspended
2.4
2.2
2.7
1.2
2.8
4.6
3.6
2.8
Non-suspended
2.4
2.4
2.7
7.2
2.8
11.4
3.6
4.3
4
Acknowledgements
The authors thank Alexander Gaeta and Yun Zhao for providing the mid-infrared photodetector. This work is
supported by the Optica Foundation, the Chan Zuckerberg Initiative Foundation(Dynamic Imaging, 2023-321175),
and the DARPA Young Faculty Award (D23AP00252-02). M.Y. and Y.Y. are supported by the U.S. Department of
Energy, Office of Science, Basic Energy Sciences, Materials Sciences and Engineering Division under Contract No.
DE-AC02-05CH11231 within the Quantum Coherent Systems Program KCAS26. AW is supported by the Office of
Navel Research through MURI Award(N00014-20-1-2558) and Airbus Institute for Engineering Research. Device
fabrication was performed at the John O’Brien Nanofabrication Laboratory at Universtity of Southern California.
The views, opinions and/or findings expressed are those of the authors and should not be interpreted as representing
the official views or policies of the Department of Defense or the U.S. Government.
5
Author contributions
M.Y. conceived the conceptual idea. C.L. designed the chip and fabricated the devices. C.L., X.R. and X.S. designed
the experiment and carried out the measurements. C.L., X.R., X.S. and M.Y. analyzed the data with the help of
W.L., Z.J., Y.Y., H.Z., Y.Z., S.O., R.K., A.T.H., M.T., A.E.W., and Z.C.. C.L., X.R., X.S. and M.Y. drafted and
revised the manuscript with contribution from all authors. M.Y., Z.C. and A.E.W. supervised the project.
6
Competing interests
C. L., Z.C. and M.Y. are involved in developing lithium niobate technologies at Opticore Inc..
11

<!-- page 12 -->
References
[1] Dmitry Kazakov, Theodore P. Letsou, Marco Piccardo, Lorenzo L. Columbo, Massimo Brambilla, Franco Prati,
Sandro Dal Cin, Maximilian Beiser, Nikola Opačak, Pawan Ratra, Michael Pushkarsky, David Caffey, Timothy
Day, Luigi A. Lugiato, Benedikt Schwarz, and Federico Capasso. Driven bright solitons on a mid-infrared laser
chip. Nature, 641(8061):83–89, 2025.
[2] Farzana I. Khatri, Zachary Gonnsen, Jade P. Wang, Christian Rivera Rivera, Jamie Burnside, Richard L.
Butler, Jesse Chang, Jean-Pierre Chamoun, Benjamin Croop, Nicholas I. Cummings, Catherine E. DeVoe,
Nicholaas du Toit, Jacob M. Gregory, Alan G. Hylton, Mahima Kaushik, Samuel S. Larson, Olga Mikulina,
John D. Moores, Sabino Piazzolla, Patricia Randazzo, Thomas E. Roberts, Jennifer A. Sager, Suzanne E.
Smith, Neal W. Spellmeyer, Kathy Strickler, Jeffrey D. Towns, John J. Veselka, Douglas T. Ward, James
Torres, Jonathan R. Woodward, Miriam D. Wennersten, David J. Israel, Glenn B. Jackson, and Bryan S.
Robinson. Experimental results from integrated LCRD low-earth orbit user modem and amplifier terminal
(ILLUMA-T) program. Free-Space Laser Communications XXXVII, 13355:1335506–1335506–10, 2025.
[3] Kaiheng Zou, Kai Pang, Hao Song, Jintao Fan, Zhe Zhao, Haoqian Song, Runzhou Zhang, Huibin Zhou, Amir
Minoofar, Cong Liu, Xinzhou Su, Nanzhe Hu, Andrew McClung, Mahsa Torfeh, Amir Arbabi, Moshe Tur,
and Alan E. Willner. High-capacity free-space optical communications using wavelength- and mode-division-
multiplexing in the mid-infrared region. Nature Communications, 13(1):7662, 2022.
[4] Robert A. McClatchey. Atmospheric Transmission Models And Measurements. Atmospheric Effects on Radia-
tive Transfer, pages 2–6, 1979.
[5] Christian Rosenberg Petersen, Uffe Møller, Irnis Kubat, Binbin Zhou, Sune Dupont, Jacob Ramsay, Trevor
Benson, Slawomir Sujecki, Nabil Abdel-Moneim, Zhuoqi Tang, David Furniss, Angela Seddon, and Ole Bang.
Mid-infrared supercontinuum covering the 1.4–13.3 µm molecular fingerprint region using ultra-high NA chalco-
genide step-index fibre. Nature Photonics, 8(11):830–834, 2014.
[6] Jordan Goldstein, Hongtao Lin, Skylar Deckoff-Jones, Marek Hempel, Ang-Yu Lu, Kathleen A. Richardson,
Tomás Palacios, Jing Kong, Juejun Hu, and Dirk Englund. Waveguide-integrated mid-infrared photodetection
using graphene on a scalable chalcogenide glass platform. Nature Communications, 13(1):3915, 2022.
[7] Abhilasha Kamboj, Leland Nordin, Aaron J Muhowski, David Woolf, and Daniel Wasserman.
Room-
Temperature Mid-Wave Infrared Guided-Mode Resonance Detectors.
IEEE Photonics Technology Letters,
34(11):615–618, 2022.
[8] Jeff Chiles and Sasan Fathpour. Mid-infrared integrated waveguide modulators based on silicon-on-lithium-
niobate photonics. Optica, 1(5):350–355, 2014.
[9] Ming Yan, Pei-Ling Luo, Kana Iwakuni, Guy Millot, Theodor W Hänsch, and Nathalie Picqué. Mid-infrared
dual-comb spectroscopy with electro-optic modulators. Light: Science & Applications, 6(10):e17076–e17076,
2017.
[10] Jatadhari Mishra, Timothy P McKenna, Edwin Ng, Hubert S Stokowski, Marc Jankowski, Carsten Langrock,
David Heydari, Hideo Mabuchi, M M Fejer, and Amir H Safavi-Naeini.
Mid-infrared nonlinear optics in
thin-film lithium niobate on sapphire. Optica, 8(6):921, 2021.
[11] Alexander Y Hwang, Hubert S Stokowski, Taewon Park, Marc Jankowski, Timothy P McKenna, Carsten Lan-
grock, Jatadhari Mishra, Vahid Ansari, Martin M Fejer, and Amir H Safavi-Naeini. Mid-infrared spectroscopy
with a broadly tunable thin-film lithium niobate optical parametric oscillator. Optica, 10(11):1535, 2023.
[12] Luis Ledezma, Arkadev Roy, Luis Costa, Ryoto Sekine, Robert Gray, Qiushi Guo, Rajveer Nehra, Ryan M
Briggs, and Alireza Marandi. Octave-spanning tunable infrared parametric oscillators in nanophotonics. Science
Advances, 9(30):eadf9711, 2023.
[13] Kiyoung Ko, Daewon Suk, Dohyeong Kim, Soobong Park, Betul Sen, Dae-Gon Kim, Yingying Wang, Shixun
Dai, Xunsi Wang, Rongping Wang, Byung Jae Chun, Kwang-Hoon Ko, Peter T. Rakich, Duk-Yong Choi, and
Hansuek Lee. A mid-infrared Brillouin laser using ultra-high-Q on-chip resonators. Nature Communications,
16(1):2707, 2025.
12

<!-- page 13 -->
[14] Nima Nader, Abijith Kowligy, Jeff Chiles, Eric J Stanton, Henry Timmers, Alexander J Lind, Flavio C Cruz,
Daniel M B Lesko, Kimberly A Briggman, Sae Woo Nam, Scott A Diddams, and Richard P Mirin. Infrared fre-
quency comb generation and spectroscopy with suspended silicon nanophotonic waveguides. Optica, 6(10):1269,
2019.
[15] Nazanin Hoghooghi, Peter Chang, Scott Egbert, Matt Burch, Rizwan Shaik, Scott A Diddams, Patrick Lynch,
and Gregory B Rieker. GHz repetition rate mid-infrared frequency comb spectroscopy of fast chemical reactions.
Optica, 11(6):876, 2024.
[16] A. Soibel, M.W. Wright, W.H. Farr, S.A. Keo, C.J. Hill, R.Q. Yang, and H.C. Liu. Midinfrared Interband
Cascade Laser for Free Space Optical Communication. IEEE Photonics Technology Letters, 22(2):121–123,
2010.
[17] Tatsuo Dougakiuchi, Akio Ito, Masahiro Hitaka, Kazuue Fujita, and Masamichi Yamanishi. Ultimate response
time in mid-infrared high-speed low-noise quantum cascade detectors. Applied Physics Letters, 118(4):041101,
2021.
[18] Hamza Dely, Thomas Bonazzi, Olivier Spitz, Etienne Rodriguez, Djamal Gacemi, Yanko Todorov, Konstantinos
Pantzas, Grégoire Beaudoin, Isabelle Sagnes, Lianhe Li, Alexander Giles Davies, Edmund H. Linfield, Frédéric
Grillot, Angela Vasanelli, and Carlo Sirtori. 10 Gbit s−1 Free Space Data Transmission at 9 µm Wavelength
With Unipolar Quantum Optoelectronics. Laser & Photonics Reviews, 16(2), 2022.
[19] Stefano Pirotta, Ngoc-Linh Tran, Arnaud Jollivet, Giorgio Biasiol, Paul Crozat, Jean-Michel Manceau, Adel
Bousseksou, and Raffaele Colombelli. Fast amplitude modulation up to 1.5 GHz of mid-IR free-space beams
at room-temperature. Nature Communications, 12(1):799, 2021.
[20] Borislav Hinkov, Andreas Hugi, Mattias Beck, and Jérôme Faist. Rf-modulation of mid-infrared distributed
feedback quantum cascade lasers. Optics Express, 24(4):3294–3312, 2016.
[21] Xiaodan Pang, Oskars Ozolins, Lu Zhang, Richard Schatz, Aleksejs Udalcovs, Xianbin Yu, Gunnar Jacobsen,
Sergei Popov, Jiajia Chen, and Sebastian Lourdudoss.
Free-Space Communications Enabled by Quantum
Cascade Lasers. physica status solidi (a), 218(3), 2020.
[22] Hossein Lotfi, Lu Li, Lin Lei, Hao Ye, S. M. Shazzad Rassel, Yuchao Jiang, Rui Q. Yang, Tetsuya D. Mishima,
Michael B. Santos, James A. Gupta, and Matthew B. Johnson. High-frequency operation of a mid-infrared
interband cascade system at room temperature. Applied Physics Letters, 108(20):201101, 2016.
[23] Xiaodan Pang, Richard Schatz, Mahdieh Joharifar, Aleksejs Udalcovs, Vjaceslavs Bobrovs, Lu Zhang, Xian-
bin Yu, Yan-Ting Sun, Gregory Maisons, Mathieu Carras, Sergei Popov, Sebastian Lourdudoss, and Oskars
Ozolins. Direct Modulation and Free-Space Transmissions of up to 6 Gbps Multilevel Signals With a 4.65-
µm$m Quantum Cascade Laser at Room Temperature. Journal of Lightwave Technology, 40(8):2370–2377,
2022.
[24] David J Benirschke, Ningren Han, and David Burghoff. Frequency comb ptychoscopy. Nature communications,
12(1):4244, 2020.
[25] Alan E. Willner, Huibin Zhou, Yuxiang Duan, Zile Jiang, Muralekrishnan Ramakrishnan, Xinzhou Su, Kaiheng
Zou, and Kai Pang. Advances in Multi-Channel Mid-IR Free-Space Optical Communications. IEEE/OSA
Journal of Lightwave Technology, 42(19):6739–6748, 2024.
[26] Milos Nedeljkovic, Stevan Stankovic, Colin J. Mitchell, Ali Z. Khokhar, Scott A. Reynolds, David J. Thomson,
Frederic Y. Gardes, Callum G. Littlejohns, Graham T. Reed, and Goran Z. Mashanovich. Mid-Infrared Thermo-
Optic Modulators in SoI. IEEE Photonics Technology Letters, 26(13):1352–1355, 2014.
[27] Siyu Xu, Zhihao Ren, Bowei Dong, Jingkai Zhou, Weixin Liu, and Chengkuo Lee.
Mid-Infrared Silicon-
on-Lithium-Niobate Electro-Optic Modulators Toward Integrated Spectroscopic Sensing Systems. Advanced
Optical Materials, 11(4), 2023.
[28] Tiening Jin, Junchao Zhou, and Pao Tai Lin. Mid-Infrared Electro-Optical Modulation Using Monolithically
Integrated Titanium Dioxide on Lithium Niobate Optical Waveguides. Scientific Reports, 9(1):15130, 2019.
13

<!-- page 14 -->
[29] R A Becker, R H Rediker, and T A Lind. Wide-bandwidth guided-wave electro-optic intensity modulator at
λ=3.39 µm. Applied Physics Letters, 46(9):809–811, 1985.
[30] L. Huang, B. Dong, Z.G. Yu, J. Zhou, Y. Ma, Y.-W. Zhang, C. Lee, and K.-W. Ang. Mid-infrared modulators
integrating silicon and black phosphorus photonics. Materials Today Advances, 12:100170, 2021.
[31] M Nedeljkovic, C G Littlejohns, A Z Khokhar, M Banakar, W Cao, J Soler Penades, D T Tran, F Y Gardes,
D J Thomson, G T Reed, H Wang, and G Z Mashanovich. Silicon-on-insulator free-carrier injection modulators
for the mid-infrared. Optics Letters, 44(4):915, 2019.
[32] Aditya Malik, Sarvagya Dwivedi, Liesbet Van Landschoot, Muhammad Muneeb, Yosuke Shimura, Guy Lepage,
Joris Van Campenhout, Wendy Vanherle, Tinneke Van Opstal, Roger Loo, and Gunther Roelkens. Ge-on-Si
and Ge-on-SOI thermo-optic phase shifters for the mid-infrared. Optics Express, 22(23):28479–28488, 2014.
[33] Tiantian Li, Milos Nedeljkovic, Nannicha Hattasan, Wei Cao, Zhibo Qu, Callum G Littlejohns, Jordi Soler Pe-
nades, Lorenzo Mastronardi, Vinita Mittal, Daniel Benedikovic, David J Thomson, Frederic Y Gardes, Hequan
Wu, Zhiping Zhou, and Goran Z Mashanovich. Ge-on-Si modulators operating at mid-infrared wavelengths up
to 8 µm. Photonics Research, 7(8):828, 2019.
[34] Tiening Jin and Pao Tai Lin. Efficient Mid-Infrared Electro-Optical Waveguide Modulators Using Ferroelectric
Barium Titanate. IEEE Journal of Selected Topics in Quantum Electronics, 26(5):1–7, 2019.
[35] Mengjie Yu, David Barton III, Rebecca Cheng, Christian Reimer, Prashanta Kharel, Lingyan He, Linbo Shao,
Di Zhu, Yaowen Hu, Hannah R. Grant, Leif Johansson, Yoshitomo Okawachi, Alexander L. Gaeta, Mian
Zhang, and Marko Lončar.
Integrated femtosecond pulse generator on thin-film lithium niobate.
Nature,
612(7939):252–258, 2022.
[36] Pierre Didier, Prakhar Jain, Mathieu Bertrand, Jost Kellner, Oliver Pitz, Zhecheng Dai, Mattias Beck, Baile
Chen, Jérôme Faist, and Rachel Grange. Integrated thin film lithium niobate mid-infrared modulator. arXiv,
2025.
[37] Hyeon Hwang, Kiyoung Ko, Mohamad Reza Nurrahman, Kiwon Moon, Jung Jin Ju, Sang-Wook Han, Ho-
joong Jung, Min-Kyo Seo, and Hansuek Lee. A wide-spectrum mid-infrared electro-optic intensity modulator
employing a two-point coupled lithium niobate racetrack resonator. APL Photonics, 10(1):016116, 2025.
[38] Ina Heckelmann, Mathieu Bertrand, Alexander Dikopoltsev, Mattias Beck, Giacomo Scalari, and Jérôme Faist.
Quantum walk comb in a fast gain laser. Science, 382(6669):434–438, 2023.
[39] Tianyi Zeng, Yamac Dikmelik, Feng Xie, Kevin Lascola, David Burghoff, and Qing Hu.
Ultrabroadband
air-dielectric double-chirped mirrors for laser frequency combs. Light, Science & Applications, 14(1):280, 2025.
[40] Mengjie Yu, Yoshitomo Okawachi, Austin G Griffith, Michal Lipson, and Alexander L Gaeta. Mode-locked
mid-infrared frequency combs in a silicon microresonator. Optica, 3(8):854, 2016.
[41] Mengjie Yu, Yoshitomo Okawachi, Austin G Griffith, Nathalie Picqué, Michal Lipson, and Alexander L Gaeta.
Silicon-chip-based mid-infrared dual-comb spectroscopy. Nature communications, 9(1):1869, 2017.
[42] Wanghua Zhu, Chunyu Deng, Dongyu Wang, Qichao Wang, Yaohui Sun, Jin Wang, Binfeng Yun, Guohua Hu,
and Yiping Cui. Broadband and easily fabricated double-tip edge coupler based on thin-film lithium niobate
platform. Optics Communications, 573:131031, 2024.
[43] Amirhassan Shams-Ansari, Guanhao Huang, Lingyan He, Zihan Li, Jeffrey Holzgrafe, Marc Jankowski, Mikhail
Churaev, Prashanta Kharel, Rebecca Cheng, Di Zhu, Neil Sinclair, Boris Desiatov, Mian Zhang, Tobias J.
Kippenberg, and Marko Lončar. Reduced material loss in thin-film lithium niobate waveguides. APL Photonics,
7(8):081301, 2022.
[44] Lingyan He, Mian Zhang, Amirhassan Shams-Ansari, Rongrong Zhu, Cheng Wang, and Lončar Marko. Low-
loss fiber-to-chip interface for lithium niobate photonic integrated circuits. Optics Letters, 44(9):2314, 2019.
[45] Qiushi Guo, Benjamin K Gutierrez, Ryoto Sekine, Robert M Gray, James A Williams, Luis Ledezma, Luis
Costa, Arkadev Roy, Selina Zhou, Mingchen Liu, and Alireza Marandi. Ultrafast mode-locked laser in nanopho-
tonic lithium niobate. Science, 382(6671):708–713, 2023.
14

<!-- page 15 -->
[46] Bao-Te Chen, Shin-Lin Tsai, Xiang Wang, Hsing-Chih Liang, and Chun-Yu Cho.
Low-threshold dual-
wavelength CW mid-IR laser from shared intracavity quasi-phase-matched OPO. Optics letters, 48(7):1770–
1773, 2023.
[47] Amirhassan Shams-Ansari, Mengjie Yu, Zaijun Chen, Christian Reimer, Mian Zhang, Nathalie Picqué, and
Marko Lončar. Thin-film lithium-niobate electro-optic platform for spectrally tailored dual-comb spectroscopy.
Communications Physics, 5(1):88, 2022.
[48] Yulong Su, Jiacheng Meng, Tingting Wei, Zhuang Xie, Shuaiwei Jia, Wenlong Tian, Jiangfeng Zhu, and Wei
Wang. 150 Gbps multi-wavelength FSO transmission with 25-GHz ITU-T grid in the mid-infrared region.
Optics Express, 31(9):15156, 2023.
[49] Pooja Sekhar, Molly Kate Kreider, Connor Fredrick, Joe P Ninan, Chad F Bender, Ryan Terrien, Suvrath
Mahadevan, and Scott A Diddams.
Tunable 30 GHz laser frequency comb for astronomical spectrograph
characterization and calibration. Optics Letters, 49(21):6257, 2024.
[50] Andrew J Metcalf, Tyler Anderson, Chad F Bender, Scott Blakeslee, Wesley Brand, David R Carlson,
William D Cochran, Scott A Diddams, Michael Endl, Connor Fredrick, Sam Halverson, Daniel D Hickstein,
Fred Hearty, Jeff Jennings, Shubham Kanodia, Kyle F Kaplan, Eric Levi, Emily Lubar, Suvrath Mahadevan,
Andrew Monson, Joe P Ninan, Colin Nitroy, Steve Osterman, Scott B Papp, Franklyn Quinlan, Larry Ramsey,
Paul Robertson, Arpita Roy, Christian Schwab, Steinn Sigurdsson, Kartik Srinivasan, Gudmundur Stefansson,
David A Sterner, Ryan Terrien, Alex Wolszczan, Jason T Wright, and Gabriel Ycas. Stellar spectroscopy in
the near-infrared with a laser frequency comb. Optica, 6(2):233, 2019.
[51] Prashanta Kharel, Christian Reimer, Kevin Luke, Lingyan He, and Mian Zhang. Breaking voltage–bandwidth
limits in integrated lithium niobate modulators using micro-structured electrodes. Optica, 8(3):357, 2021.
15

