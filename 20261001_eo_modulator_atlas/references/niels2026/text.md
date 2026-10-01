---
paper_id: niels2026
source_url: https://doi.org/10.1038/s41566-025-01832-9
doi: 10.1038/s41566-025-01832-9
license: journal=https://www.springernature.com/gp/researchers/text-and-data-mining; arxiv=CC-BY-4.0
sha256: 2284137c41db1359e38b15653bafb15f4f46978d3fdd20f6e053206f77b45915
pages: 9
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
A high-speed heterogeneous lithium tantalate silicon
photonics platform
Margot Niels1,2,∗,†, Tom Vanackere1,2,∗,†, Ewoud Vissers1,2, Tingting Zhai1,2, Patrick Nenezic1,2, Jakob Declercq2,3,
C´edric Bruynsteen2,3, Shengpu Niu2,3, Arno Moerman2,3, Olivier Caytan2,3, Nishant Singh2,3, Sam Lemey2,3, Xin
Yin2,3, Soe Janssen2, Peter Verheyen2, Neha Singh2, Dieter Bode2, Martin Davi2, Filippo Ferraro2, Philippe Absil2,
Sadhishkumar Balakrishnan2, Joris Van Campenhout2, G¨unther Roelkens1,2, Bart Kuyken1,2,∗, Maximilien Billet1,2.∗
1Department of Information Technology (INTEC), Photonics Research Group, Ghent University–imec, 9052 Ghent, Belgium.
2imec, Kapeldreef 75, 3001 Leuven, Belgium.
3Department of Information Technology (INTEC), IDLab, Ghent University–imec, 9052 Ghent, Belgium.
†These authors contributed equally to this work.
∗Margot.Niels@UGent.be, Tom.Vanackere@UGent.be, Bart.Kuyken@UGent.be, Maximilien.Billet@UGent.be
The rapid expansion of cloud computing and articial intelligence has driven the
demand for faster optical components in data centres to unprecedented levels. A key
advancement in this eld is the integration of multiple photonic components onto a
single chip, enhancing the performance of optical transceivers. Here, silicon photonics,
beneting from mature fabrication processes, has gained prominence in both academic
research and industrial applications.
The platform combines modulators, switches,
photodetectors and low-loss waveguides on a single chip. However, emerging standards
like 1600ZR+ potentially exceed the capabilities of silicon-based modulators.
To
address these limitations, thin-lm lithium niobate has been proposed as an alternative
to silicon photonics, oering a low voltage-length product and exceptional high-speed
modulation properties.
More recently, the rst demonstrations of thin-lm lithium
tantalate circuits have emerged, addressing some of the disadvantages of lithium
niobate enabling a reduced bias drift and enhanced resistance to optical damage.
As such, making it a promising candidate for next-generation photonic platforms.
However, a persistent drawback of such platforms is the lithium contamination, which
complicates integration with CMOS fabrication processes.
Here, we present for the
rst time the integration of lithium tantalate onto a silicon photonics chip.
This
integration is achieved without modifying the standard silicon photonics process design
kit.
Our device achieves low half-wave voltage (3.5 V), low insertion loss (2.9 dB)
and high-speed operation (> 70 GHz), paving the way for next-gen applications. By
minimising lithium tantalate material use, our approach reduces costs while leveraging
existing silicon photonics technology advancements, in particular supporting ultra-fast
monolithic germanium photodetectors and established process design kits.
FIG. 1. Next generation high-speed silicon photonic platform. (a) Generic view of a state-of-the-art Si-PIC platform with its
basic components. (b) View of the vision of a high-speed Si-PIC’s cross-section: next to the components already oered, lithium
tantalate modulators need to be integrated. (c) SEM (colourised) picture after the integration of LiTaO3 on a Si-PIC and FIB
cross-section (colourised) of a heterogeneous electro-optic device on a Si-PIC.
The exponential growth in data network trac leads
to hardly sustainable power consumption levels in data
centres.
Addressing this challenge, optical bres pro-
vide an ecient solution for data transport within and
between these data centres.
Photonic integration, ini-
tially developed on indium phosphide1 and more re-

<!-- page 2 -->
2
FIG. 2. Integration ow of LiTaO3 on a Si-PIC. (a) Architecture of the passive Si and SiN circuitry and corresponding optical
microscope picture. WG: waveguide, BOX: buried oxide, TR: transition, GC: grating coupler, MMI: multimode interferometer,
MZI: Mach-Zehnder interferometer. (b) Architecture of the hybrid modulator after the integration of the LiTaO3 and corre-
sponding optical microscope picture. (c) Architecture of the Mach-Zehnder modulator after the fabrication of the electrodes and
corresponding optical microscope picture. G: ground, S: signal. (d) Overview of a 7-mm-long heterogeneous Si and SiN/LiTaO3
MZM.
cently on silicon photonic platforms, has signicantly
enhanced data stream scaling while minimising power
consumption2.
Silicon photonic integrated circuits (Si-
PICs) enable the dense integration of complex function-
alities, leveraging CMOS-compatible fabrication for high-
volume, high-yield, and low-cost production. A typical
silicon photonic platform (see Fig. 1) integrates several
key components, including Si waveguides with dopants
for high-speed modulation, heaters for ecient low-speed
thermal tuning, a silicon nitride (SiN) layer for low-loss
propagation and high-quality ltering, and germanium
photodetectors.
The latter have demonstrated band-
widths exceeding 200 GHz3.
However, emerging stan-
dards such as 1600ZR+4 may require baud rates surpass-
ing 200 GBd, which exceed the performance limits of cur-
rent Si photonic modulators.
To overcome this limitation, various alternative plat-
forms are under investigation, with the heterogeneous
integration of novel electro-optic (EO) materials gain-
ing
signicant
attention.
Barium
titanate
(BTO)
presents a promising option due to its capability for
direct growth on Si layers and its compatibility with
CMOS fabrication processes5.
While high-speed oper-
ation has been achieved, challenges such as high per-
mittivity dispersion and the requirement for a con-
stant bias persist.
Alternative approaches include the
integration of organic materials6, plasmonics7–9, III-V
semiconductors10,11, and graphene12–16. However, thin-
lm lithium niobate (TFLN) has emerged as a standout
material, oering a combination of low optical loss and
a strong, fast EO coecient17. In demonstrations, band-
widths exceeding 100 GHz18 have been achieved. More-
over, lithium niobate (LiNbO3) has been heterogeneously
integrated onto Si and SiN platforms: techniques such as
wafer bonding19,20 and micro-transfer printing have been
successfully employed21.
Very recently, lithium tantalate (LiTaO3) has emerged
as an alternative to LiNbO322–25.
The material has
a similar EO coecient as LiNbO3.
Yet, it shows a
much weaker photorefractive eect and a higher dam-
age threshold. DC stable LiTaO3 EO modulators have
been demonstrated26. So far, only demonstrations on a
monolithic platform have been shown.
As with mono-
lithic LiNbO3 optical waveguide platforms, however, the
route towards high volume production in CMOS fabs is
less clear due to the contamination caused by lithium27.
Moreover, the integration of other components such as
high-speed detectors is not obvious at the moment.
In this work, we present the rst demonstration of
the heterogeneous integration of LiTaO3 onto a Si-PIC.
The integrated modulator is compatible with the Si-
PIC’s process design kit (PDK), ensuring full compati-
bility with platform components and enabling their op-
timal functionality.
Leveraging the existing PDK, we
achieve seamless integration of EO components without
compromising the performance of the established ones.
Using a back-end integration approach, based on micro-
transfer printing, we ensure compatibility with the en-
tire wafer stack, facilitating co-integration with critical
components such as heaters, lters, and germanium pho-
todetectors. A conceptual view is illustrated in Fig. 1.
Moreover, the printing technique allows for reversing the
poling axis such that complex electrode designs can be
realised, allowing for dierential driving, such as ground-
signal-signal-ground (GSSG), which is hardly doable on
monolithic LiNbO3 platforms or wafer bonded devices.
Hence, such modulators can be driven by dierential
drivers with improved linearity thanks to the cancella-
tion of even order harmonics28. Moreover, these dier-
ential drivers also have improved tolerance to electro-
magnetic interference (EMI) or power supply noise, es-
pecially important in multi-channel devices.
The un-
balanced Mach-Zehnder modulator (MZM), presented in
this work, achieves a measured bandwidth exceeding
70 GHz (limited by measurement equipment) and is es-
timated to reach 90 GHz.
It also exhibits a voltage-
length product of 2.3 V´cm, comparable to recent results
from monolithic LiTaO3 platforms24.
Performance can

<!-- page 3 -->
3
be further enhanced through the optimisation of the hy-
brid SiN/LiTaO3 cross-section. Furthermore, the hybrid
phase modulator in the arms of the Mach-Zhender has a
total insertion loss of 2.9 dB, dominated by metal losses
caused by a fabrication misalignment (1.6 dB). Finally,
data modulation up to 190 GBd is successfully demon-
strated highlighting the potential of this approach for
next-generation high-speed photonic applications.
RESULTS
Design and fabrication of the LiTaO3 MZM
The device architecture relies on an MZM implemented
on a cutting-edge Si-PIC platform (imec iSiPP200). A
subset of the commercial platform is used for device lay-
out and circuit routing. Here, only the silicon and sil-
icon nitride waveguide layers are processed.
Starting
from a silicon-on-insulator (SOI) wafer with a 220 nm de-
vice layer, the Si layer is patterned with two etch depths
(70 nm and 220 nm). Next, the wafer is planarised and
a SiN layer with a thickness of 300 nm is deposited.
This layer is patterned (with a full etch of 300 nm), and
cladded in oxide, after which the wafer is planarised. The
routing and splitting of the modulator is done in the sil-
icon waveguide layer. The 50:50 splitter and combiner
in the MZM are 1x2 multi-mode interferometers (MMIs),
which are available in the PDK. Light is coupled in and
out of the chip using grating couplers (GCs), and routing
is performed by single-mode waveguides, which are in-
cluded in the PDK. The schematic of the passive device
architecture and an optical microscope picture of the fab-
ricated circuit are depicted in Fig. 2a.
FIG. 3. Transition from the Si waveguide to the hybrid mod-
ulator.
(a) Schematic of the tri-layer adiabatic transition:
mode proles in the regions of interest are added.
(b) Ex-
pected transmission from a fully etched Si waveguide to a hy-
brid SiN/LiTaO3 phase modulator, (inset) and cross-section
of the passive stack of an MZM arm.
In a back-end heterogeneous integration step,
X-
cut thin-lm LiTaO3 is incorporated onto both arms
of the MZI, hence enabling hybrid phase modulators.
The integration of this EO material is achieved us-
ing micro-transfer printing technology29,30.The micro-
transfer printing process allows for the transfer of thin
lms with a large length-to-width ratio. Here, a commer-
cial LTOI wafer is used as the source for the transfer, on
which suspended LiTaO3 membranes are prepared. These
measure 300 nm in thickness, 30 µm in width, and 7 mm
in length. Details on the fabrication process ow of the
suspended membranes and the back-end integration are
provided in the Methods section.
These membranes are then transferred to the photonic
chip using a commercial micro-transfer printer, after ap-
plying a 50-nm-thick layer of BCB on the chip as an ad-
hesive. The high membrane density on the source wafer
allows for the fabrication of over 12000 membranes (7 mm
x 30 µm) on a 4-inch wafer, showcasing the ecient use
of the costly electro-optic material. A schematic of the
device and the corresponding optical microscope pictures
after the successful integration of the LiTaO3 on the plat-
form are illustrated in Fig. 2b. Utilising the versatility
of the micro-transfer printing approach, the modulator
operates in a push-pull conguration, with the printed
LiTaO3 membranes oriented 180° relative to one another.
This allows the use of integrated electronic drivers with
a dierential output28.
After integration, the nal metallisation step involves
depositing a 20 nm layer of titanium (Ti) followed by a
1 µm layer of gold (Au) using a lift-oprocess. This step
is required to form the metal electrodes congured in a
ground-signal-signal-ground (GSSG) arrangement on top
of the device arms. The schematic of the nalised MZM,
along with a detailed optical microscope view are shown
in Fig. 2c. The entire modulator is depicted in Fig. 2d.
A critical feature of the device is the ecient coupling
of light from the Si waveguide to the hybrid EO section of
the MZM. This coupling is achieved through an adiabatic
transition. In the design, both the Si and the SiN layers
of the Si photonic platform are utilised to minimise the
coupling loss into the EO structure. A schematic of the
coupling structure, along with the optical mode proles
for each region of interest, is shown in Fig. 3. With the
vertical tri-layer stack of Si/SiN/LiTaO3, a transition is
designed to be robust against fabrication geometry vari-
ations, e.g.
the layer thickness of SiN and LiTaO3 or
waveguide dimensions. The simulated transmission e-
ciency, depicted in Fig. 3.b, is computed using the eigen-
mode expansion (EME) method implemented in the com-
mercial software Ansys Lumerical MODE, for a range of
taper lengths, and shows that a coupling eciency of vir-
tually 100 % is possible in this hybrid structure. Further
details on the adiabatic transition design are provided
in Methods. Excluding the coupling section, the eective
length of the active EO SiN/LiTaO3 waveguide is reduced
to 6.6 mm, compared to the original 7 mm length of the
modulator arms, since a 200 um transition length was
chosen.
Performances of the modulator
The fabricated MZMs are characterised both optically
and electro-optically. Initial characterisation involves a
transmission wavelength sweep in the O-band, spanning
from 1300 nm to 1325 nm. Continuous-wave (CW) op-
tical power is coupled to the chip using a tunable laser.
Through an optical bre and a polarisation controller,
the transverse electric (TE) mode is exited in the waveg-
uide via a grating coupler. The optical power transmit-
ted through the MZM is recorded using a power meter,
while on the same structure set, a reference waveguide
transmission sweep is conducted to account for the con-
tributions of the grating couplers and Si routing.
The resulting normalised MZM transmission spectrum
is plotted in Fig. 5.a, where the interference pattern re-
veals a spectral interferometric pattern with a free spec-
tral range (FSR) of 4 nm and an extinction ratio (ER)
of 28 dB at a wavelength of 1310 nm. However, beyond
the 3.8 dB insertion loss of the MZM structure dened
by the platform PDK, the high-speed electro-optic mod-
ulation section introduces an additional 2.9 dB insertion
loss, including coupling (0.6 dB) and propagation losses
(0.7 dB). The remaining loss is attributed to a misalign-
ment of around 0.6 µm of the nal gold layer, which intro-
duces an additional 1.6 dB of optical loss. Further details
on the losses of the MZM are available in the Method sec-
tion.
Next, the device is operated in a quasi-DC modula-
tion experiment. The transmitted CW optical signal is
modulated at low electrical frequencies to characterise the
half-wave voltage (Vπ). The device is driven by a radio
frequency (RF) triangular wave at 100 kHz, generated
by an RF signal generator, while the optical wavelength
of 1309.26 nm is chosen to correspond to the quadrature

<!-- page 4 -->
4
FIG. 4. High-speed characterisation of the modulator. (a) Measurement setup (EE system) for the RF line frequency response
characterisation. (b) Results showing the comparison using regular Si substrate and high resistivity substrate. (c) Measurement
setup (EOE system) for the modulator frequency response characterisation. (d) Results showing the comparison for a modulator
integrated on a platform using a regular Si substrate and high resistivity substrate.
operating point of the modulator, enabling operation in
the linear modulation regime. The electrical signal is ap-
plied to the device using an RF probe and the optical
power transmitted through the device is recorded using a
photodetector and an oscilloscope.
This result is then
plotted as a function of the applied voltage.
The re-
sults, shown in Fig. 5.b, reveal a Vπ of 7.0 V for a single
phase shifter (one arm of the MZM), which corresponds
to 3.5 V in the push-pull amplitude modulation cong-
uration (both arms of the MZM). This yields a voltage-
length product (VπL) of 2.3 V´cm, consistent with recent
reports on LiTaO3 MZMs fabricated on the LiTaO3-on-
insulator (LTOI) platform24,26. Moreover, when compar-
ing this result to the expected Vπ curve, calculated using
nite element method (FEM) simulations in combination
with an analytical model, it demonstrates excellent agree-
ment with the measurement as depicted in the overlay in
Fig. 5.b. More information on the simulation of Vπ is
available in Methods.
The high-speed response of the device is characterised
in twofold:
rst, the electrodes are characterised by
performing an electrical-to-electrical (EE) measurement
and afterwards, the full electrical-to-optical-to-electrical
(EOE) response is measured. The EE responses are mea-
sured with a vector network analyser (VNA) connected to
the chip with a set of GSSG probes, as shown in Fig. 4.a.
This measurement is carried out on two dierent chips.
On the rst chip, the RF lines are placed to drive an MZM
FIG. 5.
Quasi-DC characterisation of the modulator.
(a)
Measurement of the modulator transmission as a function of
wavelength (b) Normalised transmission of the modulator as a
function of applied voltage as well as the simulated response,
and (inset) cross-section of the full stack of an MZM arm.
which is fabricated on a wafer with a regular Si substrate.
In the second case, a high-resistivity Si substrate is used
for the wafer.
The comparison between the two lines,
presented in Fig. 4.b, demonstrates a clear enhancement
of the RF bandwidth when using the high-resistivity Si
substrate due to lower absorption loss of the RF wave in
the substrate.
The EOE response of the MZMs, fabricated on a stan-
dard and a high resistivity Si substrate, are then mea-
sured using a similar experimental setup (Fig. 4.c). The
device is connected to one port of the VNA using a GS
probe on one side, and the opposite side of the device is
terminated with a 50 Ωload, to prevent electrical reec-
tions at the far side. CW optical power is coupled from
the laser to the Si-PIC with an optical bre, transmit-
ted through the MZM, out-coupled and converted into
an electrical signal by a high-speed photodetector. The
resulting signal is received at the second port of the VNA.
The EOE frequency response of the MZMs are presented
in Fig. 4.d for both modulator types.
Again, the de-
vice with a high resistivity substrate demonstrates a much
higher 3-dB cut-ofrequency in excess of 70 GHz, limited
by the experimental setup (the bandwidth of the VNA
and the photodetector).
Data transmission
As a demonstration, the device is utilised in a link to
transmit data for which the setup is presented in Fig. 6.a.
A dierential signal, generated by a 256 GS/s arbitrary
waveform generator (AWG), is applied to the device us-
ing a GSSG probe. A second probe is employed to termi-
nate the device on the opposite side of the transmission
line with two 50 Ωloads to prevent electrical reections.
The optical signal is amplied using a praseodymium-
doped bre amplier (PDFA) before being detected by a
high-speed photodetector. The detected signal is subse-
quently recorded using a sampling oscilloscope. To op-
timise the transmission, the link is characterised to fa-
cilitate the equalisation of the signal generated by the
AWG. As such, clear eye diagrams are generated. Sup-
plementary Data Fig. 3 shows the measured frequency
response of the data transmission link (magnitude in a.
and phase in b.), which is compensated at the AWG. A
noticeable performance drop-ois observed in the link
characterisation around 70 GHz, attributed to the band-
width limitations of the electrical cables, RF probes, the
high-speed photodetector and the AWG. All transmitted
signals are based on standardised pseudorandom bit se-
quences (PRBS-15).
A 64-tab feed-forward equalisation (FFE) algorithm is
applied at the receiver end, using the sampling oscillo-
scope’s software to enhance the received signal. Fig. 6.b
presents the measured bit error rate (BER) derived from

<!-- page 5 -->
5
FIG. 6. Data transmission experiment. (a) Measurement setup for eye-diagram measurements. (b) Bit error rate (BER) in the
transmitted bit sequence for NRZ (blue) and PAM4 (red) measured signals. A horizontal dashed line indicates the threshold
for forward error correction (FEC) with an overhead of 15 %. Eye diagrams for (c) 112 GBaud NRZ, (d) 180 GBaud NRZ, (e)
112 GBaud PAM4, (f) 140 GBaud PAM4, all gures used correction at the transmitter side and a 64 tap FFE on the receiver
side, however, no oine digital post-processing was done
the generated eye diagrams using both non-return-to-zero
(NRZ) and pulse-amplitude modulation 4-level (PAM4)
formats. A dashed line indicates the BER threshold for a
forward error correction (FEC) algorithm with an over-
head of 15 %. The lower limit of the plot, at 3x10−4,
indicates the error rate threshold below which statisti-
cally signicant assessments could not be made, due to
the limited number of bits transmitted during the demon-
stration. Fig. 6 also includes representative eye diagrams,
all of which incorporate FFE.
DISCUSSION
Performance improvement
High-speed hybrid EO/Si-PIC modulators are designed
for integration into data communication systems, encom-
passing both inter- and intra-data centre links to enable
massive data transmission. In such systems, power con-
sumption is a critical parameter. The presented devices
demonstrate signicant potential for further optimisation
in this regard, particularly through the reduction of elec-
trical driving power.
Recently, high-quality etching of LiTaO3 has been
demonstrated24. By employing this etching process, the
LiTaO3 layer in the hybrid modulator can be patterned.
The patterned LiTaO3 layers would allow for better con-
nement and hence a stronger EO eect as the electrode
separation can be reduced (see Methods). Consequently,
the gap can be reduced from 5.5 µm to 3.4 µm, result-
ing in a stronger RF eld and a lower simulated Vπ of
2.7 V, corresponding to a simulated voltage-length prod-
uct of 1.8 V´cm.
As discussed in the Method section,
for this conguration ecient adiabatic couplers can be
made on the platform that allows for high coupling e-
ciency of more than 99 %. Moreover, proper design allows
similar very high coupling eciencies (> 99 %) for trans-
verse misalignments of the LiTaO3 ridge waveguide of up
to 0.5 µm. These misalignments remain within the align-
ment accuracy specications of wafer-scale micro-transfer
printing tools.
CONCLUSIONS
The successful heterogeneous integration of lithium tan-
talate onto a silicon photonic platform with minimal in-
sertion loss is demonstrated. Implemented on a standard
200-mm Si-PIC platform, our approach ensures seam-
less integration with existing platform PDK components
while preserving their performance. By leveraging back-
end integration, we maintain compatibility across the
entire wafer stack, enabling ecient co-integration with
critical components such as waveguides, heaters, lters,
and germanium photodetectors. The manufactured MZM
achieves a measured bandwidth exceeding 70 GHz, lim-
ited by the measurement setup, and exhibits a voltage-
length product of 2.3 V´cm comparable with demon-
strated LTOI platforms. The hybrid phase modulators
in the MZM arms induce an additional 2.9 dB insertion
loss. Lastly, high-speed transmission experiments are per-
formed, reaching symbol rates of 190 Gbaud (NRZ) and
data rates of more than 320 Gbit/s (PAM4).
These results mark a signicant step toward high-
speed, high-performance hybrid lithium tantalate pho-
tonic platforms. They pave the way for scalable, industry-
compatible electro-optic integration for low-energy con-
sumption data communication links that can be fabri-
cated in high volumes.
Data Availability
The datasets generated during and/or analysed during
the current study are available from the corresponding
author on reasonable request.
Acknowledgments
The authors would like to acknowledge the contribution
of imec’s 200 mm pilot line for silicon photonics wafer
fabrication and imec’s PDK team for the mask tape-out.
The authors also would like to acknowledge the contri-
bution from Priya Eswaran and Semih Culhaoglu for the
help in Si-PIC process development, Steven Verstuyft,
Peter Geerinck, Elif ¨Oz¸ceri, Liesbet Van Landschoot for
the help during the lithium tantalate device fabrication,
and Clemens Kr¨uckel and Joris Van Kerrebrouck for
measurement support.
Author contributions
S.B., P.A., J.C., S.J., P.V., N.S.2, D.B., M.D. and F.F.
developed and fabricated the silicon nitride,
silicon
platform on 200 mm. M.N., T.Z. and M.B. fabricated
the hybrid LiTaO3/SiN devices and realised the het-
erogeneous integration.
E.V., A.M., P.N. and O.C.

<!-- page 6 -->
6
performed numerical simulations. E.V., A.M. and T.V.
designed the devices.
M.N. and T.V. performed the
characterisation (optical and quasi-dc).
M.N., T.V.,
N.S2,3, J.D., C.B. and S.N. performed the high-speed
characterisation and data communication experiment.
M.N., T.V., B.K. and M.B. prepared the gures, data,
and the manuscript with input from other authors. P.A.,
S.L., X.Y., G.R., B.K. and M.B. supervised the project:
Competing interests
The authors declare no conict of interest.
Funding
We want to thank the European Space Agency for fund-
ing under the E/0365-70 - NAVISP, LEO Project and
The Research Foundation Flanders (FWO) for project
3G035722 and the FWO and F.R.S.-FNRS under the Ex-
cellence of Science (EOS) program (40007560).
REFERENCES
1 Sneh, A. & Doerr, C. R. Indium Phosphide-Based Photonic
Circuits and Components. In Integrated Optical Circuits
and Components (CRC Press, 1999). Num Pages: 77.
2 Zhou, X., Yi, D., Chan, D. W. U. & Tsang, H. K. Silicon
photonics for high-speed communications and photonic sig-
nal processing. npj Nanophotonics 1, 1–14 (2024). Pub-
lisher: Nature Publishing Group.
3 Lischke, S. et al. Ultra-fast germanium photodiode with 3-
dB bandwidth of 265 GHz. Nature Photonics 15, 925–931
(2021). Publisher: Nature Publishing Group.
4 Gass, K. Oif adds a short-reach design to its 1600zr/ zr+
portfolio.
5 Chelladurai, D. et al. Barium Titanate and Lithium Nio-
bate Permittivity and Pockels Coecients from MHz to
Sub-THz Frequencies (2024). ArXiv:2407.03443 [physics].
6 Wolf, S. et al.
Silicon-Organic Hybrid (SOH) Mach-
Zehnder Modulators for 100 Gbit/s on-oKeying. Scien-
tic Reports 8, 2598 (2018). Publisher: Nature Publishing
Group.
7 Haner, C. et al.
All-plasmonic Mach–Zehnder modula-
tor enabling optical high-speed communication at the mi-
croscale. Nature Photonics 9, 525–528 (2015). Publisher:
Nature Publishing Group.
8 Burla, M. et al. 500 GHz plasmonic Mach-Zehnder modula-
tor enabling sub-THz microwave photonics. APL Photonics
4, 056106 (2019).
9 Baeuerle, B. et al. 120 GBd plasmonic Mach-Zehnder mod-
ulator with a novel dierential electrode design operated at
a peak-to-peak drive voltage of 178 mV. Optics Express 27,
16823–16832 (2019). Publisher: Optica Publishing Group.
10 Yamazaki, H. et al. IMDD Transmission at Net Data Rate
of 333 Gb/s Using Over-100-GHz-Bandwidth Analog Mul-
tiplexer and Mach–Zehnder Modulator. Journal of Light-
wave Technology 37, 1772–1778 (2019). Conference Name:
Journal of Lightwave Technology.
11 Estar´an, J. M. et al. 140/180/204-Gbaud OOK Transceiver
for Inter- and Intra-Data Center Connectivity.
Journal
of Lightwave Technology 37, 178–187 (2019). Conference
Name: Journal of Lightwave Technology.
12 Romagnoli, M. et al. Graphene-based integrated photonics
for next-generation datacom and telecom. Nature Reviews
Materials 3, 392–414 (2018). Publisher: Nature Publishing
Group.
13 Liu, M. et al. A graphene-based broadband optical modu-
lator. Nature 474, 64–67 (2011). Publisher: Nature Pub-
lishing Group.
14 Phare, C. T., Daniel Lee, Y.-H., Cardenas, J. & Lipson, M.
Graphene electro-optic modulator with 30 GHz bandwidth.
Nature Photonics 9, 511–514 (2015).
Publisher: Nature
Publishing Group.
15 Wu, C. et al. Graphene-Based Silicon Photonic Electro-
Absorption Modulators and Phase Modulators.
IEEE
Journal of Selected Topics in Quantum Electronics 30, 1–
11 (2024).
Conference Name: IEEE Journal of Selected
Topics in Quantum Electronics.
16 Eppenberger,
M. et al.
Resonant plasmonic micro-
racetrack modulators with high bandwidth and high tem-
perature tolerance. Nature Photonics 17, 360–367 (2023).
Publisher: Nature Publishing Group.
17 Wang, C. et al.
Integrated lithium niobate electro-optic
modulators operating at CMOS-compatible voltages. Na-
ture 562, 101–104 (2018). Publisher: Nature Publishing
Group.
18 Zhu, D. et al. Integrated photonics on thin-lm lithium
niobate. Advances in Optics and Photonics 13, 242–352
(2021). Publisher: Optica Publishing Group.
19 Churaev, M. et al. A heterogeneously integrated lithium
niobate-on-silicon nitride photonic platform. Nature Com-
munications 14, 3499 (2023). Publisher: Nature Publish-
ing Group.
20 Ruan, Z. et al. High-Performance Electro-Optic Modula-
tor on Silicon Nitride Platform with Heterogeneous Inte-
gration of Lithium Niobate. Laser & Photonics Reviews
17, 2200327 (2023).
21 Vanackere, T. et al. Heterogeneous integration of a high-
speed lithium niobate modulator on silicon nitride using
micro-transfer printing. APL Photonics 8, 086102 (2023).
22 Wang, C. et al.
Ultrabroadband thin-lm lithium tan-
talate modulator for high-speed communications (2024).
ArXiv:2407.16324 [physics].
23 Wang, H. et al. Optical switch with an ultralow DC drift
based on thin-lm lithium tantalate.
Optics Letters 49,
5019–5022 (2024). Publisher: Optica Publishing Group.
24 Wang, C. et al.
Lithium tantalate photonic integrated
circuits for volume manufacturing. Nature 629, 784–790
(2024). Publisher: Nature Publishing Group.
25 Zhang, J. et al. Ultrabroadband integrated electro-optic
frequency comb in lithium tantalate. Nature 1–8 (2025).
Publisher: Nature Publishing Group.
26 Powell, K. et al. DC-stable electro-optic modulators us-
ing thin-lm lithium tantalate. Optics Express 32, 44115–
44122 (2024). Publisher: Optica Publishing Group.
27 Wandesleben, A. F. et al. Inuences and Diusion Eects
of Lithium Contamination during the Thermal Oxidation
Process of Silicon.
Advanced Engineering Materials 26,
2400396 (2024).
28 Razavi, B.
Design of Analog CMOS Integrated Circuits
(McGraw-Hill, 2001).
29 Roelkens, G. et al. Present and future of micro-transfer
printing for heterogeneous photonic integrated circuits.
APL Photonics 9, 010901 (2024).
30 Roelkens, G. et al. Micro-Transfer Printing for Heteroge-
neous Si Photonic Integrated Circuits. IEEE Journal of
Selected Topics in Quantum Electronics 29, 1–14 (2023).
Conference Name:
IEEE Journal of Selected Topics in
Quantum Electronics.

<!-- page 7 -->
7
SUPPLEMENTARY MATERIAL
Process ow for heterogeneous LiTaO3 integration
The fabrication of suspended LiTaO3 membranes starts with a 300 nm LiTaO3, on 2 µm oxide, on a Si substrate
wafer (Supplementary Data Fig. 1.a). The LiTaO3 is patterned into rectangles of 30 µm x 7 mm with UV-lithography,
using an amorphous silicon (aSi) hard mask suitable for an argon (Ar) based reactive ion etching (RIE). The aSi hard
mask is then removed in a potassium hydroxide (KOH) solution (Supplementary Data Fig. 1.b). As a second step,
the oxide layer is patterned with UV-lithography and RIE, after which the oxide is etched down to the Si substrate
(Supplementary Data Fig. 1.c). A mechanical encapsulation layer is then added, consisting of a photoresist layer
exposed with UV-lithography which is subsequently developed. The patterns in the photoresist encapsulation are
made in order to dene mechanical links from the Si substrate to the LiTaO3 membranes, allowing for the suspension
of LiTaO3 when the devices are released (Supplementary Data Fig. 1.d). The last step is the releasing of the LiTaO3
membranes in a wet etchant: buered hydrouoric (BHF) acid (Supplementary Data Fig. 1.e). The picture from
Supplementary Data Fig. 1.f shows an overview of the nal sample. In this picture, 21 membranes of 30 µm x 7 mm
are located on a sample of 2 mm x 7 mm. The membrane density is 1.5 membranes/mm2, and for a 4-inch wafer, the
total number of membranes would be more than 12 000.
Supplementary Data Fig. 1. Description of the successive steps for the fabrication and printing of suspended LiTaO3 membranes.
(a) LiTaO3 (300 nm) / oxide (2 µm) / Si wafer (substrate) starting point. (b) Patterning of the LiTaO3 membranes. (c)
Patterning of the oxide release layer. (d) Photoresist mechanical encapsulation of the structures. (e) Undercutting (releasing)
of the oxide layer, making the structures suspended. (f) Overview of the nal sample with suspended LiTaO3. (g) picking of
the suspended LiTaO3. (h) Printing of the LiTaO3 on a pre-processed external chip. (i) Encapsulation removal. (j) Overview
of a few successfully printed LiTaO3 membranes.
At this step, the membranes are ready to be integrated on a dierent substrate. They are picked using a polymer
PDMS stamp driven by a commercial micro-transfer printer tool (Supplementary Data Fig. 1.g). For better adhesion,
a BCB buer layer is applied to the target before printing. The membranes are then placed on the host substrate,
and the stamp is retracted, leaving the membrane at its nal location (Supplementary Data Fig. 1.h). The photoresist
encapsulation is subsequently removed using an RIE oxygen plasma (Supplementary Data Fig. 1.i), after which the
BCB layer is cured at 280°C for 90 min. An example of a Si chip with a cascade of 1-mm-long LiTaO3 sections used
as an insertion loss test structure is depicted in Supplementary Data Fig. 1.j.
On top of the loss characterisation test structures, the integration of LiTaO3 membranes is done for four MZM
devices. To illustrate the stability of the process ow, the transmission as a function of the wavelength of the devices is
recorded and presented in Supplementary Data Fig. 2. The four devices, combining eight LiTaO3 integrated membranes
in total, show similar results with an ER > 25 dB for the peak near 1310 nm. The wavelength sweeps in Supplementary
Data Fig. 2 are normalised by subtracting the grating couplers and routing outside the actual MZM.
Loss characterisation
Two kinds of optical loss contributions can be distinguished: propagation loss through the hybrid SiN/LiTaO3 waveg-
uide and the coupling or transition loss to get light from a Si waveguide to a hybrid SiN/LiTaO3 waveguide (see
adiabatic coupling for more details on this transition). To determine the propagation loss, hybrid SiN/LiTaO3 waveg-
uides with dierent lengths are fabricated. These are indicated as S1 and S2 in Supplementary Data Table. I and
have a propagation length of respectively 0.66 and 6.66 mm. For the transitions, two structures are prepared where

<!-- page 8 -->
8
Supplementary Data Fig. 2. Transmission (T) measurements for four fabricated MZMs. The losses from the grating couplers
and external routing are removed. The insertion loss contribution from the Si-PIC components (C1 and C2 from Supplementary
Data Table. II) and the contribution of the back-end integration process (C3, C4 and C5 from Supplementary Data Table. II)
are described on top of the graphs.
one has only two transitions (one in, one out) and the other one has 28 transitions. Those are indicated as S3 and S4
in Supplementary Data Table. I and have, in total, the same propagation length. Comparing the transmission of all
structures, and taking the propagation loss in the Si waveguides into account, a propagation loss of 1.0 ± 0.5 dB/cm
and a transition loss of 0.3 ± 0.1 dB per transition can be deduced.
Test
structure
Amount of
membranes (#)
Length of propagation
in SiN/LiTaO3 (mm)
Amount of
transitions (#)
Si propagation
loss (dB)
Transmission at
1310 nm (dB)
S1
1
0.6
2
0.2 ± 0.01
-8.2 ± 0.2
S2
1
6.6
2
0.2 ± 0.01
-8.8 ± 0.2
S3
1
0.3
2
0.4 ± 0.01
-8.1 ± 0.2
S4
1
0.3
28
2.5 ± 0.04
-16.8 ± 0.2
Supplementary Data Table. I. Description and measured transmission for all test structures needed to determine the propagation
loss and transition loss of the LiTaO3 slabs.
Label Component
Loss in the MZM
C1
MMI splitters
0.8 ± 0.2
C2
Si routing inside MZM
3.0 ± 0.4
C3
Transitions (Si to SiN/LiTaO3) 0.6 ± 0.2
C4
EO section (SiN/LiTaO3)
0.7 ± 0.3
C5
Metal misalignment
1.6 ± 0.1
Supplementary Data Table. II. Measurement of the loss for each component of the MZM allowing to extract the contribution
to the insertion loss from the Si-PIC PDK (C1 and C2) and the back-end integration process (C3, C4 and C5).
All component losses inside the MZM are also listed in Supplementary Data Table. II. The losses of the MMI splitter
(C1) and Si routing (C2) are measured from the process control structures on the same chip. The transition from
Si to SiN/LiTaO3 (C3) and the propagation through the hybrid SiN/LiTaO3 (C4) are extracted from the loss test
structures as described above. Also, the introduction of metal traces on top of the LiTaO3 (C5) introduces extra loss.
A misalignment of about 600 nm is observed and can be seen in the FIB cut. As no test structures are available to
quantify this loss, a nite element method simulation was carried out in COMSOL to identify its contribution to be
1.6 dB.
Adiabatic coupling from Si platform to the hybrid LiTaO3 modulator
An adiabatic tapering structure is optimised to eciently couple the light from the Si waveguide to the EO section.
The following paragraph claries the design of the taper section referred to when describing the fabricated chips (hybrid
SiN/LiTaO3-on-Si modulator).
At the start of the transition, light remains conned within the Si waveguide despite the introduction of the LiTaO3
layer, due to the signicant refractive index contrast between the two materials as shown in Fig. 3.a. At this stage,
the relevant material stack consists of a 380 nm x 220 nm Si waveguide, encapsulated by a 500 nm thick oxide layer
and on top of that a 30 µm x 300 nm LiTaO3 layer.
Further in the transition structure, a 300 nm SiN layer is
introduced, positioned 150 nm above the Si and 70 nm below the LiTaO3 (see (Fig. 3.a.(i)). At this point, the SiN
waveguide is tapered from a narrow tip (150 nm), ensuring that the optical mode remains predominantly in the Si, with
minimal inuence from that SiN taper tip. Subsequently, the SiN waveguide undergoes an adiabatic width expansion,
after which the Si waveguide tapers down to 150 nm, facilitating a gradual transition of the optical mode from Si
to the hybrid SiN/LiTaO3 conguration (Fig. 3.a.(ii)). The SiN width is then adjusted to its nal value of 900 nm
to achieve the optimal connement of the hybrid mode (Fig. 3.a.(iii)), determined as a trade-obetween minimising
Vπ by having as much as possible of the light in the LiTaO3 layer and mitigating subsequent metal-induced losses by
achieving sucient connement. In this conguration, the lateral alignment of the LiTaO3 with the Si/SiN waveguides
does not signicantly aect the eciency of the transition. The taper dimensions are optimised using an EME solver
in Lumerical, resulting in a nal taper length of 200 µm.
Half wave voltage calculation
The Vπ simulations are done using a combination of multiple FEM simulations in COMSOL and an analytical model.
The EO phase shift (∆ϕ) experienced by an MZM in a push-pull conguration is governed by the following expression:
∆ϕ = 2πn4
eΓr33V
neffλG
L
(1)

<!-- page 9 -->
9
where ne = 2.1269 is the extraordinary refractive index, Γ represents the overlap factor between the RF and optical
modes, r33 = 30.5 pm/V is the EO coecient of LiTaO3, V is the applied voltage, L = 6.6 mm is the eective length
of one modulator arm, λ = 1310 nm is the optical wavelength, neff is the eective refractive index of the optical
mode, and G denotes the electrode gap. The simulated parameters (Γ, neff and G) are listed in Supplementary Data
Table. III.
Parameter Hybrid EO waveguide Unit
Γ
0.454
-
n
1.852
-
G
5.5
µm
Supplementary Data Table. III. Parameters for a hybrid EO MZM.
Using the equation and the simulation results, the theoretical values of Vπ for an EO-section with 6.6 mm length
are calculated to be 3.6 V for the slab push-pull conguration, which shows excellent agreement with measurements.
Supplementary Data Fig. 3. Data transmission link characterisation. (a) Amplitude and (b) phase characteristic of the data
transmission link, used to generate eye diagrams.

