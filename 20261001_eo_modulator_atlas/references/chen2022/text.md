---
paper_id: chen2022
source_url: https://doi.org/10.1063/5.0077232
doi: 10.1063/5.0077232
license: CC-BY-4.0
sha256: 631aa2fdbf6f96c0f3107fd5f4bd1e287ac9254a3c93839b7a5469c02c99f967
pages: 9
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
APL Photonics 7, 026103 (2022); https://doi.org/10.1063/5.0077232
7, 026103
© 2022 Author(s).
High performance thin-film lithium niobate
modulator on a silicon substrate using
periodic capacitively loaded traveling-wave
electrode 
Cite as: APL Photonics 7, 026103 (2022); https://doi.org/10.1063/5.0077232
Submitted: 31 October 2021 • Accepted: 16 January 2022 • Published Online: 02 February 2022
Gengxin Chen, Kaixuan Chen, Ranfeng Gan, et al.
COLLECTIONS
 This paper was selected as Featured
ARTICLES YOU MAY BE INTERESTED IN
High-efficiency lithium niobate modulator for K band operation
APL Photonics 5, 091302 (2020); https://doi.org/10.1063/5.0020040
Scaling up silicon photonic-based accelerators: Challenges and opportunities
APL Photonics 7, 020902 (2022); https://doi.org/10.1063/5.0070992
High-performance and compact integrated photonics platform based on silicon rich
nitride–lithium niobate on insulator
APL Photonics 6, 116102 (2021); https://doi.org/10.1063/5.0065437

<!-- page 2 -->
APL Photonics
ARTICLE
scitation.org/journal/app
High performance thin-film lithium niobate
modulator on a silicon substrate using periodic
capacitively loaded traveling-wave electrode
Cite as: APL Photon. 7, 026103 (2022); doi: 10.1063/5.0077232
Submitted: 31 October 2021 • Accepted: 16 January 2022 •
Published Online: 2 February 2022
Gengxin Chen,1 Kaixuan Chen,2,3 Ranfeng Gan,2 Ziliang Ruan,1 Zong Wang,2 Pucheng Huang,2 Chao Lu,4
Alan Pak Tao Lau,5 Daoxin Dai,1
Changjian Guo,2,3,4,a) and Liu Liu1,a)
AFFILIATIONS
1 State Key Laboratory for Modern Optical Instrumentation, Centre for Optical and Electromagnetic Research,
International Research Center for Advanced Photonics, East Building No. 5, Zijingang Campus, Zhejiang University,
Hangzhou 310058, China
2Guangdong Provincial Key Laboratory of Optical Information Materials and Technology,
South China Academy of Advanced Optoelectronics, Sci. Bldg. No. 5, South China Normal University,
Higher-Education Mega-Center, Guangzhou 510006, China
3National Center for International Research on Green Optoelectronics, South China Normal University,
Guangzhou 510006, China
4Photonics Research Center, Department of Electronic and Information Engineering, The Hong Kong Polytechnic University,
Hong Kong, China
5Photonics Research Center, Department of Electrical Engineering, The Hong Kong Polytechnic University, Hong Kong, China
a)Authors to whom correspondence should be addressed: changjian.guo@coer-scnu.org and liuliuopt@zju.edu.cn
ABSTRACT
Thin-film lithium niobate (TFLN) based traveling-wave modulators maintain simultaneously excellent performances, including large mod-
ulation bandwidth, high extinction ratio, low optical loss, and high modulation efficiency. Nevertheless, there still exists a balance between
the driving voltage and modulation bandwidth. Here, we demonstrate an ultra-large bandwidth electro-optic modulator without compromis-
ing the driving voltage based on the TFLN platform on a silicon substrate, using a periodic capacitively loaded traveling-wave electrode. In
order to compensate the slow-wave effect, an undercut etching technique for the silicon substrate is introduced to decrease the microwave
refractive index. Our demonstrated devices represent both low optical and low microwave losses, which leads to a negligible optical inser-
tion loss of 0.2 dB and a large electro-optic bandwidth with a roll-off of 1.4 dB at 67 GHz for a 10 mm-long device. A low half-wave
voltage of 2.2 V is also achieved. Data rates up to 112 Gb s−1 with PAM-4 modulation are demonstrated. The compatibility of the pro-
posed modulator to silicon photonics facilitates its integration with matured silicon photonic components using, e.g., hybrid integration
technologies.
© 2022 Author(s). All article content, except where otherwise noted, is licensed under a Creative Commons Attribution (CC BY) license
(http://creativecommons.org/licenses/by/4.0/). https://doi.org/10.1063/5.0077232
I. INTRODUCTION
Optical communication systems are now widely used all
around the world to cope with the ever-increasing need for data
transmission. The transmission rate of the data signal highly relies
upon the electrical-to-optical data conversion. Optical modula-
tors, when integrated with laser sources, can translate signals from
the electrical domain to the optical domain at a high speed,
which become essential components in long-haul telecommunica-
tion networks,1 optical quantum memories,2 radio-frequency (RF)
photonics,3 and data-center communications.4 One of the most
simple and cost-effective approaches is to internally modulate the
driving current of a semiconductor laser.5 However, this sort of
direct modulation is intrinsically limited in both modulation band-
width and extinction ratio and also suffers from modulation chirp.
On the other hand, external electro-optic (EO) modulators can
APL Photon. 7, 026103 (2022); doi: 10.1063/5.0077232
7, 026103-1
© Author(s) 2022

<!-- page 3 -->
APL Photonics
ARTICLE
scitation.org/journal/app
present better modulation performances and have been developed
in various material platforms. Pure silicon-based modulators,6 due
to the absence of the Pockels effect, rely on the carrier injection or
depletion in p–n junctions that are integrated into the optical wave-
guide. This leads to an inherent trade-off between the modulation
efficiency and optical loss. The integration of silicon or plasmonic
material with organic EO polymers, called silicon–organic hybrid7,8
or plasmonic–organic hybrid modulators,9,10 can benefit from ultra-
high EO coefficients of some engineered polymers, but their stability
still needs improvement. They often suffer from high losses as well.
Indium phosphate (InP) based modulators can achieve high digital
data rates,11 but their intrinsic modulation nonlinearity, as well as
high cost, limits their applications to a wide range.
Lithium niobate (LN) EO modulators, due to their outstand-
ing material properties, including good temperature stability, wide
transparency window, and highly efficient linear Pockels effect, have
been widely used in long-haul optical communications.12 Tradi-
tional bulk LN EO modulators, by means of proton exchange or
titanium diffusion,13 are limited in performances due to the fact
that the waveguide core is weakly guided with a low refractive index
contrast (<0.2). Its optical mode is large, which limits the minimal
spacing between the modulation electrodes and, hence, the mini-
mal driving voltage.14 Nevertheless, the invention of ultra-low loss
thin-film lithium niobate (TFLN) waveguide platforms can over-
come this fundamental limitation, which provides a high index
contrast using low refractive index materials, e.g., air or silicon diox-
ide, as the cladding layer. EO modulators based on etched ridge
waveguides on the TFLN platform exhibit excellent performances
of small footprint, low half-wave voltage (Vπ), and high band-
width beyond 100 GHz, as compared to traditional bulk LN EO
modulators.15,16
EO modulators simultaneously having a low Vπ and a large
EO bandwidth are important for applications spanning from RF
analog links to digital optical communication networks. However,
there is an intrinsic trade-off between Vπ and the EO bandwidth.
The greater the Vπ is, the shorter the length of the modulation
section L should be and hence the larger the EO bandwidth can
be. The challenge to maintain a large EO bandwidth with a low
driving voltage is how to simultaneously retain properties, includ-
ing low optical losses, low RF losses, perfect impedance matching,
and perfect velocity matching between the optical and RF waves.
The key modulation efficiency metric, called the half-wave voltage
length product (VπL), is mainly limited by the electrode gap. The
smaller the electrode gap is, the lower the VπL will be. On the other
hand, a small electrode gap will introduce additional optical losses
due to metal absorption and decreases EO bandwidths as well under
perfect impedance and velocity matching due to the increased RF
losses. In order to break this compromise, a periodic capacitively
loaded traveling-wave (CLTW) electrode structure has been pro-
posed on the TFLN platform.17 The basic idea here is to use a very
wide signal electrode to largely decrease the RF losses (which leads to
a large bandwidth). In the meantime, T-shaped periodic structures
are added in between the ground and signal electrodes to maintain a
small electrode gap (and hence a low VπL) without largely increas-
ing the distributed capacitance (keeping the impedance matching).
However, this structure would largely decrease the phase speed of
the RF signal (i.e., the slow-wave effect).17 In order to maintain
the index matching, a low permittivity substrate, e.g., quartz, was
employed.
In this paper, we demonstrate TFLN based EO modulators
using a CLTW electrode on a silicon substrate. This device allows a
small electrode gap to achieve a low VπL (∼2.20 V cm) value without
FIG. 1. Proposed EO modulator on a
silicon substrate using the CLTW elec-
trode. (a) 3D view of the whole struc-
ture. (b) 3D view and (c) 2D view of
the cross-sectional structure of the mod-
ulation section. (d) Top view of the MZI
structure.
APL Photon. 7, 026103 (2022); doi: 10.1063/5.0077232
7, 026103-2
© Author(s) 2022

<!-- page 4 -->
APL Photonics
ARTICLE
scitation.org/journal/app
additional optical absorption losses. Meanwhile, the silicon under-
cut etching technique is introduced to achieve velocity matching
between the optical and RF waves, which also helps to reduce RF
absorption losses from the semiconducting substrate.17 Compared
with EO modulators on the TFLN platform with regular traveling-
wave electrodes on a silicon substrate,15 the proposed EO modulator
shows excellent high-frequency performance with only a 1.4 dB roll-
off at 67 GHz. In addition, the proposed devices show an ultra-low
optical insertion loss of 0.2 dB and a large static extinction ratio of
>20 dB. On–off keying (OOK) modulation up to 100 Gbit s−1 and
PAM-4 modulation up to 112 Gbit s−1 are successfully achieved with
a dynamic extinction ratio of >9.7 dB.
FIG. 2. (a) Simulated TE optical mode profile and static electrical field distribution. VπL = 2.10 V cm can be extracted here. (b) Simulated RF mode profile at a RF frequency
of 100 GHz. Simulated frequency-dependent results of (c) characteristic impedance Zc, (d) RF effective index nm, and (e) RF loss αm. The two red dashed lines in (c) and
(d) correspond to the source/load impedance 50 Ω and the optical group index ng, respectively. Effects of undercut etching and h on (f) nm and (g) Zc at 70 GHz. The
optimal point is marked as a star.
APL Photon. 7, 026103 (2022); doi: 10.1063/5.0077232
7, 026103-3
© Author(s) 2022

<!-- page 5 -->
APL Photonics
ARTICLE
scitation.org/journal/app
II. DEVICE DESIGN AND FABRICATION
Figure 1(a) shows a three-dimensional (3D) view of the pro-
posed TFLN based EO modulators using a CLTW electrode on
an undercut etched silicon substrate. The device is fabricated on
a commercial x-cut LN-on-insulator (LNOI) wafer with a 3 μm
thick buried oxide (BOX) layer and a 400 nm thick TFLN layer.
The waveguide at the modulation section has a ridge height of
200 nm, i.e., half of the TFLN total thickness, and a width of 1.5 μm.
Grating couplers (GCs) are used as fiber-to-chip interfaces.18 Multi-
mode interferometers made also on TFLN are used as 3 dB couplers
for the Mach–Zehnder interferometer (MZI) structure. The CLTW
electrode based on a classic coplanar-line structure with some peri-
odically loaded T-segment structures is adopted in the proposed
device, in which optical and RF waves propagate collinearly. In the
regular TW electrode design, the high characteristic RF loss α0 of
0.75 dB cm−1 GHz−1/217 is the dominant source of limiting the EO
bandwidth performance. As mentioned in Sec. I, by properly design-
ing the parameters of the CLTW electrode, this RF loss can be largely
decreased. Figures 1(b) and 1(c) show cross-sectional views of the
EO modulators. It should be noted that the gap g between CLTW
electrodes has a major effect on the modulation efficiency VπL.
Thanks to the 900 nm thick silicon oxide (SiO2) over-cladding layer
on top of the TFLN waveguides, excessive optical absorption losses
are avoided even if a small gap of g = 1.8 μm is adopted. The thick-
ness of the T-segment electrodes is set to 200 nm to ensure an easy
lift-off process for these high-resolution structures, and the thick-
ness of the unloaded electrodes is 1.1 μm, which is at a good balance
between the performance and cost.19 The unloaded signal electrode
has a width of ws = 75 μm, which is designed large to reduce the
conductor losses for the RF waves. In order to compensate the slow-
wave effect introduced by the periodic T-segment structures, the
silicon substrate is partially removed, leaving an air hole beneath the
TFLN waveguides, as shown in Figs. 1(c) and 1(d). This effectively
decreases the permittivity of the material seen by the RF wave. The
parameters of the CLTW electrode are carefully designed to ensure
the impedance and index matching, which gives (r, c, s, t, h, hw, hl)
= (47, 3, 2, 5, 15, 9, 39) μm, and a 35 μm undercut etching of the
silicon substrate is optimal.
A multi-physics field solver based on the finite element algo-
rithm (COMSOL) was used to simulate the electrical and optical
modes of the structure. VπL = 2.10 V cm can be derived with a
low optical propagation loss of ∼0.04 dB/cm for the fundamental
transverse-electrical (TE) mode, as shown in Fig. 2(a). Figure 2(b)
shows the RF field distributions. Unlike EO modulators using the
regular TW electrode, the RF field here is localized around the
gap of the loaded T-segments, instead of the wide unloaded sig-
nal electrode. This helps us increase the overlap of the electrical
and optical fields and hence a low VπL. The frequency-dependent
characteristic impedance Zc, the RF effective index nm, and the RF
loss αm obtained from the simulations are shown in Figs. 2(c)–2(e),
respectively. Figures 2(f) and 2(g) further show effects of differ-
ent undercut etching and h of the CLTW electrode on the RF
effective index nm and characteristic impedance Zc, respectively.
Obviously, these two figures play an important role for achieving
the index and impedance matching. Nearly perfect impedance and
index matching can be achieved up to 100 GHz in this design.
The RF loss can be expressed as αm = α0(f )1/2, where f is the
FIG. 3. Process flow of the proposed
EO modulator: (a) initial LNOI wafer,
(b) definition of passive components on
LNOI, (c) LN hole etching, (d) SiO2 over-
cladding deposition, (e) SiO2 hole etch-
ing to the silicon substrate, (f) first level
metal patterning for the T-segments,
(g) second level metal patterning for
the unload TW electrodes, (h) isotropic
silicon etch to partially remove the sub-
strate, (i) scanning electron microscopy
image of the CLTW electrode, and (j)
optical image of the whole fabricated
device.
APL Photon. 7, 026103 (2022); doi: 10.1063/5.0077232
7, 026103-4
© Author(s) 2022

<!-- page 6 -->
APL Photonics
ARTICLE
scitation.org/journal/app
RF frequency and α0 is the frequency independent characteris-
tic RF loss. Here, α0 of 0.19 dB cm−1 GHz−1/2 can be drawn
from the data shown in Fig. 2(e), which is much lower than
that of the regular TW electrode based modulators as mentioned
above.
The proposed EO modulators using the designed CLTW
electrode were fabricated using the process flow shown in
Figs. 3(a)–3(h). First, an LNOI wafer consisting of 400 nm thick
x-cut TFLN on a 725 μm thick silicon substrate with a 3 μm ther-
mal oxide layer (i.e., the BOX layer) was patterned using an electron
beam lithography system (Raith VOYAGER) on a 300 nm thick
resist (ma-N 2403). Then, a 200 nm thick LN layer was removed
using Ar plasma by means of inductively coupled plasma reactive ion
etching (ICP-RIE) technology with an etching rate of ∼80 nm/min.
FIG. 4. Normalized optical transmissions of the fabricated modulators as a function
of the applied voltage for (a) a 5 mm long device, (b) a 7 mm long device, and (c) a
10 mm long device. The Vπ value for each device is also marked. The insets show
the measured transmissions in a logarithmic scale showing the insertion losses
and ERs of the corresponding devices.
Holes between the T-segments and the unload TW electrodes were
then patterned using ultra-violet contact lithography, and the resid-
ual 200 nm LN layer was etched. The device was then covered with
a 900 nm thick SiO2 layer deposited by plasma enhanced chemi-
cal vapor deposition. Subsequently, the holes were again patterned,
and the over-cladding SiO2 layer and the BOX layer were etched
to expose the silicon substrate. Next, the metal structures made of
gold were fabricated using two lift-off processes for the periodic
T-segments and the unloaded TW electrodes, respectively. Finally,
the holes were patterned for the third time, and isotropic silicon dry
etching using ICP-RIE technology was performed to remove the sil-
icon substrate beneath the modulation section. Figures 3(i) and 3(j)
show some images of the CLTW electrode and the whole fabricated
device, respectively.
FIG. 5. Measured EE reflection S11, EE transmission S12, measured EO S21, and
calculated EO S21 for (a) 5 mm, (b) 7 mm, and (c) 10 mm long devices. Deduced
(d) characteristic impedance Zc, (e) RF loss αm, and (f) RF effective index nm.
APL Photon. 7, 026103 (2022); doi: 10.1063/5.0077232
7, 026103-5
© Author(s) 2022

<!-- page 7 -->
APL Photonics
ARTICLE
scitation.org/journal/app
III. MEASUREMENT AND ANALYSIS
The fabricated EO modulators with modulation section lengths
of 5, 7, and 10 mm were characterized. The laser light was cou-
pled in and out through a pair of GCs. The coupling loss of the
GC was first measured using a reference structure consisting of two
such GCs and a short straight waveguide. The total insertion loss
including the device and the coupling GCs is 8.2 dB at the peak cou-
pling wavelength, and the on-chip loss of ∼0.2 dB for the longest
device (10 mm) can be deduced. Thanks to the thick SiO2 over-
cladding layer to block the metal absorptions, this loss nearly came
alone from the propagation loss of the TFLN waveguide (about
0.15 dB/cm). Figure 4 shows the half-wave voltage Vπ measure-
ments for the three devices with a 100 kHz triangular voltage sweep.
The Vπ values for the 5, 7, and 10 mm long devices are 4.45, 3.22,
and 2.20 V, respectively, corresponding to a VπL of 2.22, 2.25, and
2.20 V cm, respectively, which are consistent with the simulation
results in Fig. 2. The insets in Figs. 4(a)–4(c) show transmissions
of the devices with a static voltage sweep, demonstrating measured
extinction ratios (ERs) of >30, >25, and >20 dB, respectively. It is
believed that the improvement of ERs can be achieved by optimiz-
ing fabrication imperfections, such as a better write-field stitching
and metal electrode alignment.
Next, the EO and electrical–electrical (EE) responses of the
fabricated devices were characterized. As shown in Figs. 5(a)–5(c),
the measured EE transmissions S12 show an ultra-low roll-off of
less than 1.7, 2.4, and 3.5 dB at 67 GHz for the 5, 7, and 10 mm
long devices, respectively. Moreover, the EE reflection S11 is below
−25 dB over the entire measured frequency range for all the devices.
The smooth and flat transmission S12, as well as the low reflection
S11, indicates that the conditions for a good impedance matching
and ultra-low RF losses have been achieved, which can also be seen
from the deduced characteristic impedances Zc and RF losses, as
plotted in Figs. 5(d) and 5(e), respectively. The characteristic RF loss
α0 = 0.36 dB cm−1 GHz−1/2 for the measured device can also be fitted
from Fig. 5(e). This value is about twice the simulation results men-
tioned above. The additional RF loss probably comes from either the
residual silicon substrate or the quality of the deposited gold film.
Figure 5(f) shows the deduced RF index nm, where one can find
that there still exists a slight index mismatching of ∼0.15 between
the optical and RF waves. Longer isotropic undercut etching for the
silicon substrate should be done to further reduce the RF index.
The small signal EO responses S21 were further measured using a
70-GHz photodetector (Finisar XPDV3120). One RF probe was used
to deliver the modulation signal, and the other was used to provide a
50 Ω termination. The 3 dB EO modulation bandwidths of the 5,
7, and 10 mm long devices are all greater than 67 GHz beyond
the measurement limits of our network analyzer. The measured EO
S21 curves are also well matched to our calculated curves, using
the fitted data in Figs. 5(d)–5(f). Despite the slight index mis-
match, the longest fabricated device (10 mm) still shows smooth EO
modulation responses with a roll-off of only 1.4 dB at 67 GHz.
We, then, characterized the performance of the 10 mm long
device for high-speed data transmissions. Figures 6(a) and 6(b)
show the optical eye diagrams for the OOK modulation at 80 and
100 Gb s−1, respectively. We also tested the PAM-4 modulation
FIG. 6. High-speed data transmission
experiments on the fabricated EO mod-
ulator of 10 mm length. Measured opti-
cal eye diagrams for the OOK format
at data rates of (a) 80 Gb s−1 and
(b) 100 Gb s−1. Measured optical eye
diagrams for the PAM-4 format at data
rates of (c) 40 Gbaud (80 Gb s−1) and
(d) 56 Gbaud (112 Gb s−1). (e) Mea-
sured B2B BERs at different received
optical power values for 56 Gbaud PAM-
4 and 100 Gb s−1 OOK.
APL Photon. 7, 026103 (2022); doi: 10.1063/5.0077232
7, 026103-6
© Author(s) 2022

<!-- page 8 -->
APL Photonics
ARTICLE
scitation.org/journal/app
TABLE I. Comparison of several performance metrics for EO modulators on the TFLN platform.
Devices
Insert. loss (dB)
Vπ (V)
Len. of mod. sec. (mm)
EO S21 roll-off
SOI-TFLN17
2.5
5.10
5
−3 dB (at 67 GHz)
SOI-TFLN22
7.6
13.4
5
−1.5 dB (at 106 GHz)
SiN-TFLN23
NMa
3.87
8
−3 dB (at 40 GHz)
TFLN15
0.2
2.3
10
−2.1 dB (at 67 GHz)
0.5
1.4
20
−3 dB (at 45 GHz)
TFLN18 (Q)b
1.5
2.3
10
−0.8 dB (at 50 GHz)
1.5
1.3
20
−1.8 dB (at 50 GHz)
TFLN21 (Q)b
6c
3.4
5
−1.3 dB (at 67 GHz)
This work
0.2
2.2
10
−0.76 dB (at 50 GHz)
−1.4 dB (at 67 GHz)
aNM, not mentioned.
bQ, Quartz substrate.
cThe value is calculated from the total on-chip loss of 17 dB and the coupling loss of 11 dB.
format at 40 Gbaud (80 Gb s−1) and 56 Gbaud (112 Gb s−1), as
shown in Figs. 6(c) and 6(d), respectively. During the measurement,
the data were generated using an arbitrary wave generator (AWG)
with an analog bandwidth of ∼35 GHz (Micram DAC10004). The
eye diagrams were recorded using a sampling oscilloscope with a
bandwidth of 65 GHz (Agilent 81600D). The dynamic ERs are all
better than 9.7 dB in all the above-mentioned eye diagrams. The
back-to-back (B2B) bit-error rates (BERs) of 100 Gb/s OOK and
56 Gbaud PAM-4 signals were further measured using a pre-
amplified receiver that consisted of a variable optical attenuator, an
Erbium doped fiber amplifier with a noise figure of about 4.3 dB, an
optical band pass filter with a 3-nm bandwidth, a 70-GHz photode-
tector, and a real-time oscilloscope with a 59-GHz analog bandwidth
(Lecroy Labmaster 10Zi). The BERs were calculated offline using
digital signal processing algorithms including down-sampling, tim-
ing phase synchronization, and a 11-tap least-mean-square equal-
izer. One can find that for all cases, the BER can drop below the KP-4
forward error correction (FEC) threshold, as shown in Fig. 6(e). No
error floor is observed in the measured power range. It is worth-
while to note that the data transmission performance shown here is
still limited by bandwidths of the equipment, as mentioned above.
Table I summarizes key performances of some demonstrated EO
modulators based on the TFLN platform. One can find that the pro-
posed device here shows better EO responses (a lower roll-off) with
a similar modulation efficiency and insertion loss, as compared to
the state-of-the-art TFLN modulator using the conventional TW
electrode on a silicon substrate.15 It also shows a better optical inser-
tion loss performance (probably due to a more matured fabrication
process on a silicon substrate) with a similar EO response and mod-
ulation efficiency, as compared to the TFLN modulator using the
same type of electrode on a quartz substrate.17
IV. CONCLUSION
In this paper, we have demonstrated TFLN based EO mod-
ulators using the CLTW electrode on a silicon substrate. Exper-
imentally, the devices show a high modulation efficiency VπL of
∼2.20 V cm and ultra-low optical and RF losses. The longest fabri-
cated modulator exhibits a Vπ of 2.2 V and excellent EO modulation
response with a roll-off of 1.4 dB at 67 GHz. Employing the undercut
etching technique, we can compensate the slow-wave effect induced
by the CLTW electrode on a silicon substrate while maintaining
the impedance and index matching. Although such segmented elec-
trode designs have been applied on the TFLN platform on a quartz
substrate,17,20 it is the first time to introduce such a device on a
silicon substrate, which makes the proposed modulator compatible
with silicon photonic integration circuits. Using a hybrid integration
technique, e.g., wafer bonding, a high performance EO modulator
using the CLTW electrode on the silicon-LN hybrid platform can
also be achieved. In Fig. 7, we plot the predicted EO modulation
responses with different Vπ values, using the measured modula-
tion efficiency VπL = 2.20 V cm and RF loss αm in Fig. 5(e) while
considering a perfect impedance and index matching between the
optical and RF waves. A driving voltage of <1 V and 3 dB bandwidth
of >60 GHz can be achieved simultaneously using the proposed
device structure. Moreover, combined with the folded modulator
design,23,24 the length of the TFLN modulator discussed here can
be largely shortened. Based on the above experimental results and
FIG. 7. Predicted EO modulation bandwidth of the proposed device corresponding
to different voltages and modulation lengths. The measured VπL = 2.20 V cm and
αm in Fig. 5(e) are adopted here with a perfect impedance and index matching
assumed.
APL Photon. 7, 026103 (2022); doi: 10.1063/5.0077232
7, 026103-7
© Author(s) 2022

<!-- page 9 -->
APL Photonics
ARTICLE
scitation.org/journal/app
analyses; we firmly believe that our demonstration, with the pos-
sible integration of other silicon photonic devices, would open up
the possibility of achieving sub-1-V driven transmitters supporting
a single-lane 100 Gbaud data rate.
ACKNOWLEDGMENTS
The authors acknowledge financial support from the National
Research and Development Program (Grant No. 2019YFB2203200),
the National Natural Science Foundation of China (NSFC) (Grant
Nos. 62135012, 62105107, 91950205, 61961146003, and 92150302),
the Guangdong Basic and Applied Basic Research Foundation
(Grant No. 2021A1515012215), the Science and Technology Plan-
ning Project of Guangdong Province (Grant No. 2019A050510039),
and the Fundamental Research Funds for the Central Universities
(Grant No. 2021QNA5001).
AUTHOR DECLARATIONS
Conflict of Interest
The authors have no conflicts to disclose.
DATA AVAILABILITY
The data that support the findings of this study are available
from the corresponding authors upon reasonable request.
REFERENCES
1E. L. Wooten, K. M. Kissa, A. Yi-Yan, E. J. Murphy, D. A. Lafaw, P. F. Hallemeier,
D. Maack, D. V. Attanasio, D. J. Fritz, G. J. McBrien, and D. E. Bossi, “A review of
lithium niobate modulators for fiberoptic communications systems,” IEEE J. Sel.
Top. Quantum Electron. 6, 69–82 (2000).
2K. F. Reim, J. Nunn, V. O. Lorenz, B. J. Sussman, K. C. Lee, N. K. Langford,
D. Jaksch, and I. A. Walmsley, “Towards high-speed optical quantum memories,”
Nat. Photonics 4, 218–221 (2010).
3C. Wang, M. Zhang, M. Yu, R. Zhu, H. Hu, and M. Loncar, “Monolithic lithium
niobate photonic circuits for Kerr frequency comb generation and modulation,”
Nat. Commun. 10, 978 (2019).
4Q. Cheng, M. Bahadori, M. Glick, S. Rumley, and K. Bergman, “Recent advances
in optical technologies for data centers: A review,” Optica 5, 1354–1370 (2018).
5C. R. Doerr, “Direct modulation of long-cavity semiconductor lasers,”
J. Lightwave Technol. 14, 2052–2061 (1996).
6M. Li, L. Wang, X. Li, X. Xiao, and S. Yu, “Silicon intensity Mach–Zehnder
modulator for single lane 100 Gb/s applications,” Photonics Res. 6, 109–116
(2018).
7C. Kieninger, Y. Kutuvantavida, D. L. Elder, S. Wolf, H. Zwickel, M. Blaicher,
J. N. Kemal, M. Lauermann, S. Randel, W. Freude, L. R. Dalton, and C. Koos,
“Ultra-high electro-optic activity demonstrated in a silicon-organic hybrid (SOH)
modulator,” Optica 5, 739–748 (2018).
8S. Ummethala, J. N. Kemal, A. S. Alam, M. Lauermann, A. Kuzmin, Y. Kutu-
vantavida, S. H. Nandam, L. Hahn, D. L. Elder, L. R. Dalton, T. Zwick, S. Randel,
W. Freude, and C. Koos, “Hybrid electro-optic modulator combining silicon pho-
tonic slot waveguides with high-k radio-frequency slotlines,” Optica 8, 511–519
(2021).
9C. Koos, J. Leuthold, W. Freude, M. Kohl, L. Dalton, W. Bogaerts, A. L. Giesecke,
M. Lauermann, A. Melikyan, S. Koeber, S. Wolf, C. Weimann, S. Muehlbrandt, K.
Koehnle, J. Pfeifle, W. Hartmann, Y. Kutuvantavida, S. Ummethala, R. Palmer, D.
Korn, L. Alloatti, P. C. Schindler, D. L. Elder, T. Wahlbrink, and J. Bolten, “Silicon-
organic hybrid (SOH) and plasmonic-organic hybrid (POH) integration,”
J. Lightwave Technol. 34, 256–268 (2016).
10A. Melikyan, K. Koehnle, M. Lauermann, R. Palmer, S. Koeber, S. Muehlbrandt,
P. C. Schindler, D. L. Elder, S. Wolf, W. Heni, C. Haffner, Y. Fedoryshyn, D.
Hillerkuss, M. Sommer, L. R. Dalton, D. Van Thourhout, W. Freude, M. Kohl,
J. Leuthold, and C. Koos, “Plasmonic-organic hybrid (POH) modulators for OOK
and BPSK signaling at 40 Gbit/s,” Opt. Express 23, 9938–9946 (2015).
11Y. Ogiso, J. Ozaki, Y. Ueda, H. Wakita, M. Nagatani, H. Yamazaki, M.
Nakamura, T. Kobayashi, S. Kanazawa, Y. Hashizume, H. Tanobe, N. Nunoya, M.
Ida, Y. Miyamoto, and M. Ishikawa, “80-GHz bandwidth and 1.5-V Vπ InP-based
IQ modulator,” J. Lightwave Technol. 38, 249–255 (2020).
12M. Lawrene, “Lithium niobate integrated optics,” Rep. Prog. Phys. 56, 363–429
(1993).
13M. De Micheli, J. Botineau, P. Sibillot, D. B. Ostrowsky, and M. Papuchon,
“Fabrication and characterization of titanium indiffused proton exchanged (TIPE)
waveguides in lithium niobate,” Opt. Commun. 42, 101–103 (1982).
14J. Kondo, A. Kondo, K. Aoki, M. Imaeda, T. Mori, Y. Mizuno, S. Takatsuji, Y.
Kozuka, O. Mitomi, and M. Minakata, “40-Gb/s X-cut LiNbO3 optical modulator
with two-step back-slot structure,” J. Lightwave Technol. 20, 2110–2114 (2002).
15C. Wang, M. Zhang, X. Chen, M. Bertrand, A. Shams-Ansari, S. Chandrasekhar,
P. Winzer, and M. Lonˇcar, “Integrated lithium niobate electro-optic modulators
operating at CMOS-compatible voltages,” Nature 562, 101–104 (2018).
16M. He, M. Xu, Y. Ren, J. Jian, Z. Ruan, Y. Xu, S. Gao, S. Sun, X. Wen, L. Zhou, L.
Liu, C. Guo, H. Chen, S. Yu, L. Liu, and X. Cai, “High-performance hybrid silicon
and lithium niobate Mach–Zehnder modulators for 100 Gbit s−1 and beyond,”
Nat. Photonics 13, 359–364 (2019).
17P. Kharel, C. Reimer, K. Luke, L. He, and M. Zhang, “Breaking voltage-
bandwidth limits in integrated lithium niobate modulators using micro-structured
electrodes,” Optica 8, 357–363 (2021).
18Z. Ruan, J. Hu, Y. Xue, J. Liu, B. Chen, J. Wang, K. Chen, P. Chen, and L.
Liu, “Metal based grating coupler on a thin-film lithium niobate waveguide,” Opt.
Express 28, 35615–35621 (2020).
19J. Cai, C. Guo, C. Lu, A. P. T. Lau, P. Chen, and L. Liu, “Design optimization
of silicon and lithium niobate hybrid integrated traveling-wave Mach-Zehnder
modulator,” IEEE Photonics J. 13, 2200206 (2021).
20X. Liu, B. Xiong, C. Sun, J. Wang, Z. Hao, L. Wang, Y. Han, H. Li, J. Yu, and Y.
Luo, “Wideband thin-film lithium niobate modulator with low half-wave-voltage
length product,” Chin. Opt. Lett. 19, 060016 (2021).
21P. O. Weigel, J. Zhao, K. Fang, H. Al-Rubaye, D. Trotter, D. Hood, J. Mudrick, C.
Dallo, A. T. Pomerene, A. L. Starbuck, C. T. Derose, A. L. Lentine, G. Rebeiz, and S.
Mookherjea, “Bonded thin film lithium niobate modulator on a silicon photonics
platform exceeding 100 GHz 3-dB electrical modulation bandwidth,” Opt. Express
26, 23728–23739 (2018).
22A. Rao, A. Patil, P. Rabiei, A. Honardoost, R. Desalvo, A. Paolella, and S. Fath-
pour, “High-performance and linear thin-film lithium niobate Mach–Zehnder
modulators on silicon up to 50 GHz,” Opt. Lett. 41, 5700–5703 (2016).
23J. Hu, C. Li, C. Guo, C. Lu, A. P. T. Lau, P. Chen, and L. Liu, “Folded thin-
film lithium niobate modulator based on a poled Mach–Zehnder interferometer
structure,” Opt. Lett. 46, 2940–2943 (2021).
24S. Sun, M. Xu, M. He, S. Gao, X. Zhang, L. Zhou, L. Liu, S. Yu, and X. Cai,
“Folded heterogeneous silicon and lithium niobate Mach–Zehnder modulators
with low drive voltage,” Micromachines 12, 823 (2021).
APL Photon. 7, 026103 (2022); doi: 10.1063/5.0077232
7, 026103-8
© Author(s) 2022

