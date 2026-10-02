---
paper_id: kharel2021
source_url: https://doi.org/10.1364/optica.416155
doi: 10.1364/optica.416155
license: Optica OA License v1 (journal VOR, per Crossref); cached PDF is arXiv v1 and its license is not shown in the PDF (unverified)
sha256: 9296fea00a56a010598f7e919016d7e11916ae66455228db271e92830b26fcd7
pages: 7
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Breaking voltage-bandwidth limits in integrated lithium niobate modulators using
micro-structured electrodes
Prashanta Kharel,1 Christian Reimer,1 Kevin Luke,1 Lingyan He,1 and Mian Zhang1, ∗
1HyperLight, 501 Massachusetts Avenue, Cambridge, Massachusetts 02139, USA
(Dated: November 30, 2020)
Electro-optic modulators with low voltage and large bandwidth are crucial for both analog and
digital communications. Recently, thin-ﬁlm lithium niobate modulators have enable dramatic per-
formance improvements by reducing the required modulation voltage while maintaining high band-
widths. However, the reduced electrode gaps in such modulators leads to signiﬁcantly higher mi-
crowave losses, which limit electro-optic performance at high frequencies. Here we overcome this
limitation and achieve a record combination of low RF half-wave voltage of 1.3 V while maintaining
electro-optic response with 1.8-dB roll-oﬀat 50 GHz. This demonstration represents a signiﬁcant
improvement in voltage-bandwidth limit, one that is comparable to that achieved when switch-
ing from legacy bulk to thin-ﬁlm lithium niobate modulators. Leveraging the low-loss electrode
geometry, we show that sub-volt modulators with > 100 GHz bandwidth can be enabled.
I.
INTRODUCTION
Low voltage,
broadband and high signal quality
electro-optic (EO) modulators are paramount to appli-
cations spanning from radiofrequency (RF) analog links
to digital optical communication networks. Today, most
EO modulators require high voltage drives at microwave
frequencies > 50 GHz because of reduced high frequency
performances. This high RF voltage requires high-speed
and high-gain electrical ampliﬁers which pose challenges
for power consumption, linearity, signal-to-noise ratio
and cost. As the demands for high baud rate for digi-
tal communication and higher carrier frequency for ana-
log link continue to grow, aforementioned challenges only
exacerbate at even higher microwave frequencies (e.g.
> 100 GHz), as modulators’ eﬃciency continues to de-
crease, electronic ampliﬁers gain also diminishes at higher
microwave frequencies.
The challenge of achieving low voltage at high RF fre-
quencies is universal across diﬀerent photonics platforms,
considering the stringent requirement of simultaneously
retaining other desired properties including good linear-
ity, low insertion loss, high extinction ratio and high-
power handling ability. For traditional lithium niobate
(LN) modulators based on ion-indiﬀusion and proton ex-
change, a half-wave voltage Vπ ∼3.5 V, which is de-
ﬁned as the voltage needed to switch the modulation from
maximum to the nearest minimum transmission, is typi-
cally needed at low RF frequencies (e.g. 1 GHz). The EO
response is attenuated by 3-6 dB for > 50 GHz [1]. This
translates into a RF voltage requirement as high as > 7
V for modulation frequencies > 50 GHz. On integrated
platforms such as silicon, modulators typically have Vπ ∼
4-6 V and 35 GHz bandwidth [2, 3]. Extending this per-
formance to 50 GHz and beyond also points to very high
voltage (> 10 V) requirement. Indium phosphide (InP)
modulators have achieved better voltage and bandwidth
performances.
For example, Vπ = 1.5 V and 80 GHz
3-dB bandwidth has been achieved on a diﬀerential RF
drive architecture [4]. In addition, sub-volt single-drive
modulators have also been achieved with 67 GHz 6-dB
bandwidth on III-V platforms [5], but the extinction ra-
tio was limited to 3 dB and drive voltage needed to be
increased in order to accommodate higher optical power
[6]. Organic polymer modulators have shown excellent
voltage-bandwidth performances [7] but often the per-
formances need to be compromised in order to improve
stability for practical uses [8]. Plasmonic-organic hybrid
can provide extremely high bandwidths, although modu-
lation voltage and on-chip insertion loss remain relatively
high [9].
II.
SEGMENTED TRAVELING WAVE LN
MODULATOR DESIGN
Thin-ﬁlm LN modulators emerged recently as a strong
contender for next generation low-voltage and high band-
width EO modulators. This is because thin-ﬁlm LN mod-
ulators oﬀer signiﬁcantly improved voltage-bandwidth
performance over legacy LN platforms all while preserv-
ing key LN material advantages such as linear response,
high extinction ratio, high optical power handling abil-
ity and low on-chip insertion loss. However, for thin-ﬁlm
LN modulators, Vπ, especially at frequencies > 50 GHz,
remains > 3 V [10–12]. Extrapolating to 100 GHz shows
an expected Vπ > 4 V. Such performances have also been
corroborated theoretically as being close to the tradi-
tional design limit [13–16]. While the voltage-bandwidth
performances are a signiﬁcant improvement over legacy
bulk LN technologies, the level of RF voltages achieved
in existing thin-ﬁlm LN modulators at >50 GHz are still
very high for typical electronics drivers. For example, a
CMOS driver with a 100 GHz analog bandwidth, would
produce ∼0.5 V output voltage at high frequencies [17].
Therefore, a lower voltage modulator would dramatically
improve device performance such as energy consumption,
sensitivity, and noise ﬁgures which all scales quadratically
with RF Vπ [18].
Here we break the voltage-bandwidth trade-oﬀlimit in
arXiv:2011.13422v1  [physics.app-ph]  26 Nov 2020

<!-- page 2 -->
2
Quartz
LN
Au
g
(b)
(c)
(e)
s
h
(c)
(d)
h
s
r
t
c
(a)
RF signal
Optical signal
FIG. 1. Low-voltage high bandwidth traveling-wave integrated lithium niobate (LN) modulator with segmented electrodes. (a)
Artistic top view of the modulator design (not to scale) where RF signal co-propagates with the optical signal . (b) Scanning
electron microscope image of the fabricated device. Scale bar: 50 µm. (c) Artistic angled view of a regular electrode design.
Current (red) crowds at edges of the conductors. (d) Artistic angled view of a segmented design. Current crowds less and
distribute more uniformly. (e) Cross sectional view of the phase shifter. Design parameters (g, h, s, t, r, c) = (5, 6, 2, 6, 45, 5)µm.
integrated LN modulators using micro-structured elec-
trodes that dramatically reduce microwave losses while
preserving EO modulation eﬃciency as well as other de-
sired properties including high extinction ratio and low
on-chip optical loss. We experimentally demonstrate a
single drive EO modulator with Vπ,1GHz = 1.3 V and
Vπ,50GHz = 1.6 (EO roll-oﬀof 1.8 dB). We further show
that this design can be adapted to achieve sub-volt mod-
ulators with > 100 GHz 3-dB EO bandwidth. At the
same time, we maintain a on-chip loss of < 1 dB and
high extinction ratio of 20 dB. Notably, the performance
gain achieved using the micro-structured electrode design
over regular electrodes on thin-ﬁlm LN is comparable to
the improvement attained when transitioning from bulk
to thin-ﬁlm LN modulators.
The EO modulator employs traveling wave design on a
x-cut thin-ﬁlm LN-on-insulator (LNOI) platform, where
an input light is split into two arms of a Mach-Zehnder in-
terferometer (MZI) and co-propagate with a microwave
drive signal in a transmission line electrode [10].
The
traveling microwave signal modulates the light through-
out the lengths of the electrodes in a push-pull conﬁgu-
ration, inducing a phase advance in one arm and phase
delay in the other. In principle, traveling wave modula-
tors in LN with a long enough electrode can achieve THz
bandwidth and millivolt driving voltage since Pockels ef-
fect takes place on femtosecond time scale [19]. In prac-
tice, bandwidth and voltage performance is limited by
three key factors: 1) microwave loss in the transmission
line causing driving voltage to be attenuated over length
of the electrode; 2) ﬁnite velocity mismatch between the
electrical and optical traveling signal causing modulation
to cease accumulating; and 3) design trade-oﬀs that re-
duce modulation eﬃciency per unit length which could
require even longer devices to achieve a low voltage thus
further limiting the bandwidth.
In other words, to achieve high-bandwidth and low-
voltage operations on LN, the speed of the electrical sig-
nal should match that of the optical signal, microwave
loss (i.e. the attenuation of electrical current ﬂowing in
the direction of the traveling wave) should be low, and
the electric ﬁeld should be strong between the electrode
gaps so that the modulation is phase matched and can
eﬃciently accumulate along the traveling wave direction.
While modulation eﬃciency can be increased and ve-
locity matching is more readily maintained in thin-ﬁlm
LN modulators when compare to legacy bulk LN modu-
lators [20], microwave loss unfortunately increases signif-
icantly in integrated LN electrodes in comparison to bulk
LN designs [13, 20, 21]. Microwave electrode losses orig-
inate from two sources: substrate absorption loss (e.g.
LN, Si) and ohmic conductor loss from ﬁnite resistivity
of metals, with the latter being the dominant loss mech-
anism in existing thin-ﬁlm LN modulators. This is be-
cause the smaller electrode gaps in thin-ﬁlm LN modula-
tors, enabled by high conﬁnement optical waveguides, im-
prove key eﬃciency metrics like half-wave voltage length
product (Vπ ·L) at the expense of dramatically increased
ohmic loss. The narrow metal gap, on the order of a few
micrometers, causes the electrical current to crowd close
to the gap from the largely increased capacitance, eﬀec-
tively reducing conductor area and thus increasing RF

<!-- page 3 -->
3
TABLE I. Comparison of simulated performance for various electrode designs
Electrode design
(substrate)
Signal width
(µm)
Gap
(µm)
Vπ · L
(V·cm)
Z (Ω)
RF loss
(dB cm−1GHz−1/2)
Est.3-dB EO
bandwidth for
Vπ,DC = 1V (GHz)
Regular (Si)
30
5
2.1
39
0.75
27
Regular (Si)
30
7
2.6
45
0.64
25
Regular (Si)
30
10
3.7
48
0.57
17
Regular (Q)
100
5
2.1
41
0.37
13*
Segmented (Q)
100
5
2.1
42
0.21
228
Q: quartz. Z: impedance.
∗Velocity mismatch assumed.
loss. As a result, non-ideal trade-oﬀs such as increasing
the electrode gap which lowers modulation eﬃciency (in-
creases Vπ·L) and/or reducing electrode length have been
predicted in order to maintain ﬂat RF responses (Table
1) [13, 14, 16]. Such design trade-oﬀs lead to underuti-
lization of a tightly conﬁned optical mode for eﬃcient EO
modulation on on integrated LN platform.
In contrast, our design maintains the high modulation
eﬃciency (low Vπ · L) on thin-ﬁlm LN platform while
reducing the electrode losses and maintaining velocity
matching condition. To achieve this, we employ a trav-
eling wave electrode with micro-structures to control the
ﬂow of currents.
The micro-structured electrodes con-
sist of rectangular channel electrode regions like conven-
tional coplanar transmission line designs and segments
extending out from the main electrode (Fig. 1a,b). The
segments prevent electrical current from ﬂowing in the
closest gap region, while allowing the current to be dis-
tributed more uniformly in the wide channel region. As
a result, the eﬀective conductor size is increased and the
ohmic loss in the electrode is reduced without having to
increase the gap (g) between the electrodes (Fig. 1c,d).
Such micro-structured electrodes (also known as seg-
mented electrodes) have been used previously on semi-
conductor substrates to help velocity matching [22] and
also moderately improve conductor loss in III-V and sil-
icon modulators [23, 24]. On insulators with high per-
mittivity such as LN (ϵLN ∼30), such structures have
not been previously implemented to improve RF perfor-
mances. This is because while the extent of improvement
of microwave loss with such electrode design on insulator
was unknown, a collateral and deleterious eﬀect of the
segmented design is that the microwave velocity would
be signiﬁcantly reduced relative to a rectangular elec-
trode design due to the increased capacitance per unit
length of the transmission line from the segments. For
traditional LN substrate, this slow wave eﬀect would be
detrimental for velocity matching since RF velocity is al-
ready slower than optical group velocity to begin with
[1]. On thin-ﬁlm LN on silicon substrate, the slow wave
eﬀect is also undesirable since silicon substrate with a
reasonably thick silicon dioxide insulating layer can al-
ready provide a good velocity match between light and
microwave [10, 20].
Here we instead take advantage of the slow wave eﬀects
by using a quartz substrate, which has a near ideal mi-
crowave properties with low permittivity (ϵQz ∼4.5) and
microwave absorption tangent < 10−4 [25]. On a LN thin
ﬁlm with quartz substrate, one would obtain a RF index
∼1.8 for conventional rectangular electrodes, which is
signiﬁcantly lower than the typical optical group index
of ∼2.2. While this velocity mismatch would be detri-
mental for high speed operations for regular electrode
designs [26, 27], we utilizes it to our advantage to de-
sign segments that pushes the current away from narrow
gaps. We show that our segmented design can be used
to achieve RF index of ∼2.2, which can be ﬁne-tuned
by the dimensions of the segments to match the optical
group velocity in LN precisely. Importantly, as opposed
to marginal RF loss improvements in other semiconduc-
tor platforms, the segmented electrode design in thin-ﬁlm
LN reduces the RF loss substantially.
We simulated the performance of segmented elec-
trodes and regular electrodes using ﬁnite element meth-
ods (HFSS) and compare the results in Table I. For reg-
ular electrodes on silicon, the ﬁlm stacks used were LN:
600 nm, SiO2: 2 µm and Si: 500 µm. For electrodes on a
quartz substrate the ﬁlm stacks wereare LN: 600 nm and
quartz: 500 µm. The segmented electrode uses a param-
eter set of (h, s, t, r, c) = (6, 2, 6, 45, 5) µm as deﬁned in
Fig. 1. It is clear from Table 1. that using segmented
electrodes, the modulator 3-dB bandwidth can be im-
proved by more than 8 times to 228 GHz in comparison
to existing approaches for the same designed DC Vπ.
III.
MEASUREMENTS AND ANALYSIS
We fabricated LN modulator device using a wafer
(NanoLN) consisted of a 600 nm thick x-cut LN thin-
ﬁlm on a 500 µm thick quartz handle. We patterned the
optical device with electron beam lithography and etched
350 nm of LN ﬁlm to deﬁne integrated waveguides using
a previously reported method [28]. We then patterned
800 nm thick gold electrodes with a lift-oﬀprocess. The
device was also cladded with 1 µm thick silicon dioxide,

<!-- page 4 -->
4
10
20
30
40
50
0
0
2
4
6
8
10
Regular
Segmented
Fit
αRF (dB/cm)
Frequency (GHz)
2
4
6
8
10
0
Frequency (          )
(b)
0.69 dBcm-1GHz-1/2
0.26 dBcm-1GHz-1/2
(a)
GHz
FIG. 2. Measured RF loss of a 10-mm long regular electrode on LN-on-silicon and segmented electrode on LN-on-quartz with
square-root ﬁtting on (a) linear frequency axis (b) square-root frequency axis.
deposited by chemical vapor deposition.
We show that the segmented electrode completely
transformed the RF performance while preserving other
desired modulator performances. We measured the elec-
trical loss on the segmented electrode using a 50-Ωvec-
tor network analyzer (VNA) with all the RF components
up to the device de-embedded. We obtain RF loss of 2
dB/cm in the segmented electrode at 50 GHz in compar-
ison to 7 dB/cm in regular electrode design with identi-
cal gold electrode thickness of 800 nm and electrode gap
g = 5µm (Fig. 2a). We also measured a RF phase index
of 2.23 which agrees with our simulation.
Our measured results have excellent agreement with
theory. Ohmic loss in the electrode αRF, is ∝Lf 1/2 as
a result of the skin eﬀect in metal [29], where L is the
length of the electrode and f is microwave frequency. We
measure Regular electrodes on thin-ﬁlm LN have αRF,reg
= 0.69 dB cm−1GHz−1/2 compare to αRF,seg = 0.26 dB
cm−1GHz−1/2 for the segmented electrodes. The square
root dependence of the loss is clear from Fig. 2b, corrob-
orating the assumption that ohmic loss is the dominate
source of RF attenuation. Note that microwave absorp-
tion loss has a linear dependence with microwave fre-
quency so it would show up as a superlinear contribution
in Fig. 2b at high frequencies. Our results show that the
linear loss is still too small in comparison to conductor
loss up to 50 GHz. Based on our ﬁtting, we estimate an
upper limit for RF absorption on LN on quartz substrate
of 0.007 dB cm−1GHz−1, which translates to less than
0.35 dB/cm at 50 GHz and 0.7 dB/cm at 100 GHz. Using
a loss tangent of 0.004 for LN bulk crystals [30], our simu-
lation produced absorption loss of 0.003 dB cm−1GHz−1,
which agrees well with the limited estimated by our mea-
surements.
We measured the EO performance of the modulators
using a telecom wavelength tunable laser at 1560 nm. We
coupled light through a pair of grating couplers with total
insertion loss 13 dB and on-chip loss < 1 dB estimated by
comparing transmission loss of a pair of gratings couplers
with and without the modulator. Majority of the optical
loss results from the 6 dB/facet loss of grating couplers
due to the absence of a high index substrate. The total
insertion loss can be dramatically reduced by using edge
couplers [31], or buried metal back reﬂectors [32] to ∼4
dB. We obtained DC Vπ of 1.35 V on an oscilloscope for a
20-mm long modulator. The extinction of the modulator
was measured to be about 20 dB (Fig 3a).
The ultralow RF loss enabled measured EO responses
of only 1.8 dB attenuation for the 20-mm long modulator
at 50 GHz comparing to reference Vπ at 1 GHz (Fig. 3b).
We choose to reference the EO roll-oﬀto Vπ,1GHz which
is also conventional, since RF Vπ is overall a better met-
ric for gauging modulator performances. This is because
LN modulators at close to DC frequencies are prone to
additional slow eﬀects such as photorefractive eﬀect that
can lead to over- or underestimation of Vπ. Our measure-
ments at 1 GHz using a Bessel function transform method
[33] indicate Vπ,1GHz,20-mm = 1.3 V. In other words, the
RF Vπ at 50 GHz is a record low of only 1.6 V. We also
measured a 10-mm modulator with segmented electrode,
which shows only 0.8 dB roll-oﬀat 50 GHz relative to a
measured Vπ,1GHz,10-mm = 2.3 V. The electrical reﬂection
(S11) from the electrode for both cases is maintained be-
low -15 dB. The measured RF Vπ for 20-mm electrode is
higher than expected due to ﬁnite resistance of the thin
metal electrode, which can be improved with thicker met-
als.
IV.
DISCUSSION
The ﬂexibility of substrate engineering and micro-
structured electrode design on thin-ﬁlm LN enabled dra-
matic performance improvement over standard electrode
designs on LN-on-silicon substrates.
The current seg-
mented design is estimated to work and behave like a

<!-- page 5 -->
5
-30
-25
-20
-15
-10
-5
0
-3 dB
-6 dB
10
20
30
40
50
EO response & S11(dB)
Frequency (GHz)
10-mm: Vπ,1GHz = 2.3 V
20-mm: Vπ,1GHz = 1.3 V
-1.5
-1.0
-0.5
0
0.5
0
-10
-20
Voltage (V)
Transmission (dB)
ER = 20 dB
Vπ,20-mm, 1MHz = 1.35 V
(b)
(a)
FIG. 3. Electro-optic performance of segmented LN modulators (a) Measured DC Vπ and extinction for a 20-mm long modulator.
(b) Measured EO response and electrical reﬂection (S11) referenced to RF Vπ at 1 GHz for a 10-mm and a 20-mm long modulator.
The response is normalized to photodiode electric signal, i.e. Vπ,3-dB ∼1.4Vπ,1GHz and Vπ,6-dB ∼2Vπ,1GHz
3-dB bandwidth (GHz)
100
-10
-8
-6
-4
-2
0
EO response (dB)
(a)
(b)
-3 dB
-6 dB
50
100
150
200
20-mm regular (silicon)
20-mm segmented (quartz)
20-mm segmented (air)
Segmented thin-ﬁlm
Regular thin-ﬁlm
Legacy
10
1
0.1
1
10
This work
Frequency (GHz)
Vπ at 1 GHz (V)
[10]
[12]
[11]
[10]
[33]
[33]
[34]
FIG. 4.
Comparison of micro-structured modulator performance to prior designs.
(a) Predicted performances of 20-mm
long electrode for regular and segmented design up to 200 GHz bandwidth. Material legends indicate the substrate handle.
(b) Voltage-bandwidth performance comparison of legacy LN (0.3 dB cm−1GHz−1/2, 15 V·cm), thin-ﬁlm regular LN (0.69
cm−1GHz−1/2, 2.1V·cm) and thin-ﬁlm segmented LN modulators (0.26 cm−1GHz−1/2, 2.3 V·cm). The shaded areas correspond
to improved design space with other substrates. The bandwidth for this work is extrapolated as shown in (a).
lumped element until ∼300 GHz, where frequency de-
pendent phase-shift is expected to cause velocity mis-
match [23]. This frequency limit can be readily overcome
by using smaller segments structure allowing operation in
the THz regimes.
We analyze the bandwidth and voltage trade-oﬀs in the
segmented design (this work) over integrated LN modu-
lators. We extrapolate the expected voltage-bandwidth
performance based on our measurements up to 200 GHz
(Fig 4a). We see that for the same electrode length of
20 mm, the signiﬁcant reduction of RF loss in segmented
design could lead to 180 GHz conductor loss limited 3-dB
bandwidth in comparison to regular thin-ﬁlm modulators
that have 40 GHz bandwidth (Fig 4a). The performance
of the segmented electrode design could be increased fur-
ther using an even lower permittivity substrate such as
fused silica (ϵfs = 3.8) or air (ϵair = 1).
Decrease in
substrate permittivity allows longer segments, which fur-
ther reduces current crowding while still maintaining per-
fect velocity matching conditions. We estimate that mi-
crowave loss 0.15 dB cm−1GHz−1/2 is within reach for
such designs. This improvement in microwave loss should
permit modulators with Vπ of 780 mV at 100 GHz using
an electrode length of 25 mm and optimized gap (Vπ of
650 mV at 1 GHz and > 200 GHz 3-dB bandwidth with
respect to 1 GHz). Note that for such low drive volt-
ages designs, while the electrode length is substantial,
the tight optical mode conﬁnement in thin-ﬁlm LN and
small lateral size of the electrode permits bending of the
electrodes to ﬁt in a small footprint < 10 mm × 0.5 mm.
The micro-structured electrodes push integrated LN
modulator performance into a completely new perfor-

<!-- page 6 -->
6
mance space. Here we compare the voltage-bandwidth
limits of legacy modulators [34, 35], regular thin-ﬁlm LN
modulators [10, 11] and here-presented segmented thin-
ﬁlm LN modulators (Fig 4b).
The voltage-bandwidth
performance due to the current crowding eﬀect, typi-
cally have 5-7 dB/cm RF loss at 50 GHz (0.7-1 dB
cm−1GHz−1/2), which is much higher than typical RF
loss in legacy modulators of ∼2 dB/cm at 50 GHz (∼0.3
dB cm−1GHz−1/2). Still, thin-ﬁlm LN modulators out
performances legacy designs due to the nearly 5 times
reduction in Vπ · L. On segmented electrode platform,
we can maintain the Vπ · L close to the regular elec-
trode and at the same time improve RF loss to ∼0.2
dB cm−1GHz−1/2. Lower index substrates allow the loss
to be improved further. From Fig. 4b, we can see that
the new segmented design leads to a similar performance
gain to what regular thin-ﬁlm design has achieved over
legacy bulk LN.
V.
CONCLUSION
We have demonstrated an integrated LN EO modula-
tor with ultra-ﬂat frequency response and low RF Vπ us-
ing segmented traveling-wave electrode on low permittiv-
ity substrate. We show that sub-volt level of microwave
driving voltage can be achieved even at frequency > 100
GHz. We believe the signiﬁcantly improved EO modula-
tion performance in micro-structured thin-ﬁlm LN mod-
ulators will lead to a paradigm shift for both analog and
digital ultra-highspeed RF links. For example, for digital
applications with sub-volt modulators, high speed elec-
tronic drivers may have largely reduced gain-bandwidth
requirement or possibly be completely by-passed with
the modulators directly driven from electronic processors
[36].
∗mian@hyperlightcorp.com
[1] E. Wooten, K. Kissa, A. Yi-Yan, E. Murphy, D. Lafaw,
P. Hallemeier,
D. Maack,
D. Attanasio,
D. Fritz,
G. McBrien,
and D. Bossi, IEEE Journal of Selected
Topics in Quantum Electronics 6, 69 (2000).
[2] J. Zhou, J. Wang, L. Zhu, Q. Zhang, Q. Zhang,
and
J. Hong, in Optical Fiber Communication Conference
(Optical Society of America, 2019) p. Tu2H.2.
[3] M. Li, L. Wang, X. Li, X. Xiao,
and S. Yu, Photonics
Research 6, 109 (2018).
[4] Y. Ogiso, J. Ozaki, Y. Ueda, H. Wakita, M. Na-
gatani, H. Yamazaki, M. Nakamura, T. Kobayashi,
S. Kanazawa, Y. Hashizume, H. Tanobe, N. Nunoya,
M. Ida, Y. Miyamoto, and M. Ishikawa, Journal of Light-
wave Technology 38, 249 (2020).
[5] S. Dogru and N. Dagli, Optics Letters 39, 6074 (2014).
[6] P. Bhasker, J. Norman, J. Bowers, and N. Dagli, Journal
of Lightwave Technology 38, 2308 (2020).
[7] C. Kieninger, Y. Kutuvantavida, D. L. Elder, S. Wolf,
H. Zwickel, M. Blaicher, J. N. Kemal, M. Lauermann,
S. Randel, W. Freude, L. R. Dalton, and C. Koos, Optica
5, 739 (2018).
[8] C. Kieninger, Y. Kutuvantavida, H. Miura, J. N. Kemal,
H. Zwickel, F. Qiu, M. Lauermann, W. Freude, S. Ran-
del, S. Yokoyama,
and C. Koos, Optics Express 26,
27955 (2018).
[9] M. Burla, C. Hoessbacher, W. Heni, C. Haﬀner, Y. Fe-
doryshyn, D. Werner, T. Watanabe, H. Massler, D. L.
Elder, L. R. Dalton, and J. Leuthold, APL Photonics 4,
056106 (2019).
[10] C. Wang, M. Zhang, X. Chen, M. Bertrand, A. Shams-
Ansari, S. Chandrasekhar, P. Winzer,
and M. Loncar,
Nature 562, 101 (2018).
[11] M. Xu, M. He, H. Zhang, J. Jian, Y. Pan, X. Liu,
L. Chen, X. Meng, H. Chen, Z. Li, X. Xiao, S. Yu, S. Yu,
and X. Cai, Nature Communications 11, 3911 (2020).
[12] A. N. R. Ahmed, S. Shi, A. Mercante, S. Nelan, P. Yao,
and D. W. Prather, APL Photonics 5, 091302 (2020).
[13] A. Honardoost, F. A. Juneghani, R. Saﬁan, and S. Fath-
pour, Optics Express 27, 6495 (2019).
[14] A. Honardoost, R. Saﬁan, A. Rao, and others, J. Light-
wave Technol. (2018).
[15] R. Saﬁan, M. Teng, L. Zhuang,
and S. Chakravarty,
Optics Express 28, 25843 (2020).
[16] W. Peter Orlando, V. Forrest, Z. Jie, L. Huiyan,
and
M. Shayan, Journal of Physics: Photonics (2020).
[17] X. Chen, S. Chandrasekhar, S. Randel, G. Raybon,
A. Adamiecki, P. Pupalaikis,
and P. J. Winzer, Jour-
nal of Lightwave Technology 35, 411 (2017).
[18] V. J. Urick, J. D. McKinney, and K. J. Williams, Fun-
damentals of microwave photonics, Wiley series in mi-
crowave and optical engineering (Wiley, 2015).
[19] R. W. Boyd, Nonlinear Optics (Elsevier, 2008).
[20] A. Rao and S. Fathpour, IEEE Journal of Selected Topics
in Quantum Electronics 24, 1 (2018).
[21] M. Doi, M. Sugiyama, K. Tanaka, and M. Kawai, IEEE
J. Sel. Top. Quantum Electron. 12, 745 (2006).
[22] S. JaeHyuk, C. Ozturk, S. R. Sakamoto, Y. J. Chiu, and
N. Dagli, IEEE Transactions on Microwave Theory and
Techniques 53, 636 (2005).
[23] J. Shin, S. R. Sakamoto, and N. Dagli, Journal of Light-
wave Technology 29, 48 (2011).
[24] R. Ding, Y. Liu, Y. Ma, Y. Yang, Q. Li, A. E. Lim, G. Lo,
K. Bergman, T. Baehr-Jones, and M. Hochberg, Journal
of Lightwave Technology 32, 2240 (2014).
[25] R. G. Geyer and J. Krupka, IEEE Transactions on In-
strumentation and Measurement 44, 329 (1995).
[26] V. E. Stenger, J. Toney, A. PoNick, D. Brown, B. Griﬃn,
R. Nelson, and S. Sriram, in 2017 European Conference
on Optical Communication (ECOC) (2017) pp. 1–3.
[27] A. J. Mercante, S. Shi, P. Yao, L. Xie, R. M. Weikle, and
D. W. Prather, Optics Express 26, 14810 (2018).
[28] M. Zhang, C. Wang, R. Cheng, A. Shams-Ansari,
and
others, Optica (2017).
[29] S. Haxha, B. M. A. Rahman,
and K. T. V. Grattan,
Applied Optics 42, 2674 (2003).
[30] M. Lee, Applied Physics Letters 79, 1342 (2001).
[31] L. He, M. Zhang, A. Shams-Ansari, R. Zhu, C. Wang,
and L. Marko, Opt. Lett. 44, 2314 (2019).

<!-- page 7 -->
7
[32] Z. Chen, R. Peng, Y. Wang, H. Zhu, and H. Hu, Optical
Materials Express 7, 4010 (2017).
[33] R. Nagarajan, “Technique for measuring the vpi-AC of a
mach-zehnder modulator,” (1999-09-22).
[34] Thorlabs, “Lithium niobate electro-optic modulators,
ﬁber-coupled,” (2020).
[35] EOSpace, “40+ GB/s MODULATORS,” (2020).
[36] K. Li, S. Liu, D. J. Thomson, W. Zhang, X. Yan,
F. Meng,
C. G. Littlejohns,
H. Du,
M. Banakar,
M. Ebert, W. Cao, D. Tran, B. Chen, A. Shakoor,
P. Petropoulos, and G. T. Reed, Optica 7, 1514 (2020).

