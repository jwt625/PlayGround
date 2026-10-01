---
paper_id: ogiso2016
source_url: https://doi.org/10.1049/el.2016.2987
doi: 10.1049/el.2016.2987
license: publisher-copyright
sha256: 00907945f51b591d25c6457e109a0435a0bac91cc7789bbb622ebe036e421673
pages: 2
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
100 Gb/s and 2 V Vπ InP Mach-Zehnder
modulator with an n-i-p-n heterostructure
Y. Ogiso✉, J. Ozaki, N. Kashio, N. Kikuchi, H. Tanobe,
Y. Ohiso and M. Kohtoku
An ultra-high bandwidth (BW) and a low Vπ InP Mach-Zehnder modu-
lator with an n-i-p-n heterostructure is proposed. The combination of
the n-i-p-n heterostructure and the capacitive-loaded travelling-wave
electrode provides a modulator with extremely low electrical loss.
The device exhibits a 3 dB electro-optic BW of over 67 GHz and a
Vπ of 2.0 V. A 100 Gb/s non-return-to-zero on–off keying modulation
with an extinction ratio of over 10 dB is also realised.
Introduction: An ultra-high-speed and low half-wavelength-voltage
(Vπ) optical Mach-Zehnder modulator (MZM) is a key component for
both coherent systems operating beyond 100 Gb/s and intensity-
modulation direct-detection systems operating beyond 100 Gb/s/λ. In
general, there is trade-off between electro-optic bandwidth (EO-BW)
and Vπ because the electro-optic (EO) interaction depends strongly on
electrode length which also affects the EO-BW. Thus, previously
reported ultra-high speed modulators tend to shorten the electrode
length, and an increase in Vπ is inevitable [1, 2]. Recently, broadband
InP modulators with a low Vπ (<2 V) were demonstrated by utilising a
low-loss electrical line such as a capacitive-loaded travelling-wave elec-
trode (CL-TWE) [2–5]. However, although these bandwidths (BWs) are
sufﬁcient for a 32 Gbaud modulation, they are not sufﬁcient for higher
baud rate modulations, e.g. over 64 and 100 Gbaud modulations. The
main obstacle to higher-speed modulation is the high series resistance
of the semiconductor. In particular, the p-doped layer has a contact
and bulk resistance about one order of magnitude higher than an
n-doped layer. Therefore, the resistance of the p-doped layer must be
lower if we are to extend the BW. In this Letter, we report a new high-
speed MZM with a low Vπ. By employing both an n-i-p-n heterostruc-
ture, where a p-doped cladding layer replaces the n-doped layer, [6] and
a CL-TWE, we can realise a modulator with an extremely low electrical
loss. The device exhibits a 3 dB EO-BW of over 67 GHz and a Vπ of
2.0 V with an excess absorption loss of <0.5 dB. Furthermore, we
also demonstrated up to 100 Gb/s non-return-to-zero on–off keying
(NRZ-OOK) modulations with an extinction ratio (ER) of over 10 dB.
Device structure: Fig. 1 shows a schematic illustration and cross-
section diagram of the Mach-Zehnder (MZ) modulation region. An
n-contact layer, an n-InP cladding layer, an undoped multi-quantum
well (InGaAlAs/InAlAs) core layer, a p-InAlAs electron blocking
layer, an n-InP cladding layer, and an n-contact layer were grown
from the top on an semi-insulating (SI) InP substrate by using horizontal
low-pressure metal-organic vapour-phase epitaxy. The n-doped over-
cladding layer in the non-modulation region was replaced by an SI clad-
ding
layer
to
achieve
electrical
isolation.
The
MZM
has
inverted-trapezoidal-shape ridge waveguides in the modulation region
and deep ridge waveguides in the I/O passive waveguide region.
These waveguides are formed along the [011] stripe direction to
obtain a synergistic EO effect [6]. The CL-TWE is then plated on bis-
benzocyclobutene. The electrode conﬁguration is based on a conven-
tional series push–pull drive (single-ended 50 Ω design) [1–4]. The
total travelling-wave electrode is 3 mm long.
MZM
[011]
BCB
Sl-lnP sub.
DC bias
[01–1]
[1–00]
n
n
p
i
Fig. 1 Schematic illustration and cross-sectional diagram of MZ modulation
region
Experimental results: First, we measured the DC ER characteristics for
operating wavelengths of 1530, 1540, 1550, and 1560 nm as shown in
Fig. 2. A differential voltage was applied across the electrode as with
a high-frequency response. The Vπ for all the measured wavelengths
was set at 2.0 V (DC) by adjusting the DC bias voltage where the
excess absorption loss was <0.5 dB. The ER exceeded 22 dB for the
entire C-band and was as high as 27 dB for a wavelength of 1560 nm.
The entire insertion loss was 10 dB, which included the coupling loss
of the lensed ﬁbre (4 dB/facet) and the on-chip loss was estimated to
be 2 dB.
0
–5
–10
–15
–20
–25
transmittance, dB
–30
–3
–2
–1
0
Vp: 2 V
1530 nm
1540 nm
1550 nm
1560 nm
differential voltage, V
1
2
3
Fig. 2 ER characteristics
We then measured the radio frequency (RF) and EO responses. Fig. 3
shows the small-signal electrical S-parameters of the MZM. The applied
DC voltage for a reverse bias was set at −5 V. A 6 dB electrical BW
(S21) of over 67 GHz and an electrical reﬂection (S11) of less than
−10 dB were obtained. Furthermore, the measured 3 dB EO-BW
(1.5 GHz reference) was over 67 GHz as shown in Fig. 4. The results
indicated that the MZM simultaneously satisﬁed the lightwave-microwave
velocity and the characteristic impedance matching conditions.
0
–6
–12
–18
–24
RF response, dB
–30
0
10
20
S21
S11
30
frequency, GHz
40
50
Vbias: –5 V
60
70
Fig. 3 Small-signal RF responses
0
10
–3
EO response, dB
0
20
30
frequency, GHz
40
50
60
70
Fig. 4 Small-signal EO response
Finally, we demonstrated the dynamic characteristics. The 50 and
100 Gb/s non-return-to-zero (NRZ) signals with a pseudo-random
binary sequence of 231 −1 were ampliﬁed and fed into a single-RF elec-
trode by an RF probe. The operating wavelength, input optical power,
and driving voltage were 1550 nm, +13 dBm, and 2.3 Vpp, respectively.
Fig. 5 shows generated 50 and 100 Gb/s NRZ-OOK eye diagrams. Clear
eye openings and dynamic ERs of over 18 and 10 dB were obtained,
respectively. These results indicate that our MZM is superior to other
high-speed modulators such as an electro-absorption modulator in pro-
viding a high-ER and a low optical loss.
ELECTRONICS LETTERS
27th October 2016
Vol. 52
No. 22
pp. 1866–1867

<!-- page 2 -->
a
c
b
Fig. 5 Measured eye diagrams
a 50 Gb/s modulated signal
b 100 Gb/s electrical input signal
c 100 Gb/s modulated signal
Conclusion: We demonstrated that the combination of an n-i-p-n
heterostructure and a CL-TWE can greatly reduce the loss of an RF
electrode, which enhances the EO-BW of the MZM compared with a
conventional p-i-n heterostructure. A 100 Gb/s NRZ modulation with
a dynamic ER of over 10 dB was successfully demonstrated. We
believe our low electrical loss structure to be suitable for high capacity
transmission systems in various optical networks.
© The Institution of Engineering and Technology 2016
Submitted: 16 August 2016
E-ﬁrst: 30 September 2016
doi: 10.1049/el.2016.2987
One or more of the Figures in this Letter are available in colour online.
Y. Ogiso, J. Ozaki, N. Kashio, N. Kikuchi, H. Tanobe and M. Kohtoku
(NTT Device Innovation Center, NTT Corporation, 3-1 Morinosato
Wakamiya, Atsugi-shi, Kanagawa Pref. 243-0198, Japan)
✉E-mail: ogiso.yoshihiro@lab.ntt.co.jp
Y. Ohiso (NTT Device Technology Laboratories, NTT Corporation, 3-1
Morinosato Wakamiya, Atsugi-shi, Kanagawa Pref. 243-0198, Japan)
References
1
Klein, H.N., Chen, H., Hoffmann, D., Staroske, S., Steffan, A.G., and
Velthaus, K.-O.: ‘1.55 μm Mach-Zehnder modulators on InP for
optical 40/80 Gb/s transmission networks’. Indium Phosphide and
Related Materials Conf. (IPRM) 2006, TuA2.4, Princeton, NJ, USA,
May 2006
2
Wang, G., and Woods, I.: ‘Low Vπ, high bandwidth, small form factor
InP modulator’. Avionics, Fiber-Optics and Photonics Technology
Conf. (AVFOP) 2014, WB3, Atlanta, GA, USA, November 2014
3
Letal, G., Prosyk, K., Millett, R., et al.: ‘Low loss InP C-band IQ modu-
lator with 40 GHz bandwidth and 1.5 V Vπ’. Optical Fiber Communication
Conf. (OFC) 2015, Th4E.3, Los Angeles, CA, USA, March 2015
4
Poirier, M., Boudreau, M., Lin, Y., et al.: ‘InP integrated coherent
transmitter
for
100 Gb/s
DP-QPSK
transmission’.
Optical
Fiber
Communication Conf. (OFC) 2015, Th4F.1, Los Angeles, CA, USA,
March 2015
5
Rouvalis, E.: ‘Indium phosphide based IQ-modulators for coherent plug-
gable optical transceivers’. IEEE Compound Semiconductor Integrated
Circuit Symp. (CSICS) 2015, H2, New Orleans, LA, USA, October 2015
6
Ogiso, Y., Ohiso, Y., Shibata, Y., and Kohtoku, M.: ‘[011] waveguide
stripe direction n-i-p-n heterostructure InP optical modulator’, Electron.
Lett., 2014, 50, (9), pp. 688–690
ELECTRONICS LETTERS
27th October 2016
Vol. 52
No. 22
pp. 1866–1867
 1350911x, 2016, 22, Downloaded from https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/el.2016.2987, Wiley Online Library on [03/07/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

