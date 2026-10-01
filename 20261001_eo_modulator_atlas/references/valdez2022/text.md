---
paper_id: valdez2022
source_url: https://doi.org/10.1038/s41598-022-23403-6
doi: 10.1038/s41598-022-23403-6
license: CC-BY-4.0
sha256: 9a375d0ab7995bff5f0f49841522a2a8627c4b56d97af6c719e80fc2f2863d84
pages: 13
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
arXiv:2210.14785v1  [physics.optics]  26 Oct 2022
110 GHz, 110 mW Hybrid Silicon-Lithium Niobate
Mach-Zehnder Modulator
Forrest Valdez1,*, Viphretuo Mere1, Xiaoxi Wang1, Nicholas Boynton2, Thomas A.
Friedmann2, Shawn Arterburn2, Christina Dallo2, Andrew T. Pomerene2, Andrew L.
Starbuck2, Douglas C. Trotter2, Anthony L. Lentine2, and Shayan Mookherjea1,**
1University of California, San Diego, Department of Electrical and Computer Engineering, 9500 Gilman Drive, MC
0407, La Jolla, California, USA
2Sandia National Laboratories, Applied Microphotonic Systems, Albuquerque, New Mexico 87185, USA
*fgvaldez@eng.ucsd.edu
**smookherjea@ucsd.edu
ABSTRACT
High bandwidth, low voltage electro-optic modulators with high optical power handling capability are important for improving
the performance of analog optical communications and RF photonic links. Here we designed and fabricated a thin-ﬁlm lithium
niobate (LN) Mach-Zehnder modulator (MZM) which can handle high optical power of 110 mW, while having 3-dB bandwidth
greater than 110 GHz at 1550 nm. The design does not require etching of thin-ﬁlm LN, and uses hybrid optical modes formed
by bonding LN to planarized silicon photonic waveguide circuits. A high optical power handling capability in the MZM was
achieved by carefully tapering the underlying Si waveguide to reduce the impact of optically-generated carriers, while retaining
a high modulation efﬁciency. The MZM has a VπL product of 3.1 V.cm and an on-chip optical insertion loss of 1.8 dB.
Introduction
Electro-optic modulators (EOMs) are essential components for optical communications, radio frequency (RF)-photonic links,
frequency combs, optical phased arrays and optical information processing. In recent years, thin-ﬁlm lithium niobate (TFLN)
EOMs with a high electro-optic 3-dB bandwidth of at least 100 GHz have been reported with a low voltage-times-length
(VπL) product and low optical insertion loss1–4. However, these demonstrations have been performed at relatively low optical
power levels less than 10 mW. The ability to handle high optical powers without increasing Vπ would be beneﬁcial, by
increasing the RF link gain and reducing the noise ﬁgure of an analog communications system5–7. For a given photodiode
responsivity, reducing Vπ and increasing the optical power are both important in the noise ﬁgure of an RF photonic link.
Recently, a higher optical power level, around 25 mW, was shown using TFLN Mach-Zehnder modulator (MZM) with about
50 GHz bandwidth8. Here, we report the ﬁrst TFLN MZM which achieves both above-100 GHz bandwidth and above-100
mW optical power handling capability, while also achieving a low VπL product and low on-chip optical insertion loss, and
integration with silicon photonics.
TFLN-based waveguides have been made using various methods such as etching, blade-dicing or polishing of ridge or rib
waveguides, and rib-loading of a TFLN slab using other materials1,9–19. Our approach is based on hybrid optical modes, in
which the waveguide core consists of a combination of silicon (Si) and LN, and the cladding consists of silicon dioxide (SiO2).
This approach takes advantage of both the scalability of Si photonics, which is fabricated on large wafers using mature foundry
processing, and the EO properties of crystalline LN without having to etch or pattern it. The Si waveguides are formed in the
crystalline layer of a silicon-on-insulator (SOI) wafer, and are tapered in the hybrid region to implement inter-layer optical
mode transitions; the TFLN layer is not etched or patterned. The Si waveguides can be continuously connected, without
breaks, to other photonic components outside the bonded region, and thus present a simple and scalable way to include
ultrahigh bandwidth EOMs with other photonic components19,20.
At higher optical powers, Si waveguides suffer from two-photon absorption (TPA) and free carrier absorption (FCA) and
dispersion (FCD) effects21–24. The nonlinear optical absorption of amorphous Si is sensitive to processing25 and can even
exceed that of crystalline Si by a factor of 30 or more26. Therefore, it is preferable in hybrid TFLN design to use a layer
of crystalline Si waveguides rather than depositing Si on top of TFLN which has been a popular strategy27,28. Also, by
carefully designing the Si layer dimensions and tapers, the amount of optical power in the Si region can be minimized while
still remaining in the single-mode regime, which is necessary for a high-bandwidth MZM. While our strategy does not totally
eliminate the nonlinear loss of Si, it can keep the impairments of the hybrid mode adequately low for practically-useful optical

<!-- page 2 -->
power levels over the device lengths needed for a high-bandwidth, low-voltage MZM.
Figure 1. (a) Top-view schematic (not to scale) of the Si/LN MZM. The transition region consists of the transition
waveguide and adiabatic tapers (shown in green), and the phase shifter section consists of the hybrid waveguide (shown in
red). Cross-sectional simulated Poynting vector of the (b) feeder Si waveguide (no LN ﬁlm), (c) transition Si waveguide, and
(d) hybrid Si/LN waveguide. (e) The simulated effective index of the ﬁrst three Si/LN modes (top), and the conﬁnement
factor in the TFLN layer and Si waveguide (bottom) as a function of Si waveguide width at λ = 1550 nm. The dashed lines
indicate the modes shown in panels (c) and (d).
Results
Theory and Design
The phase shifter sections of the MZM, shown by the regions that are colored red in Fig. 1(a), consist of a long single-mode
waveguide in which the transverse localization of the hybrid optical mode is controlled by narrowed Si features (275 nm wide
and 150 nm tall). Adiabatically tapering down the Si waveguide reduces the effective index of the guided mode (neff), and
transitions the conﬁnement from the high index Si to the LN layer, as shown in Fig. 1(e). For this TE-polarized hybrid mode,
light exists mainly in the LN section, and as shown in Fig. 1(d), only 2% of the Poynting vector resides in Si, with 84%
in TFLN, and the rest in the cladding. Therefore, the nonlinear impairments can be expected to be much lower than for a
conventional Si carrier-depletion MZM where most of the light is in Si29. Because the amount of light in Si is so small, the
hybrid TFLN phase shifter does not have a signiﬁcant nonlinear optical penalty compared to a etched TFLN design.
The transition waveguide regions, shown by the regions that are colored green in Fig. 1(a), consist of wider Si features
(650 nm wide and 150 nm tall) which conﬁnes light mainly in Si (Fig. 1(c): 52% of the Poynting vector magnitude of the
fundamental waveguide mode resides in Si, only 20% in the thin ﬁlm of LN, and the rest in the cladding). Figure 1(b) and 1(c)
show the simulated mode proﬁle of the Si waveguides without and with the LN ﬁlm above, respectively. The effective index
(neff) difference between these modes is 0.08 which results in an estimated loss of only about 0.074 dB. Because the mode
fractions in Si (ΓSi) are similar, we do not expect this loss to signiﬁcantly change at higher powers when the refractive index
of Si changes slightly. Overall, this loss should be negligible compared to other losses in the device that are described below
in detail. This allows light to propagate with minimal loss (< 0.1 dB) across the bonded LN edge, where one side has LN
as part of the multi-layer stack and the other side does not. We can thus mitigate sensitivity to micro-roughness in the diced
edges of the LN chip, which simpliﬁes the fabrication process. The length of the transition waveguide segment is short: a few
tens of microns can be sufﬁcient to traverse the edge of the LN ﬁlm, and thus, the length-integrated effects of these nonlinear
impairments are low as shown using numerical simulations which are discussed below.
2/13

<!-- page 3 -->
Silicon waveguides in which the mode closely resembles the transition mode (but with no TFLN layer on top, as shown
in Fig. 1(b)) were used to form bends and the 3-dB multi-mode interference (MMI) couplers. These segments can also be
kept very short, and the length-integrated nonlinear impairments are small at the relevant optical power levels. Our test chip
also include tapered-waveguide edge couplers for guiding light from optical ﬁbers to the waveguides. In our experiments, it
was these segments, rather than the waveguides and transitions that constituted the MZM, were damaged at the highest optical
power levels (above 150 mW). A detailed study of high-power ﬁber-to-waveguide coupling is beyond the scope of this report.
Nonlinear impairments in a traveling-wave MZM device can, in principle, limit the intrinsic MZM EO bandwidth, insertion
loss (IL), and half-wave voltage (Vπ). For a given material dielectric stack, high speed EO modulation depends on three design
rules: 1) matching the velocities of the RF and optical waves; 2) matching the transmission line, source, and load impedances
(typically 50 Ω); and 3) minimizing the RF propagation losses30–33. The density of photo-generated electronic carriers caused
by TPA is
NTPA(P) = GTPAτe = βτe
2h f
 P
Aeff
2
,
(1)
where β is the TPA coefﬁcient of Si (assumed to be 1.0 cm/GW22), h f is the energy of the photon, P is the optical power, and
Aeff is the effective area of the optical mode. The electronic carrier lifetime τe in crystalline Si waveguides is typically in the
range of 1 to 100 ns, and depends on the waveguide geometry and the surface properties34. We estimate our 150 nm thick
strip waveguide to have an upper limit lifetime of 15 ns35. These photogenerated carriers lead to additional optical losses and
a shift in refractive index in the Si waveguide via the plasma dispersion effect36 given by,
αFC(P) = 1.45 × 10−17NTPA
(2a)
∆nFC(P) = −8.8 × 10−22NTPA −8.5 × 10−18NTPA0.8.
(2b)
If the change in the modal cross-sectional area Aeff is negligible, the evolution of the guided optical power as a function of
propagation distance (y) with linear and nonlinear losses is
dP(y)
dy
= −(αl + αFCΓSi)P−β
Aeff
ΓSiP2
(3)
where αl is the linear absorption loss of the waveguide and ΓSi =
ZZ
Si |Eo|2dA

·
ZZ +∞
−∞|Eo|2dA
−1
is the optical conﬁne-
ment factor in the Si waveguide in terms of the electric ﬁeld distribution of the optical mode, Eo(x,y).
Figure 2(a) shows the calculated change in IL due to nonlinear effects (TPA and FCA) by numerically solving Eq. (3) for
different lengths of hybrid waveguide (solid lines) and transition waveguides (dashed lines). We assume that the electronic
carriers generated by light are conﬁned to the Si waveguide, via ΓSi in Eq. (3). The cross-sectional modal areas Aeff of the
transition and hybrid modes are calculated to be 0.24 µm2 and 1.48 µm2, respectively. For the maximum considered lengths
of 2 cm and 0.15 cm for the hybrid and transition waveguides in Fig. 2(a), the 1 dB IL penalty due to TPA occurs at an optical
power ten times higher in the hybrid waveguide than in the transition waveguide. If the transition section is reduced to 100 µm
(which is approximately the minimum for an adiabatic taper between the modes shown in Fig. 1(c) and 1(d)37), then the total
additional IL should be less than 1 dB for optical powers up to 0.6 W.
The electro-optic response (EOR) as a function of RF frequency and optical power is30
m(ω,P) = RL + RG
RL

Zin
Zin + ZG


(ZL + Zc)F(u+(P))+ (ZL −Zc)F(u−(P))
(ZL + Zc)exp[γmLps]+ (ZL −Zc)exp[−γmLps]
,
(4a)
where the input impedance of the coplanar traveling wave electrodes is deﬁned as
Zin = Zc
ZL + Zctanh(γmLps)
Zc + ZLtanh(γmLps),
(4b)
and the complex RF propagation constant is
γm = αm + jω
c nm.
(4c)
3/13

<!-- page 4 -->
Figure 2. (a) The calculated insertion loss (IL) due to TPA as a function of peak optical power in the hybrid waveguide
mode (solid) and the transition mode (dashed) assuming different lengths of the phase-shift segment. (b) The change in
refractive index due to the additional free carriers generated via TPA as a function of peak optical power for the hybrid and
transition modes. (c) The calculated IL due to TPA as a function of peak optical power in the hybrid (solid) and transition
(dashed) mode assuming a length of 0.5 cm for different waveguide widths. (d) The calculated 3-dB bandwidth of a
traveling-wave MZM as a function of Lps assuming impedance matching to a 50 Ωload, and for different values of the index
mismatch, ∆nmo. Abbreviations: WG: waveguide, TFLN: thin-ﬁlm lithium niobate.
Here, RL,G are the load and generator resistances, ZL,G,c are the load, generator, and characteristic impedances, αm is the RF
propagation loss, nm is the RF effective index, and Lps is the phase-shifter interaction length. F [u±(P)] and u±(P) are deﬁned
as
F [u±(P)] = 1 −exp[u±(P)]
u±(P)
(4d)
and
u±(P) = ±αmLps + jω
c (±nm −ng(P))Lps.
(4e)
The optical power dependence in Eq. (4a) is included by the associated FCD term affecting the optical group index, ng, from
photogenerated carriers in Eq. (4e). There will be a similar shift in VπL due to FCD, where the effective modulator length, L,
may be limited by the additional nonlinear losses [Eq. (3)] and, thus, impact Vπ. Since the hybrid MZM uses an x-cut TFLN
ﬁlm and a TE-polarized fundamental waveguide mode, the effects of FCD on Vπ can be expressed as
VπLps(P) = neff(P)λ0G
2n4er33Γmo
(5)
4/13

<!-- page 5 -->
where neff is the hybrid mode effective index, λ0 is the free-space wavelength, G is the signal-to-ground electrode gap spacing,
ne is the extraordinary index of TFLN, r33 is the electro-optic coefﬁcient of TFLN, and Γmo is the overlap integral between
the RF and optical modes in the electro-optic region.
Figure 2(b) shows the change in refractive index. Changes in the refractive index will effect the index matching condition
(∆n = (1−nm/ng)×100%) between the traveling RF and optical waves. Since the hybrid mode is predominantly localized in
the LN ﬁlm, the effect is expected to be negligible for optical peak powers less than 10 W in theory. Although high intensity
optical inputs can also result in an increase in local temperature (and an associated thermo-optic effect), the time scales of
such thermal shifts is much longer compared to the modulation speeds38 and could be compensated by the bias controller.
Furthermore, the photorefractive effect of TFLN at a wavelength of 1550 nm results in refractive index changes on the order
of 10−5,39 which would only change ∆n by 0.005%.
Assuming the impedance matching condition is met, the 3-dB bandwidth of a traveling wave hybrid TFLN MZM is plotted
as a function of the length of the phase shifter segment for different values of ∆n in Fig. 2(d) using Eq. (4a). For Lps = 0.5
cm, to reduce the bandwidth by a factor of two (i.e., 3 dB), the ∆n caused by nonlinear high power effects would need to
be about 10%, corresponding theoretically to a peak power greater than 100 W, which is much greater than we are able to
test experimentally. A MZM device using a longer phase shifter of length 1 cm, would decrease the bandwidth by 3 dB for
∆n = 5%.
In this work, the total length of the MZM is about 1.7 cm, which includes the edge couplers, MMIs, transition region
(LTr = 0.14 cm), and the hybrid region (LHb = 1.4 cm). The 3-dB 2x2 MMI couplers are 38 µm long, 6 µm wide, and are
designed for the C-band. Bezier S-bends are implemented to separate the MZM arms to allow appropriate space for a standard
55 µm wide electrode. In the hybrid bonded region, the transition waveguides are adiabatically tapered down from a width of
650 nm to a width of 275 nm over a length of 300 µm which pushes the optical mode up into TFLN. These transition sections
were intentionally made longer than necessary to simplify our bonding process. If the waveguide had 300 nm width (instead
of 275 nm), then ΓSi increases slightly from 2% to 3.3%, which will increase the number of photogenerated carriers in Si.
Figure 2(b) and 2(c) show the additional index change and IL as the Si width changes for a hybrid waveguide length of 0.5
cm. For an optical input power of 110 mW, the additional IL due to TPA increases from 0.002 dB to 0.006 dB as the hybrid
waveguide Si width increases from the target value of 275 nm to 300 nm.
Hybrid Si/LN Chip Fabrication
The hybrid Si/LN EOM was made by direct hydrophilic bonding without using an intermediate adhesive layer, such as ben-
zocyclobutene (BCB)40, whose performance and stability at high optical powers is uncertain. The Si waveguides were built
on a 200 mm diameter SOI wafer with 150 nm thick Si, 3 µm thick silicon dioxide (SiO2), and 725 um thick high resistivity
Si handle. The Si waveguide features were fabricated using deep-UV lithography followed by reactive-ion-etching. SiO2 was
deposited on top of the Si waveguide features and then the surface was planarized using a CMP process to achieve a bondable
surface, leaving a thin SiO2 layer (< 50 nm). The LNOI wafers were procured commercially from NanoLN, Jinan Jingzheng
Electronics Co., Ltd. The LNOI is an x-cut 600 nm thick LN ﬁlm with 2 µm thick buried SiO2 and 500 µm thick Si handle.
There is typically some variation of LN ﬁlm thickness and thin SiO2 layer of the SOI across the wafers. Variations in the
thickness of SiO2 layer between the bonded Si waveguide and LN ﬁlm will change the mode fraction (Γ) contained in each
region, and will also change the modulation efﬁciency17. However, based on the outcomes of the CMP process, this effect
should be relatively small. For less than 20 nm change in the SiO2 thickness, the corresponding change in ΓSi or ΓLN remains
within a few percent of the target values. For each reticle of the larger SOI wafer, the thickness of the layer stack was veriﬁed
by ellipsometry before bonding. Based on the ellipsometric measurements of both the LN wafer and the processed silicon
wafer after CMP, a die dicing plan is developed, and speciﬁc pairs of diced LN and Si chips are selected for bonding. In this
way, the yield of devices from a wafer is increased, and the EO properties of each hybrid modulator will be closer to optimal.
Figure 3(a) illustrates the direct bonding hybrid Si/LN modulator fabrication process. Before bonding, the surfaces of the
LNOI and SOI singulated-dies were cleaned using an RCA-1 process followed by plasma surface activation19. After plasma
surface activation, the dies were soaked in deionized water. After drying the dies, the bonding was performed by contacting
the two dies at room temperature. To improve the bond strength, the bonded sample was annealed using temperature cycles
up to 300 ◦C under an applied pressure of 9.8 N.cm−2. At this stage, as described in our earlier work19, the bonded samples
can be stored for processing at a later stage, if required. A plasma-enhanced-chemical-vapor-deposition (PECVD) was used
to deposit a few microns of thick SiO2 as the top cladding. Before the handle removal step, a polymer coating was applied
to protect bonded chips except on top of the LNOI Si handle. Next, the oxide on top of the LNOI Si handle was etched
using hydroﬂuoric acid. The LNOI Si handle was removed using a selective XeF2-based etching process and then the oxide
layer above LN was etched using HF. The polymer was cleaned using a solvent cleaning step, and a direct-laser writer was
used to deﬁne the traveling wave electrode patterns using a negative photoresist. Finally, titanium and gold of thicknesses 20
nm and 750 nm, respectively, was deposited followed by a lift-off process to complete the fabrication of the electrodes. All
5/13

<!-- page 6 -->
Figure 3. (a) Fabrication process ﬂow of the hybrid bonded Si/LN MZM. (b) Optical microscope image of hybrid bonded
Si/LN chip with gold SWE. (b) The fabricated hybrid bonded Si/LN MZM chip. (c) A stitched optical microscope image of
a 5 mm long SWE. (d) A scanning electron microscope image of the inductively loaded slot features.
steps from bonding of the TFLN chips to the ﬁnal electrode fabrication were performed either at room temperature or at a
modestly-elevated temperatures, not exceeding about 300 ◦C.
For this device, an inductively loaded slow-wave electrode (SWE) structure [Fig. 3(b)-(d)] was designed to achieve
velocity matching to the hybrid Si/LN optical mode. The gap between the signal and ground electrodes is 9 µm, and the signal
width is 55 µm. The inductive loading feature is a periodic slot that is 5 µm wide (w) and 4 µm deep (h) with a period of
25 µm, as shown in Fig. 3(d). These periodic features increase the effective index of the microwave mode and slow the RF
traveling wave to match the optical wave41–45. Capacitively-loaded (T-rail based) SWE structures have been demonstrated
in other reports45,46 to achieve RF-optical velocity matching when using lower index substrates such as quartz or selectively-
removed Si. In this case, the hybrid Si/LN group index is simulated to be 2.32 and the slow-wave RF index is 2.34 at 110
GHz, resulting in ∆n < 1% between the optical and RF waves before the nonlinear change of the index is taken into account.
For the fabricated structures with about 40 nm of CMP SiO2 thickness, we estimate the peak electric ﬁeld strength at 110 mW
optical power in the oxide region of Fig. 1(c) and Fig. 1(d) to be about 5.4 × 104 V.cm−1 and 1.3 × 104 V.cm−1, which is
6/13

<!-- page 7 -->
smaller than the estimated breakdown ﬁeld strength of the SiO2, about 5 × 106 V.cm−1.
High-Speed, High-Power Measurements
Figure 4. (a) The measured optical transmission of the asymmetric hybrid-bonded Si/LN MZM with no modulation signal
applied. The asymmetric path-length difference gives an FSR of 2.3 nm, and an average ER of 28 dB over 50 nm. (b) The
measured VπL of the hybrid MZM (Lps = 0.5 cm) as a function of driving trapezoidal signal frequency. (c) A schematic
block diagram of the quasi-CW high power electro-optic response measurement setup. Active RF multipliers were used for
generating f > 50 GHz.
The asymmetric MZM under test was measured using a tunable CW C-band laser source which resulted in a free spectral
range of 2.3 nm, a mean extinction ratio (ER) of 28 dB over 50 nm (maximum of 31 dB at λ = 1560 nm), and a ﬁber-to-ﬁber
IL of 12.2 dB as shown in Fig 4(a). The half-wave voltage of the Si/LN modulator with Lps = 0.5 cm was measured by
overdriving it with a trapezoidal signal and projecting the overshoots of the modulated signal19, resulting in an average VπL of
3.1 V.cm in the range of 0.1 to 10 MHz as shown in Fig. 4(b). Figure 4(c) shows a block diagram of the quasi-CW high power
modulation experiment. An external commercial LN modulator (labeled Pulse Carver in Fig. 4(c)) with 30 dB extinction ratio
was used to carve the CW laser source into quasi-CW pulses. The pulse carver was driven by a pulse pattern generator, setting
voltages with a pulse width of 1 µs and a period of 100 µs. This corresponds to a pulse length greater than 100 m, which
is much greater than the length (1.7 cm) of the chip. A polarization maintaining erbium-doped ﬁber ampliﬁer (PM-EDFA)
was used to amplify the quasi-CW pulses to the Si/LN MZM. Quasi-CW pulses were used to extract maximum gain from
the PM-EDFA by reducing the average power and gain saturation. Also, the lower average power reduces the likelihood of
damage to the lensed ﬁbers used in the experiments. The ampliﬁed pulses were characterized by monitoring the PM-EDFA
output using a PM 90-10 splitter, where a sampling oscilloscope (DCA) veriﬁed the pulse shape and peak power, while an
optical power monitor measured the average power of the pulses as a function of the PM-EDFA current. Optical attenuators
were placed before the DCA and optical spectrum analyzer (OSA) to avoid damage. This device was measured to have an edge
coupling loss of 5.2 dB when using lensed tapered ﬁbers with a nominal 2.5 µm mode ﬁeld diameter; therefore, the on-chip IL
of the MZM was 1.8 dB (see Methods for more detail). These losses, as well as the ﬁber, splitter, and connector losses were
factored in to calculate the peak power out of the PM-EDFA. The power levels labeled in Fig. 5 and in this discussion refer to
the on-chip optical power levels in the feeder waveguide before the start of the MZM section. A peak power level of 110 mW
corresponds to an intensity of 460 kW/mm2 and 74 kW/mm2 in the transition and hybrid waveguides, respectively.
The high-power quasi-CW pulses were then coupled to the device and modulated with varying frequency sinusoidal
waves using an RF signal generator. A high-resolution OSA was used to detect the generated sidebands and carrier signal as
the frequency was swept47. This high-speed modulation measurement was performed in three frequency bands: 1 to 50 GHz,
47 to 78 GHz, and 72 to 110 GHz, where active RF multipliers were used to reach the latter two bands. A CS-5 calibration
substrate (GGB Industries, Inc.) and a 110 GHz RF power sensor were used to de-embed the cables and probes used in the
experiment and determine the RF power delivered. These calibration measurements were then used to extract the measured
7/13

<!-- page 8 -->
Figure 5. The OSA-measured (blue points) EOR of the hybrid bonded Si/LN MZM with Lps = 0.5 cm using no RF
multiplier (circles), a 4x multiplier (diamonds), and a 6x multiplier (squares) resulting in a measured 3-dB bandwidth of 110
GHz for both: (a) Low power, CW optical input with 4 mW of power; and (b) Quasi-CW optical input with peak power of
110 mW. The solid red line is the calculated linear model using Eq. (4a) with no nonlinearities. The cyan line in panel (a) is
the EOR of an identically designed Si/LN MZM using a high speed LCA.
EORs of the modulator at relatively low power CW optical input (4 mW), and at input peak powers of 110 mW, as shown in Fig.
5(a) and 5(b), respectively. In both cases, the measured EOR has a 3-dB bandwidth of 110 GHz. The overlapping measured
data between the bands was averaged and shifted accordingly to stitch the frequency responses together. Furthermore, the
EO S21 of an identically designed hybrid bonded MZM on a different chip was independently veriﬁed at low CW optical
power using a lightwave component analyzer (LCA). The LCA consisted of a Keysight PNA-X network analyzer, frequency
extenders, and a 110 GHz photodetector which allowed us to measure modulated signals ranging from 100 MHz to 110 GHz.
As can be seen in Fig. 5(a), the OSA-measured response (blue marks) and LCA-measured response (cyan line) agree in both
slope and 3-dB bandwidth. The 3-dB bandwidth is limited from impedance matching. Due to the ﬁxed separation between the
Si waveguides of the fabricated SOI chip, the signal electrode to electrode gap spacing is also ﬁxed. This speciﬁc MZM was
fabricated with an electrode gap of 9 µm, resulting in a transmission line impedance of 42 Ω. If the Si waveguide separation
is changed to accommodate a more optimal signal electrode width, then the impedance will be better matched to a 50 Ωload,
and further increase the 3-dB bandwidth of the MZM.
Discussion
The red lines in Fig. 5(a) and 5(b) are the calculated EORs using a linear model of the response function of a traveling
wave MZM with no intensity dependent nonlinear effects30. The theoretical EOR predicts a 3-dB bandwidth above 110 GHz
(approximately 156 GHz) for this 0.5 cm long modulator, which is agreement with the measured frequency responses in both
the low and high optical power regimes. A large concentration of free carriers overlapping with the optical mode would cause
a free-carrier dispersion effect which would decrease the 3-dB bandwidth. However, since the hybrid mode has only 2% of
light in the Si waveguide, there is an insigniﬁcant impairment in our device, and the response at 110 mW is effectively the
same as that at low (4 mW) power levels. Table 1 lists performance metrics of recent TFLN-based MZM. The present device
is the ﬁrst (to our knowledge) TFLN MZM that demonstrates greater than 100 GHz modulation of high optical power inputs
greater than 100 mW. FOM (units: dB) in Table 1 is deﬁned as FOM = 20 log10(Vπ)−20 log10(rd × Popt × 50 Ω), where rd
is the detector responsivity, assumed to be 1 A/W. FOM is that portion of the noise ﬁgure (NF) expression for an RF photonic
link6 which represents the contribution to the NF from the modulator Vπ and optical power level. In addition, the laser RIN
plays an important role in the overall NF in a practical system, but is not considered here in the FOM for the modulator alone.
8/13

<!-- page 9 -->
Lower values of the modulator FOM are preferable because they reduce the minimum achievable NF of the link. This shows
the beneﬁts of increasing Popt while keeping Vπ low. The on-chip IL of our hybrid Si/LN device is comparable to that of other
TFLN devices. The VπL can be slightly improved by introducing a buffer SiO2 layer between the electrodes and LN ﬁlm,
which will reduce the optical loss due to electrode interaction and allow for a narrower electrode spacing48. In comparison
to these TFLN MZM devices, a traditional Si depletion-mode MZM made with 250 nm thick Si rib waveguide (90 nm slab
height and 650 nm width) will have 75% of the Poynting energy conﬁned in Si and an Aeff on the order of 0.2 µm2. Both the
linear and nonlinear optical propagation loss coefﬁcients are high, and will limit the high-power handling capabilities of a Si
MZM. Furthermore, the modulation speed of carrier depletion MZMs is typically RC-limited, with reported 3-dB electro-optic
bandwidths < 60 GHz49–53.
Table 1. Comparison of recent TFLN-based MZM Performance Metrics
Device
3-dB BW (GHz)
VπL (V.cm)
IL (dB)
Lps (cm)
Popt (mW)
FOM(dB)
Etched LNOI1
100
2.2
0.5
0.5
1
38.9
Etched LNOI3
> 110
2.4
18 (ﬁber-to-ﬁber)
0.5
1
39.6
Etched LNOI8
50
2.2
NR
0.5
25
10.9
Etched LNOI48
>67
1.7
17 (ﬁber-to-ﬁber)
0.5
NR
-
Etched LNOI54
>67
2.2
0.2
1
0.03
63.3
Bonded (BCB-assisted) Si/LN46
>70
2.1
3
1.25
NR
-
Polished LNOI55
>50
2.2
0.6
0.7
NR
-
Bonded (BCB-assisted) Si + Etched LN15
>70
2.5
2.5
0.5
NR
-
Bonded (Direct) SiN/LN56
>50
6.7
13 (ﬁber-to-ﬁber)
0.5
4
36.5
Bonded (Direct) Si/LN - This Work
110
3.1
1.8
0.5
110
1.04
NR: Not Reported. IL (dB): on-chip Insertion Loss in dB; ﬁber-to-ﬁber Insertion Loss in dB is reported where noted. FOM(dB) is deﬁned as 20 log10(Vπ)−
20 log10(PoptrdRs), where rd = 1 A/W is the photodetector responsivity (assumed) and Rs = 50 Ωis the source resistance. Entries are left blank where Popt
was not reported.
Our experimental observations are consistent with the theoretical predictions shown in Fig. 2(b). Using Eq. (5), the effect
of TPA induced carriers on the VπL through FCD [Fig. 2(b)] is insigniﬁcant at 110 mW, and would contribute less than 1%
change in modulation efﬁciency assuming thermo-optic and photorefractive effects in the TFLN layer are negligible. The
change in IL due to carrier effects in the Si regions are predicted to be under 1 dB as shown by Fig. 2(a). Practically, such
small changes are indistinguishable from other losses in this study, such as repeatability of ﬁber coupling in these bare-die
test chips. Fiber-pigtailing the device will allow for easier testing. Although these tests were performed using long quasi-CW
pulses, we expect similar performance, when packaged, under CW optical excitation as well.
As we scale to longer phase shifters (Lps > 1 cm) to reduce the voltage, the walk-off between the RF and optical phases can
occur from smaller velocity differences. Therefore, the changes caused by higher optical power can have a larger effect on the
3-dB bandwidth. However, our calculations show that the majority of the impairments will occur in the mode transition region
rather than the phase-shifter region. The theoretical minimum length for a low-loss adiabatic taper between the transition mode
and hybrid mode in this design is about 100 µm37. In this batch of test chips, the transition waveguides were intentionally
made longer for fabrication ease, and therefore the nonlinear losses will inevitably be higher. The feeder waveguide and taper
lengths could be reduced, which in turn will minimize the TPA-induced FCA of the transition section. If the combined input
and output transition sections were only 100 µm long instead of 1.4 mm long as in these chips, then the additional insertion
loss due to TPA would incur a 1 dB penalty only for optical powers greater than 0.6 W. For higher power operation, other
materials with lower multiphoton absorption effects can be used. As an example, silicon nitride is another CMOS-compatible
material, does not experience TPA in the telecom wavelength regime, and has been shown to be suitable for hybrid bonding
with unpatterned LNOI56.
In summary, we have demonstrated that hybrid TFLN MZMs, which use an unetched thin-ﬁlm of LN bonded to planarized
crystalline Si waveguides, can withstand high optical input powers of 110 mW while maintaining high speed modulation
bandwidths of 110 GHz. A high EO bandwidth was achieved by using an inductively loaded SWE design to achieve velocity
matching between the RF and optical waves. The hybrid Si/LN device design was optimized to achieve a high modulation
efﬁciency (VπL = 3.1 V.cm) while minimizing the nonlinear intensity dependent effects of Si. The on-chip IL of the hybrid
9/13

<!-- page 10 -->
MZM structure including the MMI couplers, transitions into and out of the bonded region and electrode interaction loss was
measured to be 1.8 dB. High optical power operation was tested by amplifying quasi-CW pulses with a pulse width of 1
µs and at power levels of up to 110 mW, we did not observe degradation of the EO bandwidth compared to measurements
made at a conventional power level (4 mW). Compared to measurements made at an input power of 6 dBm, the additional
nonlinear insertion loss was less than 1 dB at 20.4 dBm optical power. Our work demonstrates the integration of Mach-Zehnder
modulators with high bandwidth and power handling capabilities well beyond the capabilities of traditional Si photonics.
Methods
Insertion Loss Characterization
Figure 6. A (not to scale) schematic of the hybrid bonded Si/LN MZM showing the total measured insertion loss and the
loss attributed to the edge coupler (EC), transition, hybrid, and phase shifter sections.
To determine the low power (1 mW) IL of the hybrid bonded Si/LN MZM, a CW tunable laser was coupled to the device
using lensed tapered ﬁbers (2.5 µm nominal mode ﬁeld diameter). As shown in Fig. 6, the hybrid bonded Si/LN MZM
consists of edge couplers (ECs), MMIs, wide Si transition sections and a narrow Si hybrid section. The propagation loss of
the transition waveguide section is 0.8 dB/cm, measured using the cutback method 57; whereas the propagation loss of the
hybrid section was reported in an earlier work as 0.6 dB/cm58. To determine the IL of the 3-dB 2x2 MMI splitters, adjacent
test MMIs of the same design was measured, resulting in a loss of 0.2 dB per splitter. The 1 dB bandwidth of the MMI is
over 50 nm with an imbalance between the output ports less than 0.3 dB. The IL of the phase shifter section was measured by
subtracting the transmission spectra of the test device and an identical reference device, but without electrodes. Therefore, the
IL of the Lps = 5 mm long phase shifter is 0.4 dB. From these measurements, the individual component losses of the MZM
were subtracted from the total transmission, resulting in an edge coupling loss of 5.2 dB/facet and an on-chip loss of 1.8 dB
(assuming 0.1 dB of loss per LN edge transition) as shown in Fig. 6.
Data availability.
Data underlying the results presented in this paper are not publicly available at this time but may be obtained from the
corresponding author upon reasonable request.
References
1. Wang, C. et al. Integrated lithium niobate electro-optic modulators operating at CMOS-compatible voltages. Nature 562,
101–104 (2018).
10/13

<!-- page 11 -->
2. Wang, X., Weigel, P. O., Zhao, J., Ruesing, M. & Mookherjea, S. Achieving beyond-100-GHz large-signal modulation
bandwidth in hybrid silicon photonics Mach Zehnder modulators using thin ﬁlm lithium niobate. APL Photonics 4 (2019).
3. Yang, F. et al. Monolithic thin ﬁlm lithium niobate electro-optic modulator with over 110 GHz bandwidth. Chin. Opt.
Lett. 20, 022502 (2022).
4. Xu, M. et al. Dual-polarization thin-ﬁlm lithium niobate in-phase quadrature modulators for terabit-per-second transmis-
sion. Optica 9, 61–62 (2022).
5. Ackerman, E. et al. Low noise ﬁgure, wide bandwidth analog optical link. Int. Top. Meet. on Microw. Photonics, MWP
2005 2005, 325–328 (2005).
6. Cox, C. H., Ackerman, E. I., Betts, G. E. & Prince, J. L. Limits on the performance of RF-over-ﬁber links and their
impact on device design. IEEE Trans. Microw. Theory Tech. 54, 906–920 (2006).
7. Williamson, R. C. & Esman, R. D. RF photonics. J. Light. Technol. 26, 1145–1153 (2008).
8. Shams-Ansari, A. et al. Electrically pumped laser transmitter integrated on thin-ﬁlm lithium niobate. Optica 9, 408–411
(2022).
9. Rabiei, P. & Steier, W. H. Lithium niobate ridge waveguides and modulators fabricated using smart guide. Appl. Phys.
Lett. 86, 161115 (2005).
10. Hu, H., Ricken, R., Sohler, W. & Wehrspohn, R. Lithium niobate ridge waveguides fabricated by wet etching. IEEE
Photon. Technol. Lett. 19, 417–419 (2007).
11. Zhao, J. et al. Shallow-etched thin-ﬁlm lithium niobate waveguides for highly-efﬁcient second-harmonic generation. Opt.
Express 28, 19669–19682 (2020).
12. Courjal, N. et al. High aspect ratio lithium niobate ridge waveguides fabricated by optical grade dicing. J. Phys. D: Appl.
Phys. 44, 305101 (2011).
13. Wu, R. et al. Long low-loss-litium niobate on insulator waveguides with sub-nanometersurface roughness. Nanomaterials
8, 910 (2018).
14. Liang, Y. et al. Monolithically integrated electro-optic modulator fabricated on lithium niobate on insulator by pho-
tolithography assisted chemo-mechanical etching. J.Phys. Photonics 3, 034019 (2021).
15. He, M. et al. High-performance hybrid silicon and lithium niobate Mach-Zehnder modulators for 100 Gbit s−1 and
beyond. Nat. Photonics 13, 359–364 (2019).
16. Rao, A. et al. Heterogeneous microring and mach-zehnder modulators based on lithium niobate and chalcogenide glasses
on silicon. Opt. Express 23, 22746–22752 (2015).
17. Weigel, P. O. et al. Bonded thin ﬁlm lithium niobate modulator on a silicon photonics platform exceeding 100 GHz 3-dB
electrical modulation bandwidth. Opt. Express 26, 23728 (2018).
18. Ahmed, A. N. R. et al. Subvolt electro-optical modulator on thin-ﬁlm lithium niobate and silicon nitride hybrid platform.
Opt. Lett. 45, 1112 (2020).
19. Mere, V., Valdez, F., Wang, X. & Mookherjea, S. A modular fabrication process for thin-ﬁlm lithium niobate modulators
with silicon photonics. J.Phys. Photonics 4, 024001 (2022).
20. Weigel, P. O. et al. Lightwave Circuits in Lithium Niobate through Hybrid Waveguides with Silicon Photonics. Sci. Rep.
6, 1–9 (2016).
21. Tsang, H. K. et al. Optical dispersion, two-photon absorption and self-phase modulation in silicon waveguides at 1.5 µm
wavelength. Appl. Phys. Lett. 80, 416–418 (2002).
22. Bristow, A. D., Rotenberg, N. & Van Driel, H. M. Two-photon absorption and kerr coefﬁcients of silicon for 850–2200
nm. Appl. Phys. Lett. 90, 191104 (2007).
23. Lin, Q., Painter, O. J. & Agrawal, G. P. Nonlinear optical phenomena in silicon waveguides: modeling and applications.
Opt. Express 15, 16604–16644 (2007).
24. Leuthold, J., Koos, C. & Freude, W. Nonlinear silicon photonics. Nat. Photonics 4, 535–544 (2010).
25. Wathen, J. J. et al. Non-instantaneous optical nonlinearity of an a-Si:H nanowire waveguide. Opt. Express 22, 22730–
22742 (2014).
26. Ikeda, K., Shen, Y. & Fainman, Y. Enhanced optical nonlinearity in amorphous silicon and its application to waveguide
devices. Opt. Express 15, 17761–17771 (2007).
11/13

<!-- page 12 -->
27. Cao, L., Aboketaf, A., Wang, Z. & Preble, S. Hybrid amorphous silicon (a-Si:H)–LiNbO3 electro-optic modulator. Opt.
Commun. 330, 40–44 (2014).
28. Wang, Y. et al. Amorphous silicon-lithium niobate thin ﬁlm strip-loaded waveguides. Opt. Mater. Express 7, 4018–4028
(2017).
29. Witzens, J. High-speed silicon photonics modulators. Proc. IEEE 106, 2158–2182 (2018).
30. Ghione, G. Semiconductor devices for high-speed optoelectronics, vol. 116 (Cambridge University Press Cambridge,
2009).
31. Noguchi, K., Mitomi, O., Miyazawa, H. & Seki, S. A broadband Ti:LiNbO3 optical modulator with a ridge structure. J.
Light. Technol. 13, 1164–1168 (1995).
32. Cox, C. Techniques and performance of intensity-modulation direct-detection analog optical links. IEEE Trans. Microw.
Theory Tech. 45, 1375–1383 (1997).
33. Honardoost, A., Saﬁan, R., Rao, A. & Fathpour, S. High-speed modeling of ultracompact electrooptic modulators. J.
Light. Technol. 36, 5893–5902 (2018).
34. Kuwayama, T., Ichimura, M. & Arai, E. Interface recombination velocity of silicon-on-insulator wafers measured by
microwave reﬂectance photoconductivity decay method with electric ﬁeld. Appl. Phys. Lett. 83, 928–930 (2003).
35. Claps, R., Raghunathan, V., Dimitropoulos, D. & Jalali, B. Inﬂuence of nonlinear absorption on Raman ampliﬁcation in
Silicon waveguides. Opt. Express 12, 2774–2780 (2004).
36. Soref, R. & Bennett, B. Electrooptical effects in silicon. IEEE J. Quantum Electron. 23, 123–129 (1987).
37. Fu, Y., Ye, T., Tang, W. & Chu, T. Efﬁcient adiabatic silicon-on-insulator waveguide taper. Photonics Res. 2, A41–A44
(2014).
38. Ruske, J.-P., Zeitner, B., Tunnermann, A. & Rasch, A.
Photorefractive effect and high power transmission in
LiNbO3channel waveguides. Electron. Lett. 39, 1048–1050 (2003).
39. Jiang, H. et al. Fast response of photorefraction in lithium niobate microresonators. Opt. Lett. 42, 3267–3270 (2017).
40. Poberaj, G. et al. Ion-sliced lithium niobate thin ﬁlms for active photonic devices. Opt. Mater. 31, 1054–1058 (2009).
41. Spickermann, R. & Dagli, N. Millimetre wave coplanar slow wave structure on GaAs suitable for use in electro-optic
modulators. Electron. Lett. 29, 774–775 (1993).
42. Sakashita, Y. & Segawa, H. Preparation and characterization of LiNbO3 thin ﬁlms produced by chemical-vapordeposition.
J. Appl. Phys. 77, 5995–5999 (1995).
43. Shin, J., Sakamoto, S. R. & Dagli, N. Conductor loss of capacitively loaded slow wave electrodes for high-speed photonic
devices. J. Light. Technol. 29, 48–52 (2010).
44. Rosa, Á., Verstuyft, S., Brimont, A., Thourhout, D. V. & Sanchis, P. Microwave index engineering for slow-wave coplanar
waveguides. Sci. Rep. 8, 1–8 (2018).
45. Kharel, P., Reimer, C., Luke, K., He, L. & Zhang, M. Breaking voltage–bandwidth limits in integrated lithium niobate
modulators using micro-structured electrodes. Optica 8, 357–363 (2021).
46. Wang, Z. et al. Silicon–lithium niobate hybrid intensity and coherent modulators using a periodic capacitively loaded
traveling-wave electrode. ACS Photonics 9, 2668–2675 (2022).
47. Shi, Y., Yan, L., Willner, A. E. & Member, S. High-Speed Electrooptic Modulator Characterization. J. Light. Technol. 21,
2358–2367 (2003).
48. Liu, X. et al. Wideband thin-ﬁlm lithium niobate modulator with low half-wave-voltage length product. Chin. Opt. Lett.
19, 060016 (2021).
49. Watts, M. R., Zortman, W. A., Trotter, D. C., Young, R. W. & Lentine, A. L. Low-voltage, compact, depletion-mode,
silicon mach–zehnder modulator. IEEE J. Sel. Top. Quantum Electron. 16, 159–164 (2010).
50. Dong, P., Chen, L. & Chen, Y.-k. High-speed low-voltage single-drive push-pull silicon mach-zehnder modulators. Opt.
Express 20, 6163–6169 (2012).
51. DeRose, C. T., Trotter, D. C., Zortman, W. A. & Watts, M. R. High speed travelling wave carrier depletion silicon
mach-zehnder modulator. In 2012 Optical Interconnects Conference, 135–136 (IEEE, 2012).
52. Xiao, X. et al. High-speed, low-loss silicon mach–zehnder modulators with doping optimization. Opt. Express 21,
4116–4125 (2013).
12/13

<!-- page 13 -->
53. Li, M., Wang, L., Li, X., Xiao, X. & Yu, S. Silicon intensity mach–zehndermodulator for single lane 100 gb/s applications.
Photonics Res. 6, 109–116 (2018).
54. Chen, G. et al. High performance thin-ﬁlm lithium niobate modulator on a silicon substrate using periodic capacitively
loaded traveling-wave electrode. APL Photonics 7, 026103 (2022).
55. Wu, R. et al. High-production-ratefabrication of low-loss lithium niobate electro-optic modulators using photolithography
assisted chemo-mechanical etching (place). Micromachines 13, 378 (2022).
56. Boynton, N. et al. A heterogeneously integrated silicon photonic/lithium niobate travelling wave electro-optic modulator.
Opt. Express 28, 1868–1884 (2020).
57. Wang, X., Valdez, F., Mere, V. & Mookherjea, S. Integrated thin-silicon passive components for hybrid silicon-lithium
niobate photonics. Opt. Continuum 1, 2233–2244 (2022).
58. Weigel, P. O. High-speed hybrid silicon-lithium niobate electro-optic modulators & related technologies (University of
California, San Diego, 2018).
Additional Information
Competing Interests
The authors declare no competing interests.
13/13

