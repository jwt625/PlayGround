---
paper_id: renaud2023
source_url: https://doi.org/10.1038/s41467-023-36870-w
doi: 10.1038/s41467-023-36870-w
license: https://creativecommons.org/licenses/by/4.0
sha256: bbce59f987fb9b62479aad7fcd36f3c34383fd7aee5258b3c3da01a0039b0758
pages: 7
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Article
https://doi.org/10.1038/s41467-023-36870-w
Sub-1 Volt and high-bandwidth visible to
near-infrared electro-optic modulators
Dylan Renaud
1,5
, Daniel Rimoli Assumpcao1,5, Graham Joe1,
Amirhassan Shams-Ansari
1, Di Zhu
1,2, Yaowen Hu1,3, Neil Sinclair1,4 &
Marko Loncar
1
Integrated electro-optic (EO) modulators are fundamental photonics compo-
nents with utility in domains ranging from digital communications to quantum
information processing. At telecommunication wavelengths, thin-ﬁlm lithium
niobate modulators exhibit state-of-the-art performance in voltage-length
product (VπL), optical loss, and EO bandwidth. However, applications in optical
imaging, optogenetics, and quantum science generally require devices oper-
ating in the visible-to-near-infrared (VNIR) wavelength range. Here, we realize
VNIR amplitude and phase modulators featuring VπL’s of sub-1 V ⋅cm, low
optical loss, and high bandwidth EO response. Our Mach-Zehnder modulators
exhibit a VπL as low as 0.55 V ⋅cm at 738 nm, on-chip optical loss of ~0.7 dB/cm,
and EO bandwidths in excess of 35 GHz. Furthermore, we highlight the
opportunities these high-performance modulators offer by demonstrating
integrated EO frequency combs operating at VNIR wavelengths, with over 50
lines and tunable spacing, and frequency shifting of pulsed light beyond its
intrinsic bandwidth (up to 7x Fourier limit) by an EO shearing method.
Integrated photonics at visible and near-infrared (VNIR) wavelengths is
important for applications ranging from sensing1–3 and spectroscopy4 to
communications5 and quantum information processing6,7. For example,
visible integrated photonic platforms can be combined with any of the
large variety of atomic or atomic-like systems with transitions in the
VNIR such as alkali and alkaline-earth metal atoms8–10, rare-earth ions11,
diamond color centers12,13 and quantum dots14–16. Concerning quantum
applications, VNIR photonics enables photon routing17,18, spectral
shifting for interfacing disparate quantum emitters11,19, or realizing
higher-dimensional encoded quantum states20,21, all in a scalable and
compact approach.
A variety of visible integrated photonic platforms have been
demonstrated, including silicon nitride2,22–25, aluminum ntiride26,27,
diamond28,29, and lithium niobate (LN)30–32. LN is particularly compel-
ling due to its large electro-optic (EO) coefﬁcient, low optical loss, and
wide transparency window, making it the workhorse material for the
modern day telecommunications industry. Recent work has shown the
promise of thin-ﬁlm lithium niobate (TFLN) at telecommunications
wavelengths33. Beyond the inherent miniaturization and integratability
achievable with TFLN, the strong optical conﬁnement and increased
tailorability have enabled performance not achievable with bulk LN,
including CMOS compatible drive voltages and high bandwidth
operation34–36. As a result of LN’s large transparency window, EO TFLN
devices in the visible regime have been demonstrated30–32. However,
half-wave voltages (Vπ) and large bandwidths beyond that realized in
visible bulk devices has yet to be demonstrated in VNIR TFLN. In par-
ticular, the combination of high-bandwidth and low drive-voltage
optical modulation would enable on-chip routing and spectral control:
a critical requirement for quantum applications.
In this work, we realize VNIR TFLN amplitude and phase mod-
ulators (Fig. 1a) operating with VπL of sub-1 V ⋅cm (Fig. 1b), extinction
ratios beyond 20 dB, and 3 dB EO bandwidths in excess of 35 GHz.
Received: 5 September 2022
Accepted: 21 February 2023
Check for updates
1John A. Paulson School of Engineering and Applied Sciences, Harvard University, Cambridge 02139 MA, USA. 2Institute of Materials Research and Engi-
neering, Agency for Science, Technology and Research (A*STAR), Singapore 138634, Singapore. 3Department of Physics, Harvard University, Cambridge
02139 MA, USA. 4Division of Physics, Mathematics and Astronomy, and Alliance for Quantum Technologies (AQT), California Instituteof Technology, Pasadena
91125 MA, USA. 5These authors contributed equally: Dylan Renaud, Daniel Rimoli Assumpcao.
e-mail: renaud@g.harvard.edu; loncar@g.harvard.edu
Nature Communications|  (2023) 14:1496 
1
1234567890():,;
1234567890():,;

<!-- page 2 -->
We perform two demonstrations to highlight applications of these
devices. We demonstrate an integrated and tunable EO frequency
comb source in the VNIR, showing over 50 lines in a single comb at638,
738, and 838 nm, and displaying ﬂat-top spectra with less than 10 dB
power variation. Furthermore, we use our devices to demonstrate
spectral shearing of optical pulses over 7 times their intrinsic spectral
bandwidth. Together these demonstrations highlight the widespread
utility of TFLN modulators operating in the VNIR spectrum.
Results
Design and fabrication
Figure 1c illustrates the design of our TFLN VNIR modulators. We
fabricate devices on 300 nm thick X-cut TFLN on 2 μm of thermally
grown silicon dioxide on Si (NanoLN). For complete device fabrication
details, see methods. Outside of the electrode region, the waveguides
are designed to be single mode (support transverse-electric, TE00,
and transverse-magnetic, TM00) at 740 nm. We choose this constraint
to minimize excitation of higher-order modes, which can lead to
a reduction in EO performance in the electrode region. Using ﬁnite-
difference eigenmode simulations (Lumerical), the required wave-
guide top width for single mode operation is determined to be
approximately 300 nm. A disadvantage of this width is that it reduces
mode conﬁnement. This leads to higher optical loss due to mode
overlap with sidewalls and cladding, and absorption loss from elec-
trodes. For this reason, we adiabatically increase the waveguide width
to 600 nm in the electrode region. Finally, our amplitude modulators
feature Y-splitters with excess losses of approximately 0.2 dB/
splitter30. An optical micrograph of a fabricated 5 mm long amplitude
modulator is provided in Fig. 1d.
For the electrodes, we employ a push-pull conﬁguration with co-
planar waveguide (CPW) travelling-wave electrodes. Finite element
method (COMSOL) simulations are used to design electrodes with
impedance close to 50 Ω and a simulated microwave phase index of
nRF = 2.22 at 50 GHz. Due to the relatively large optical group index at
VNIR wavelengths compared to telecommunication wavelengths
(nvis ≈2.38, ntel ≈2.25), perfect velocity matching requires a reduction
in bottom oxide (BOX) thickness and/or gold thickness, which comes
at the expense of increased optical and RF loss. To avoid this, our
devices have an index mismatch between the microwave phase and
optical group index of the TE00 mode of Δn~0.17. For an impedance
matched, 1 cm long lossless modulator, this index mismatch corre-
sponds to a theoretical bandwidth of ~80 GHz.
Visible-to-near-infrared Mach Zehnder modulators
We fabricate 1 cm long Mach-Zehnder modulators (MZMs) with vary-
ing gap sizes and experimentally evaluate their performance across
both wavelength and electrode gap parameter spaces. The experi-
mental setup is shown in the inset of Fig. 2a (see methods for mea-
surement details).
As shown in Fig. 2a, the Vπ of our 1 cm long, 3 μm gap devices is as
low as 0.42 V at 532 nm, and increases only slightly to 0.45, 0.55, and
0.85 V at 638, 738, and 838 nm, respectively. The increase in VπL for
longer wavelengths follows from the smaller phase accumulation
for the same modulator length. Our VπL is a factor of 2-3 smaller,
depending on wavelength considered, than the best previously
reported values for VNIR TFLN modulators, without compromising
bandwidth or device insertion loss30–32. We note that our improvement
stems predominantly from the reduction in electrode gap, i.e.,
enhancement in optical-microwave ﬁeld overlap.
We perform the same measurements at 738 nm, but with MZMs of
varied gap sizes, seeing an increasing VπL for larger gap sizes (Fig. 2b).
For comparison, we also theoretically calculate VπL as a function of
gap. The simulated response shows excellent agreement with our
measured results. Notably, we measure VπL < 1 V ⋅cm for gap sizes as
large as 5 μm. For comparison, recent work on VNIR devices with
smaller gaps (2 μm) have reported larger low frequency VπL31.
We extract the on-chip modulator loss by fabricating and mea-
suring 3 μm gap modulators of varying electrode length. From this we
obtain an on-chip loss of ~ 0.7 ± 0.2 dB/cm (see supplementary ﬁg. 1 for
details). This value is over an order of magnitude smaller than that
reported in other recent demonstrations for VNIR LN modulators32,37.
Including the lensed ﬁber-to-chip coupling loss ( ~ 7 dB/facet), the total
device insertion loss comes to ~ 15 dB. We note that because the total
device insertion loss is dominated by coupling loss (mismatch between
the lensed ﬁber and rib waveguide mode), it can be reduced by
nearly an order of magnitude using techniques such as tapered ﬁber
(b)
(a)
(c)
Normalized Transmission
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.6
0.8
0.4
1.0
Vπ = 0.55 V
Optical Input
Frequency Shifting
Frequency Comb
Frequency
Amplitude Modulation
Time
(d)
SiO2 (1 um)
TFLN
SiO2 (2 um)
300 nm
600 nm
180 nm
120 nm
800 nm
Au
Optical Output
Optical Input
< 1 V
< 1 V
Voltage (V)
Time
1 mm
gap
Contact 
 Region
Interaction 
   Region
Y-splitter
Waveguides
Fig. 1 | Ultra-low Vπ modulators operating at visible-to-near-infrared
wavelengths. a In the time domain, VNIR amplitude modulators with ultra-low
drive voltages ( < 1 V) can modulate continuous-wave optical inputs at CMOS vol-
tages. Similarly, sub-volt phase modulators enable VNIR frequency comb genera-
tion and frequency shifting over multiple pulse bandwidths. b Normalized optical
transmission of a 10 mm long amplitude (Mach-Zehnder) modulator as a function
of the applied voltage. At λ = 738 nm, the Vπ is 0.55 V at 1 MHz. c Cross section
illustrations of modulator waveguide and electrode regions. d Optical micrograph
of a 5 mm long VNIR TFLN amplitude modulator. The micrograph additionally
shows the unbalanced amplitude modulator waveguides, along with the Y-splitters,
probe contact region of the electrodes, and the electrode gap taper along the probe
contact region to the interaction region.
Article
https://doi.org/10.1038/s41467-023-36870-w
Nature Communications|  (2023) 14:1496 
2

<!-- page 3 -->
coupling38. In addition, we observe an extinction ratio of over 25 dB for
5 μm gap devices, and ~ 21 dB for 3 μm gap devices (Fig. 2b, inset),
comparable with state-of-the-art.
To evaluate the high frequency response of the devices, the 3 dB
EO bandwidth of our MZM is extracted via ﬁrst measuring the optical
frequency spectrum of the transmitted light on a Fabry-Perot (FP)
cavity or optical spectrum analyzer (OSA) and ﬁtting the sideband
powers to a Bessel function for varied frequency applied sinusoidal
voltage. Figure 2c shows the high frequency response for a 3 μm gap,
1 cm long MZM operating at 738 nm. The extracted 3 dB bandwidth is
approximately 35 GHz (w.r.t. to 3 GHz), and is limited by RF loss of
the CPW (1.35 dB cm−1 GHz−1/2). A non-DC reference is chosen due to
both the rapid roll-off originating from the CPW impedence mis-
match, and the commonly observed instability in LN modulators at
low frequencies due to photorefractive effects. We use the measured
electrical transmission coefﬁcients of the CPW to also theoretically
predict the modulator 3 dB bandwidth, which we ﬁnd to be ~36 GHz,
in excellent agreement with our measured result. We emphasize that
since the device bandwidth is limited by CPW RF loss, it can be
improved upon by implementing capacitively loaded traveling wave
electrodes to reduce current crowding and its associated increase of
RF loss39.
To fully contextualize the performance of our device, we compare
our results with other previously demonstrated VNIR modulators in
terms of bandwidth and low-frequency Vπ (BW/Vπ ratio), including
both integrated and non-integrated VNIR modulators (Fig. 2d). We
emphasize the improvement in performance between this work and
state-of-the-art commercial bulk VNIR LN modulators. While prior
works have demonstrated the superior performance of TFLN over bulk
LN in the telecommunication band40, to date, the same has not been
shown in TFLN VNIR modulators. Here, we show that VNIR TFLN
modulators can exhibit voltage-bandwidth performance exceeding
30 GHz V−1, a performance metric not achievable in bulk devices, or
other current integrated VNIR photonic platforms. Finally, we note
that while other visible modulator platforms have recently been
demonstrated41,42, they are presently limited to operating frequencies
in the 1–100 kHz range.
Visible-to-near-infrared electro-optic frequency comb
To demonstrate the utility of these EO devices for sensing applications,
we fabricate TFLN phase modulators (PM) operating at VNIR wave-
lengths. Because of the low required driving voltages and broadband
optical operation, we use these devices to generate EO frequency
combs operating at high-frequency and with variable comb spacing.
Our combs feature over 50 sidebands when driven at 30 GHz by a ~ 3 W
microwave source ( ~ 8Vπ), and they operate over a broad wavelength
range. Results for a 3 μm device operating at 638, 738, and 838 nm are
shown in Fig. 3a–c. The insets depicts the comb spacing between two
Fig. 2 | Ultra-low Vπ visible-to-near-infrared wavelength Mach-Zehnder mod-
ulators with greater than 35 GHz bandwidth. a Experimental setup illustration
and measured low-frequency (1 MHz) Vπ for a 1 cm MZM with an electrode gap of
3 μm. Data shown corresponds to Vπ at λ = 532, 638, 738, 838, and 938 nm. Simu-
lated Vπ is shown by the solid line. b Simulated and measured Vπ (1 MHz) at
λ = 738 nm for varied electrode gap. Inset shows a measured extinction ratio of
~21 dB for a 1 cm long, 3 μm gap modulator. c Frequency dependence of Vπ for a
1 cm long MZM. A 3 dB EO bandwidth of ~ 35 GHz is extracted from the response.
The bandwidth is measured with respect to a low frequency reference, here taken
to be 3 GHz. The grey dashed-lines denote the 3 dB bandwidth w.r.t 3 GHz.
d Comparison of modulator ﬁgure of merit BW/Vπ between this work, state-of-the-
art commercial LN modulators, previous VNIR thin-ﬁlm LN modulators, and other
VNIR modulator platforms. This work exhibits signiﬁcantly higher BW/Vπ values
than all previously reported works, including previously demonstrated TFLN VNIR
modulators30 --32,37. The dashed lines correspond to constant values of BW/Vπ. Note
that for fair comparison, we compare the reported Vπ at < 1 GHz for all devices. The
TFLN data points with cross annotations denote devices for which the reported BW
was limited by the equipment used. Detailed comparison of referenced works can
be found in supplementary tables 1 and 2. RF Radiofrequency, F.P.C. Fiber polar-
ization controller, TFLN Thin-ﬁlm lithium niobate, OSC Oscilloscope, OSA Optical
Spectrum Analyzer, FP Fabry-Perot.
Article
https://doi.org/10.1038/s41467-023-36870-w
Nature Communications|  (2023) 14:1496 
3

<!-- page 4 -->
lines at higher magniﬁcation. Due to the limited resolution of the OSA
at these wavelengths, the comb visibility is not fully resolved. We fur-
ther note that for shorter wavelengths, the comb envelope displays
greater asymmetry. This phenomenon originates from the fact that the
waveguides begin to support higher order modes at these wave-
lengths, each of which propagate at different group velocities and
possess varying velocity mismatch with respect to the RF ﬁeld. We
support this assessment by calculating the theoretical comb spectrum
after including additional modes which shows good qualitative
agreement with our measured results (see supplementary ﬁgure 2).
Finally, in Fig. 3d, the 1st sideband is shown as a function of applied RF
drive frequency, thereby illustrating the tunable nature of the VNIR
EO combs.
This is an important component that can be utilized for a variety
of applications including sensing43, astrophysical spectroscopy4, and
frequency-bin encoding of quantum information20.
Visible spectral shearing
Our low Vπ modulators enable high-bandwidth frequency control of
input light. Namely, by applying a quasi-linear phase ramp ϕ(t) = −Kt to
the modulator, input light can be shifted in frequency by K. This fre-
quency shift is referred to as spectral shearing44,45. This is useful for
quantum applications where frequency shifting can be used to bridge
the inhomogenous distribution of quantum emitters or for performing
frequency bin operations on non-classical states46. The former is par-
ticularly useful at visible wavelengths where a variety of quantum
emitters have their optical transitions.
The shift achievable via frequency shearing is dependent on the
total phase applicable to the device. This is proportional to
V
V π f RF,
where V is the applied voltage, and fRF is the frequency of the applied
RF tone. Thus a low Vπ phase modulator is required to achieve a large
total shift. Although previous demonstrations of shearing have
demonstrated large frequency shifts up to 640 GHz47, these demon-
strations relied on ultra-short pulses to utilize a high frequency RF
drive45,47,48, and thus shifting beyond the bandwidth of the pulse has
not been demonstrated.
We use a 100 MHz RF tone with an amplitude of ~ 20Vπ applied to
our device and lock 1 ns duration square-shaped optical pulses
(λ = 737 nm) to the rising or falling linear regime of the RF tone to apply
a quasi-linear phase proﬁle to the pulse (Fig. 4a, b). Our TFLN device is
a 1 cm long, 3 μm gap phase modulator with a Vπ ~ 1 V at 100 MHz. We
observe a spectral shift of ± 6.6 GHz when locking the pulse to the
rising or falling edge, respectively (Fig. 4c). Approximately 85% of
input power is shifted into the desired lobe, with our estimate limited
by the ﬁnite extinction of the input optical pulses ( ~ 20 dB), which can
be improved via gating the detected signal.
Transmission (dB)
0
-5
-10
-15
-20
-25
-30
Wavelength (nm)
0
-10
-20
-30
-40
-50
0
-10
-20
-30
-40
-50
637
638
639
737
738
739
837
838
839
(a)
(b)
(c)
Normalized Intensity
Frequency (GHz)
14 GHz
20 GHz
30 GHz
40 GHz
10
15
20
25
30
35
40
45
0.0
0.2
0.4
0.6
0.8
1.0
(d)
-2
-8
-4
-6
-18
-12
-4
-20
-12
837.86
838.00
738.01
738.12
638.01
638.09
Fig. 3 | Integrated visible-to-near-infrared electro-optic frequency combs.
Tunable frequency combs operating at a, λ = 638, b 738 nm, and c 838 nm with
30 GHz spacing and more than 50 lines. Insets show magniﬁed view of the comb
lines. The asymmetry shown in the 638 nm and 738 nm combs is attributable to the
waveguide supporting higher order modes at shorter wavelengths (see
supplementary ﬁg. 2). d Normalized 1st modulated sideband as a function of
applied RF frequency showing comb tunability up to 40 GHz. The pump wave-
length is 738 nm. Spectra are under-sampled due to the limited resolution of the
spectrometer. Gaussian ﬁts are provided to guide the eye.
Article
https://doi.org/10.1038/s41467-023-36870-w
Nature Communications|  (2023) 14:1496 
4

<!-- page 5 -->
Given the pulse’s bandwidth of ~ 0.9 GHz, the observed shift is
over 7 times larger than the pulse bandwidth, almost an order of
magnitude larger than previous studies have achieved47. We empha-
size that the wavelength and pulse duration used in this work is
comparable to the wavelength and lifetime of photons emitted from a
variety of different visible solid-state quantum emitters and the
achieved shift similar to their corresponding inhomogenous distribu-
tion, thus providing a route for deterministically bridging this fre-
quency gap49.
Discussion
We have demonstrated VNIR TFLN phase and amplitude modulators
featuring sub-1 V half-wave voltage, extinction ratios above 20 dB,
on-chip insertion loss as low as 0.7 dB, and electro-optic bandwidths
exceeding 35 GHz. With this performance we have demonstrated
an integrated VNIR EO frequency comb with over 50 lines, and
measured spectral shifting with shifts beyond the intrinsic bandwidth
of the pulse. Together, these results show the suitability of
these devices for both spatial and spectral control of input light.
The performance and scalability of our integrated platform ensures
its suitability for a wide variety of applications. Through combining
this visible platform with other demonstrated TFLN technologies
such as periodically poled lithium niobate (PPLN)50 or laser
integration51,52, applications such as visible on-chip spectroscopy,
photon pair generation and manipulation, and efﬁcient visible light
communication can be realized.
Methods
Device fabrication
The optical waveguide layer is realized using Ar+-based reactive-ion
etching (RIE) with a lithographically deﬁned (Elionix ELS-F125) hydro-
gen silsesquioxane hard-mask30. After etching (180 nm) and cleaning,
the device is cladded with silicon dioxide (~1 μm) via plasma-enhanced
chemical vapor deposition. To reduce optical loss and mitigate pho-
torefractive effects, devices are subsequently annealed36,53. Next, the
electrode layer is deﬁned with an electron beam lithography step, RIE
(C3F8, Ar+), electron beam evaporation ( ~ 10/800 nm, Ti/Au), and
lift-off.
Measurement details
Measurement of half-wave voltage. Devices are characterized in the
634–638 nm and 720–940 nm ranges using New Focus Velocity and M2
SolsTis tunable lasers. The laser source polarization is set using a ﬁber
polarization controller (Thorlabs, FPC560), and the output is launched
into the device coupling waveguide using a single-mode lensed ﬁber
(OZ Optics TSMJ-3A-650-4/125-0.25-20-2-10-1). The transmitted light is
then collected using a second lensed ﬁber at the output waveguide and
sent to a high sensitivity avalanche photodetector (APD410A), home-
built fabry-perot cavity (FP), or optical spectrum analyzer (OSA,
AQ6730) depending on the measurement being performed. Coplanar
ground-signal-ground (GSG) electrodes are contacted using 50 Ω GSG
probes (GGB Industries, 40A-GSG-100-F). The DC performance is
evaluated using a 1 MHz triangle waveform, and the normalized
transmission is recorded as a function of applied voltage.
Measurement of high frequency half wave voltage. In order to
measure the half-wave voltage (Vπ) of the MZMs at high frequencies, a
sinusoidal RF tone is applied at varying frequencies and the resulting
optical spectrum is measured. The optical frequency spectrum of an
MZM given an input CW optical carrier frequency of ω0, an applied RF
tone at frequency ωm and amplitude V0, and internal phase between
the arms of ϕ is given by:
Iðω0 + kωmÞ / 1
2 J2
kðπV 0=V πÞ½1 + ð1Þk cos ðϕÞ
ð1Þ
where k is an integer of the harmonic of the drive frequency54.
The frequency spectrum is measured using a home-built Fabry-
Perot cavity (linewidth = 200 MHz) for lower frequencies ( < 15 GHz)
and an optical spectrum analyzer (OSA, AQ6730) or Czerny-Turner
spectrometer (Princeton Instruments, SpectraPro HRS) for higher
frequencies. For the FP cavity, the intensity of the carrier is measured
as a function of applied voltage and ﬁt to (1). For measurements with
the OSA, a wider spectrum featuring multiple sidebands is measured at
various powers and the relative intensities of the even sidebands of
each spectrum is ﬁt to equation (1).
Frequency shearing measurement. To implement spectral shearing,
the RF tone applied to the device and optical pulse are generated using
the same arbitrary waveform generator (AWG, Tektronix 700001b),
while also ensuring minimal jitter between the two. Optical pulses are
deﬁned using a commercial amplitude electro-optic modulator
(EOSpace, AZ-AV5-40-PFA-PFA-737) with a continuous-wave laser. The
spectrum of the pulse is measured using a FP cavity. We numerically
ﬁnd the non-linearity of the sine tone in this quasi-linear region to have
a negligible effect on the resulting spectrum for the given drive fre-
quency and pulse duration.
Normalized Counts
Frequency (GHz)
0.0
Laser
F.P.C.
TFLN
AWG
FP Cavity
OSC
RF Amp
0.2
0.4
0.6
0.8
1.0
-10
-5
0
5
10
Original
Redshift
Blueshift
EOM
50 Ω 
RF Drive
(a)
(c)
(b)
Blueshift
Redshift
Fig. 4 | Visible spectral shearing. a Principle of spectral shearing, where the rising
(falling) edge of a sinusoidal tone is used to provide a linear phase across an optical
pulse in order to blue (red) shiftits frequency. b Diagram of the experimental setup.
An arbitrary waveform generator (AWG) is used to generate both 1 ns square
electrical pulses and the 100 MHz sinusoidal drive for the phase modulator. The
square electrical pulses generate square optical pulses via a commercial amplitude
EO modulator, which are then routed to the low Vπ TFLN phase modulator. The RF
drive from the AWG is ampliﬁed before reaching the device. c Frequency spectrum
of the pulses after applying a sinusoidal tone and synchronizing the pulse with the
falling (redshift) or rising (blueshift) edge of the RF tone, showing a shift of ± 6.6
GHz (over 7 times the pulse bandwidth) relative to the original spectrum with no RF
tone applied. EOM Electro-optic modulator, AWG Analog waveform generator,
F.P.C. Fiber polarization controller, TFLN Thin-ﬁlm lithium niobate, FP Fabry-Perot,
OSC Oscilloscope.
Article
https://doi.org/10.1038/s41467-023-36870-w
Nature Communications|  (2023) 14:1496 
5

<!-- page 6 -->
Data availability
All data that supports the conclusions of this study are included in the
article and the Supplementary Information ﬁle. The data presented in
this study is available from the corresponding authors upon request.
References
1.
Hainberger, R. et al. Cmos-compatible silicon nitride waveguide
photonic building blocks and their application for optical coher-
ence tomography and other sensing applications (2020).
2.
Sacher, W. D. et al. Visible-light silicon nitride waveguide devices
and implantable neurophotonic probes on thinned 200 mm silicon
wafers. Opt. Express 27, 37400 (2019).
3.
Goykhman, I., Desiatov, B. & Levy, U. Ultrathin silicon nitride
microring resonator for biophotonic applications at 970 nm wave-
length. Appl. Phys. Lett. 97, 081108 (2010).
4.
McCracken, R. A., Charsley, J. M. & Reid, D. T. A decade of
astrocombs: Recent advances in frequency combs for astronomy.
Opt. Express 25, 15058–15078 (2017).
5.
Strinati, E. C. et al. 6g: The next frontier: From holographic messa-
ging to artiﬁcial intelligence using subterahertz and visible light
communication. IEEE Vehicular Technol. Mag. 14, 42–50 (2019).
6.
Awschalom, D. et al. Development of quantum interconnects
(quics) for next-generation information technologies. PRX Quantum
2, 017002 (2021).
7.
Moody, G. et al. Roadmap on integrated quantum photonics. arXiv
preprint arXiv:2102.03323 (2021).
8.
Levine, H. et al. High-ﬁdelity control and entanglement of rydberg-
atom qubits. Phys. Rev. Lett. 121, 123603 (2018).
9.
Madjarov, I. S. et al. High-ﬁdelity entanglement and detection of
alkaline-earth Rydberg atoms. Nat. Phys. 16, 857–861 (2020).
10.
Bruzewicz, C. D., Chiaverini, J., McConnell, R. & Sage, J. M. Trapped-
ion quantum computing: Progress and challenges. Appl. Phys. Rev.
6, 021314 (2019).
11.
Sinclair, N. et al. Spectral multiplexing for scalable quantum pho-
tonics using an atomic frequency comb quantum memory and
feed-forward control. Phys. Rev. Lett. 113, 053603 (2014).
12.
Bradac, C., Gao, W., Forneris, J., Trusheim, M. E. & Aharonovich, I.
Quantum nanophotonics with group iv defects in diamond. Nat.
Commun. 10, 5625 (2019).
13.
Ruf, M., Wan, N. H., Choi, H., Englund, D. & Hanson, R. Quantum
networks based on color centers in diamond. J. Appl. Phys. 130,
070901 (2021).
14.
Uppu, R. et al. Scalable integrated single-photon source. Sci. Adv.
6, eabc8268 (2020).
15.
Zhai, L. et al. Low-noise gaas quantum dots for quantum photonics.
Nat. Commun. 11, 4745 (2020).
16.
Liu, J. et al. A solid-state source of strongly entangled photon pairs
with high brightness and indistinguishability. Nat. Nanotechnol. 14,
586–593 (2019).
17.
Saha, U. et al. Routing single photons from a trapped ion using a
photonic integrated circuit. arXiv preprint arXiv:2203.08048
(2022).
18.
Monroe, C. et al. Large-scale modular quantum-computer archi-
tecture with atomic memory and photonic interconnects. Phys. Rev.
A 89, 022317 (2014).
19.
Levonian, D. et al. Optical entanglement of distinguishable quan-
tum emitters. Phys. Rev. Lett. 128, 213602 (2022).
20. Reimer, C. et al. Generation of multiphoton entangled quantum
states by means of integrated frequency combs. Science 351,
1176–1180 (2016).
21.
Kues, M. et al. Quantum optical microcombs. Nat. Photonics 13,
170–179 (2019).
22. Dong, M. et al. High-speed programmable photonic circuits in a
cryogenically compatible, visible-near-infrared 200 mm cmos
architecture. Nat. Photonics 16, 59–65 (2022).
23. Muñoz, P. et al. Silicon nitride photonic integration platforms for
visible, near-infrared and mid-infrared applications. Sensors 17,
2088 (2017).
24. Khan, M., Babinec, T., McCutcheon, M. W., Deotare, P. & Lončar, M.
Fabrication and characterization of high-quality-factor silicon
nitride nanobeam cavities. Opt. Lett. 36, 421 (2011).
25. Romero-García, S., Merget,F., Zhong, F., Finkelstein, H. & Witzens, J.
Silicon nitride cmos-compatible platform for integrated photonics
applications at visible wavelengths. Opt. Express 21, 14036 (2013).
26. Lu, T.-J. et al. Aluminum nitride integrated photonics platform for
the ultraviolet to visible spectrum. Opt. Express 26, 11147 (2018).
27.
He, J. et al. Nonlinear nanophotonic devices in the ultraviolet to
visible wavelength range. Nanophotonics 9, 3781–3804 (2020).
28. Burek, M. J. et al. High quality-factor optical nanocavities in bulk
single-crystal diamond. Nat. Commun. 5, 5718 (2014).
29. Nguyen, C. T. et al. An integrated nanophotonic quantum register
based on silicon-vacancy spins in diamond. Phys. Rev. B 100,
165428 (2019).
30. Desiatov, B., Shams-Ansari, A., Zhang, M., Wang, C. & Lončar, M.
Ultra-low-loss integrated visible photonics using thin-ﬁlm lithium
niobate. Optica 6, 380–384 (2019).
31.
Celik, O. T. et al. High-bandwidth cmos-voltage-level electro-optic
modulation of 780 nm light in thin-ﬁlm lithium niobate. Opt. Express
30, 23177 (2022).
32. Li, C. et al. High modulation efﬁciency and large bandwidth thin-
ﬁlm lithium niobate modulator for visible light. Opt. Express 30,
36394–36402 (2022).
33. Zhu, D. et al. Integrated photonics on thin-ﬁlm lithium niobate. Adv.
Opt. Photonics 13, 242–352 (2021).
34. Wang, C. et al. Ultrahigh-efﬁciency wavelength conversion in
nanophotonic periodically poled lithium niobate waveguides.
Optica 5, 1438–1441 (2018).
35. Zhang, M., Wang, C., Cheng, R., Shams-Ansari, A. & Lončar, M.
Monolithic ultra-high-q lithium niobate microring resonator. Optica
4, 1536–1537 (2017).
36. Shams-Ansari, A. et al. Reduced material loss in thin-ﬁlm lithium
niobate waveguides. Apl. Photonics 7, 081301 (2022).
37. Sund, P. I. et al. High-speed thin-ﬁlm lithium niobate quantum
processor driven by a solid-state quantum emitter. arXiv preprint
arXiv:2211.05703 (2022).
38. Renaud, D., Assumpcao, D., Shams-Ansari, A., Barton, D. & Loncar,
M. Low-loss ﬁber-to-chip visible couplers in thin-ﬁlm lithium nio-
bate. CLEO Conference 2022, Poster Session (2022).
39. Kharel, P., Reimer, C., Luke, K., He, L. & Zhang, M. Breaking
voltage–bandwidth limits in integrated lithium niobate modulators
using micro-structured electrodes. Optica 8, 357–363 (2021).
40. Wang, C. et al. Integrated lithium niobate electro-optic modulators
operating at cmos-compatible voltages. Nature 562,
101–104 (2018).
41.
Kamada, S. et al. Superiorly low half-wave voltage electro-optic
polymer modulator for visible photonics. Opt. Express 30,
19771–19780 (2022).
42. Liang, G. et al. Robust, efﬁcient, micrometre-scale phase mod-
ulators at visible wavelengths. Nat. Photonics 15, 908–913 (2021).
43. Shams-Ansari, A. et al. Thin-ﬁlm lithium-niobate electro-optic plat-
form for spectrally tailored dual-comb spectroscopy. Commun.
Phys. 5, 1–8 (2022).
44. Johnson, L. M. & Cox, C. H. Serrodyne optical frequency translation
with high sideband suppression. J. Lightwave Technol. 6,
109–112 (1988).
45. Wright, L. J., Karpiński, M., Söller, C. & Smith, B. J. Spectral shearing
of quantum light pulses by electro-optic phase modulation. Phys.
Rev. Lett. 118, 023601 (2017).
46. Lukens, J. M. & Lougovski, P. Frequency-encoded photonic qubits
for scalable quantum information processing. Optica 4, 8 (2017).
Article
https://doi.org/10.1038/s41467-023-36870-w
Nature Communications|  (2023) 14:1496 
6

<!-- page 7 -->
47. Zhu, D. et al. Spectral control of nonclassical light pulses using an
integrated thin-ﬁlm lithium niobate modulator. Light.: Sci. Appl. 11,
1–9 (2022).
48. Puigibert, M. G. et al. Heralded single photons based on spectral
multiplexing and feed-forward control. Phys. Rev. Lett. 119,
083601 (2017).
49. Evans, R. E., Sipahigil, A., Sukachev, D. D., Zibrov, A. S. & Lukin, M. D.
Narrow-linewidth homogeneous optical emitters in diamond
nanostructures via silicon ion implantation. Phys. Rev. Appl. 5,
044010 (2016).
50. Xin, C. et al. Spectrally separable photon-pair generation in
dispersion engineered thin-ﬁlm lithium niobate. Opt. Lett. 47,
2830–2833 (2022).
51.
Shams-Ansari, A. et al. Electrically pumped laser transmitter inte-
grated on thin-ﬁlm lithium niobate. Optica 9, 408–411 (2022).
52. de Beeck, C. O. et al. Iii/v-on-lithium niobate ampliﬁers and lasers.
Optica 8, 1288–1289 (2021).
53. Xu, Y. et al. Mitigating photorefractive effect in thin-ﬁlm lithium
niobate microring resonators. Opt. Express 29, 5497–5504 (2021).
54. Shi, Y., Yan, L. & Willner, A. E. High-speed electrooptic modulator
characterization using optical spectrum analysis. J. Lightwave
Technol. 21, 2358 (2003).
Acknowledgements
This work was supported in part by AFOSR FA9550-20-1-0105 (M.L.),
FA9550-19-1-0376 (M.L.), ARO MURI W911NF1810432 (M.L.), NSF EEC-
1941583 (M.L.), OMA-2137723 (M.L.), and OMA-2138068 (M.L.), DOE
DE-SC0020376 (M.L.), MIT Lincoln Lab 7000514813 (M.L.), AWS Center
for Quantum Networking’s research alliance with the Harvard Quantum
Initiative (M.L.), Ford Foundation Fellowship, (D.R.), NSF GRFP No.
DGE1745303 (D.R., D.A.), NSERC PGSD scholarship (G.J.), Harvard
Quantum Initiative (HQI) postdoctoral fellowship (D.Z.), A*STAR
Central Research Fund (D.Z.), AQT Intelligent Quantum Networks and
Technologies (N.S.), and NSF Center for Integrated Quantum Materials
No. DMR-1231319 (D.R., N.S.). We acknowledge fruitful discussions with
Lingyan He, Prashanta Kharel, Ben Dixon, and Alex Zhang. Device fab-
rication was performed at the Center for Nanoscale Systems (CNS), a
member of the National Nanotechnology Coordinated Infrastructure
Network (NNCI), which is supported by the National Science Foundation
under NSF Grant No. 1541959.
Author contributions
G.J. and D.R. designed devices. D.R. fabricated devices. D.A., D.R., and
G.J. designed and performed the measurements. A.S. assisted with
electronics measurements. D.A., D.R., and D.Z. analyzed the data. Y.H.
provided a theory on frequency combs. D.R., D.A., and A.S. wrote
the manuscript with extensive input from the other authors. M.L.
and N.S. supervised the project. These authors contributed equally:
D.R. and D.A.
Competing interests
M.L. is involved in developing lithium niobate technologies at Hyper-
Light Corporation. The remaining authors declare no competing
interests.
Additional information
Supplementary information The online version contains
supplementary material available at
https://doi.org/10.1038/s41467-023-36870-w.
Correspondence and requests for materials should be addressed to
Dylan Renaud or Marko Loncar.
Peer review information Nature Communications thanks the anon-
ymous reviewer(s) for their contribution to the peer review of this
work. Peer reviewer reports are available.
Reprints and permissions information is available at
http://www.nature.com/reprints
Publisher’s note Springer Nature remains neutral with regard to jur-
isdictional claims in published maps and institutional afﬁliations.
Open Access This article is licensed under a Creative Commons
Attribution 4.0 International License, which permits use, sharing,
adaptation, distribution and reproduction in any medium or format, as
long as you give appropriate credit to the original author(s) and the
source, provide a link to the Creative Commons license, and indicate if
changes were made. The images or other third party material in this
article are included in the article’s Creative Commons license, unless
indicated otherwise in a credit line to the material. If material is not
included in the article’s Creative Commons license and your intended
use is not permitted by statutory regulation or exceeds the permitted
use, you will need to obtain permission directly from the copyright
holder. To view a copy of this license, visit http://creativecommons.org/
licenses/by/4.0/.
© The Author(s) 2023
Article
https://doi.org/10.1038/s41467-023-36870-w
Nature Communications|  (2023) 14:1496 
7

