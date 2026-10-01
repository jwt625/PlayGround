---
paper_id: porto2026
source_url: https://doi.org/10.1364/ofc.2026.th1c.4
doi: 10.1364/ofc.2026.th1c.4
license: publisher-copyright
sha256: c0d6bb9d73e2e2a9cd84b91ae2d6898aef792862169fbbe6c92d9f992f74c8a5
pages: 3
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
1.6 Tb/s Monolithic InP Transmitter PIC with DFB, MZM, 
and SOA Arrays 
 
S. Porto, F. Sang, J. Lavrencik, P. Abolghasem, A. Rundquist, J. Macario, A. Meighan, C. Ozdemir, Y.-P. 
Wang, W. Fu, S. Sakpal, P. Gavrilovic, T. Frost, J. Zhang, P. Turla, J. Anthony, K. Hoshino, C. DiGiovanni, 
A. Diba, S. Barik, M. Kuntz, V. Lal, P. Evans, M. Ziari 
Nokia Corporation, 1322 Bordeaux Dr. Sunnyvale, CA  94089, USA 
stefano.porto@nokia.com 
 
Abstract: We report integration of laser sources and amplifiers with low-voltage of ≤1.5 V 
modulators in an eight-channel monolithic InP PIC and demonstrate single-channel 212 Gbps direct 
linear drive with fiber-coupled output power of +5 dBm. © 2025 Nokia Corporation 
 
1. Introduction 
Datacenters designed for artificial intelligence applications are rapidly increasing in computational capacity and, 
consequently, power consumption.  Optical links within these datacenters are necessary but consume power that 
could otherwise be used for computation.  In order to enable more computation and less overall power consumption, 
more efficient optical links are needed.  Linear Pluggable Optics (LPOs) are being developed without digital signal 
processors (DSPs) that otherwise dominate power dissipation in pluggables.  We recently reported on 8-channel 
monolithic InP transmitter photonic integrated circuits (PICs) in Octal Small Form-Factor Pluggable (OSFP) module 
that demonstrated bit error rates (BER) < 10-7 over 15 km SMF in a retimed OSFP configuration [1].  However, 
since each channel pair in these PICs shared a laser, it was not possible to independently control the output power of 
individual channels in normal operation or to shutter individual channels when needed.  It is therefore desirable to 
implement semiconductor optical amplifiers (SOAs) to enable per-channel power boost, control, and shuttering, and 
to demonstrate linear drive operation.  SOAs have been widely used in coherent transmitter PICs after Mach-
Zehnder modulators (MZMs) to preserve signal quality [2,3].  SOAs have also been successfully used after electro-
absorption modulators [4] in links with intensity modulation with direct detection (IM/DD) and pulse amplitude 
modulation with 4 levels (PAM4). 
In this paper we report on eight-channel monolithic InP transmitter PICs that integrate arrays of four O-band 
DFB lasers, eight MZMs, and eight SOAs.  We first test PICs on carrier for DC performance, then characterize one 
channel of a PIC for RF performance when wire-bonded to a SiGe BiCMOS MZM driver (MZMD) on carrier.  The 
RF data are taken for 106 GBd PAM4, 212 Gbps operation at 60 °C for 1.6 Tb/s LPO O-band applications. 
2. PIC Architecture 
The transmitter PIC is composed of DFB, MZM, and SOA arrays, as shown in Fig. 1.  Each of four DFBs supplies 
two optical channels, one from each end of the DFB.  Eight MZMs receive light from the corresponding DFBs and 
are modulated by RF input signals from an eight-channel open-collector SiGe driver chip.  Each traveling-wave 
modulator contains DC phase adjustors in each arm and is terminated by integrated 30-ohm metallic load resistors 
that electrically connect the two modulator arms.  The modulated optical signal from each MZM is routed to a per-
channel SOA.  The output of each SOA is tapped with a multi-mode interference (MMI) coupler to supply an on-
chip power monitor for maintaining the modulator at quadrature and attaining the desired overall output power, 
before routing to an integrated spot-size converter at the edge of the PIC. This monolithic InP PIC was fabricated 
using the same process as our prior work that did not use SOAs [1]. 
 
 
Fig. 1.  Monolithic PIC Architecture comprises 4 DFBs, 8 MZMs, 8 SOAs, and 8 taps with power monitors. 
 
Th1C.4.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 2 -->
3. DC Characterization 
PICs are first attached and have DC connections wire-bonded to AlN carriers, and all 8 channels are characterized 
for DC performance at 60 °C.  In Fig. 2(a) SOA DC gain vs. input power is shown for a variety of SOA currents 
with the MZMs set to their maximum transmission state by tuning the DC phase adjustors.  1 dB and 3 dB 
compression points are labeled for convenience.  The SOA length should be chosen so that its gain remains 
adequately linear over the range of operating current needed to span the range of required average output power.  
SOAs have a fairly constant noise figure when operated at least five times their transparency current [5].  Other 
PAM4 transmission work using SOAs to boost electro-absorption modulated laser (EML) outputs indicated that 
operating up to 3 dB of gain compression results in improved BER when using a linear equalizer DSP [4].  Fig. 2(a) 
shows the gain compression for the expected SOA operating current range, and we expect that this configuration 
will operate at less than 1 dB gain compression.  For best noise figure and at most 1dB compression, this SOA can 
therefore safely operate from 40 to 160 mA.  In live networks, it is sometimes necessary to shut down a channel by 
greatly reducing output power.  In Fig. 2(b), the SOA is reverse-biased to quantify its capability to shutter a channel 
when the DFB is still forward-biased at a fixed current of 150mA and the MZM is set to the quadrature point.  As 
the SOA forward-bias current is reduced to zero, there is loss of gain and some absorption below transparency.  As 
the SOA is then reverse-biased, absorption increases.  For normal operation, the DFB currents are set to 150 mA, 
and a fiber with photodetector is coupled to the PIC to measure power.  SOA currents are adjusted (ranging from 57 
to 95 mA) for each channel so that fiber-coupled output power is +5 dBm.   
It is also necessary to characterize the DFB and SOA for noise behavior.  Relative Intensity Noise (RIN) of the 
DFB is measured by operating it at 150 mA and measuring output intensity fluctuations vs. frequency with a high-
speed photodiode when the SOA current is set to 100 mA and the MZM is set to maximum transmission state.  
Integrating from 0.1 to 18GHz, we find the average RIN is 0.4%. Lastly, the noise figure of the SOA is 
characterized.  An estimate of optical power into the SOA is made by reverse-biasing the SOA and measuring its 
photocurrent, then measuring the SOA under normal bias conditions of 60 to 160 mA to extract the noise figure, 
typically 6.5 dB. The DFB side mode suppression ratio was measured > 50 dB for all four DFBs. 
4. RF Characterization 
To enable high speed testing, PIC and MZMD chips are mounted on an AlN carrier, and one channel is 
characterized for RF performance by contacting the MZMD input with a microwave probe and aligning an optical 
fiber probe to the corresponding output facet. For clarity, only the performance of a representative channel is 
presented here. A high-speed digital-to-analog converter (DAC) injects a differential 106.25 GBd SSPRQ pattern 
into the driver and the resulting optical signal is analyzed with a sampling oscilloscope, as shown in Fig. 3. Both the 
transmitter and receiver ends employ a feed-forward equalizer (FFE) with 15 T-spaced taps, though fewer than ten 
taps exhibited non-negligible weights, with the remaining coefficients being close to zero. The modulator is 
designed for RF Vπ less than 1.5 V while providing over 25 dB of extinction and 53 GHz of bandwidth. The SiGe 
MZMD provides about 5-16 dB of tunable electrical peaking at 53 GHz to compensate for potential electrical path 
 
 
Fig. 2. (a) SOA DC gain vs. input power and SOA current, with 1 dB and 3 dB compression points labelled.  (b) SOA output power vs. bias 
current and reverse bias voltage, for a fixed DFB current of 150mA. 
 
Th1C.4.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 3 -->
losses. To demonstrate the linearity of the SOA, we set the DFB current to 150 mA and two power targets of ~3 
dBm and ~5 dBm average power into the fiber under modulation. The SOA current is 60 mA for Fig. 3(a) and 160 
mA for Fig. 3(b).  Per the quantified data on the left side of Fig. 3, we are able to maintain the same transmitter 
dispersion and eye closure quaternary (TDECQ), outer extinction ratio (ER), and ratio level mismatch (RLM) across 
a 2.5x change in SOA drive current.  In further experiments, the PIC and SiGe chips were packaged in an OSFP 
LPO configuration with a commercially available receiver.  Driving the OSFP in 106 GBd PAM4 on all 8 channels 
produced modulated average output powers in the range of 2.8-3.0 dBm.  RF set-up, optimization and reporting will 
follow. 
5. Conclusion 
We have fabricated a 106 GBd PAM4 monolithic InP transmitter PIC with four DFBs and eight-channel arrays of 
MZMs, SOAs and power monitors.  DC, noise, and RF characterization indicate that the SOAs are suitable for this 
application, providing optical power control, boost, and shuttering capabilities.  We demonstrated consistent DC and 
RF-probed performance driven by a SiGe ASIC over a range of SOA currents fixed at 60 °C, including -20 dBm 
shuttering, RLM ~0.97, outer ER ~3.5 dB, TDECQ ~3dB, and average fiber coupled power ~3 to ~5 dBm.  The 
demonstrated PIC architecture is a promising technology for the transmitter optical engine in 1.6 Tb/s LPO 
applications. 
6. References 
 
[1] A. Rundquist, J. Lavrencik, P. Abolghasem, M. Chagnon, S. Porto, P. Day, J. Anthony, C. Ozdemir, J. Patton, P. Gavrilovic, T. Frost, P. 
Turla, J. Macario, R. Safian, F. Sang, A. Diba, A. Meighan, M. Kuntz, R. Maher, V. Lal, M. Ziari, P. Evans, “8-Channel Monolithic InP 
Transmitter PIC Integrating DFB and MZM Arrays, Capable of Operating 106 GBd PAM4 at 85 °C,” 2025 European Conference on Optical 
Communication (ECOC), Tu.04.03.2, Copenhagen, Denmark (2025). 
 
[2] V. Lal, P. Studenkov, T. Frost, H. Tsai, B. Behnia, J. Osenbach, S. Wolf, R. Going, S. Porto,R. Maher, H. Hodaei, J. Zhang, C. Di Giovanni, 
K. Hoshino, T. Vallaitis, B. Ellis, J. Yan, K. Fong, E. Sooudi, M. Kuntz, S. Buggaveeti, D. Pavinski, S. Sanders, Z. Wang, G. Hoefler, P. Evans, 
S. Corzine, T. Butrie, M. Ziari, F. Kish, D. Welch, “1.6Tbps Coherent 2-Channel Transceiver Using a Monolithic Tx/Rx InP PIC and Single SiGe 
ASIC,” 2020 Optical Fiber Communications Conference and Exposition (OFC), M3A.2, San Diego, USA 2020. DOI: 10.1364/OFC.2020.M3A.2 
 
[3] M. Ziari, V. Lal, P. Studenkov, H. Hodaei, T. Frost, C. Tsai, K. Hoshino, B. Behnia, R. Going, S. Wolf, S.Porto, M. Kuntz, T. Vallaitis, E. 
Sooudi, J. Lavrencik, R. Salvatore, N. Modi, P. Abolghasem, M. Fisher, S. Murthy, S. Buggaveeti, R. Brigham, D. Pavinski, J. Zhang, J. Zhang, 
S. Corzine, P. Evans, V. Dominic, R. Maher, P. Mertz, S. Sanders, H. Sun, J. Osenbach, P. Kandappan, “High-performance 100 Gbaud Coherent 
Photonic Modules,” 2021 Optical Fiber Communications Conference and Exposition (OFC), F2A.3, San Diego, USA 2021. DOI: 
10.1364/OFC.2021.F2A.3 
 
[4] H. H. Lee , K. O. Kim, K.-H. Doo, J.-Y. Oh, G.-Y. Moon, S.-H. Baek, and H. S. Chung, “Demonstration of High-Power Budget TDM-PON 
System With 50 Gb/s PAM4 and Saturated SOA,” Journal of Lightwave Technology, Vol. 39, No. 9, May 1, 2021, pp. 2762-2768. 
 
[5] S. S. Saini, J. Bowser, R. Enck, V. Luciani, P. J. S. Heim, and M. Dagenais, “A Semiconductor Optical Amplifier with high saturation power, 
low noise figure and low polarization dependent gain over the C-band,” The 17th Annual Meeting of the IEEE Lasers and Electro-Optics Society, 
MK6, pp. 102-103, (2004).  DOI: 10.1109/LEOS.2004.1363131 
Fig. 3.  Inset on left summarizes RF performances of (a) eye diagram at SOA current = 60 mA and (b) eye diagram at SOA current = 160mA. 
Parameters marked with * were measured using an optical attenuator to maintain the oscilloscope input power at ~0dBm. 
Th1C.4.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

