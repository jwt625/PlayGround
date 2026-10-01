---
paper_id: ogiso2024
source_url: https://doi.org/10.1364/ofc.2024.tu2d.7
doi: 10.1364/ofc.2024.tu2d.7
license: publisher-copyright
sha256: 406874f77094f63cce16b047a14c9c882f2203ebe8895fe3e3077415ac268be2
pages: 3
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
High Speed InP Modulator for Beyond 200 Gbaud 
 
Yoshihiro Ogiso, Josuke Ozaki, Kenta Sugiura, Yusuke Saito, and Mitsuteru Ishikawa 
NTT Innovative Devices Corporation, 3-1 Morinosato Wakamiya, Atsugi City, Kanagawa, Japan 
yoshihiro.ogiso@ntt-devices.com 
 
Abstract: We developed a next-generation InP twin-IQ modulator PIC for beyond 200-Gbaud 
operations. A 3-dB electro-optic bandwidth of the modulator exceeds 100 GHz while maintaining 
a half-wave voltage of 1.5 V and total on-chip optical insertion loss of less than 3.5 dB. 
OCIS codes: (250.4110) Modulators; (320.7080) Ultrafast optics 
 
1. Introduction 
Although spectral efficiency isn’t expected to improve, there is still strong demand for ultra-high bandwidth electro-
optic components. In particular, the recent growth in artificial intelligence and machine learning (AI/ML) 
technologies, such as generative AI, has accelerated development of higher-speed and lower-energy optical 
transmitters. The electro-optic (EO) modulator is a key component for high-speed applications, as well as high-
speed electronics such as digital-signal processing (DSP) and amplifier ICs. For realizing the next generation 1.6-
Tbps system in which 200-Gbaud-class operation would be required, all of these components must have their 
bandwidths of around 100 GHz. In addition to needing a high bandwidth, they must also have a low optical insertion 
loss and low half-wave voltage (Vπ) in order to have sufficient optical signal-to-noise ratio (OSNR) characteristics. 
Here, InP is one of the most promising platforms for practical transceivers, because it not only provides high-
modulation efficiency with a small footprint but also has mature epitaxial growth technology for high-precision 
manufacturing. 
We have already developed a first-generation ultra-high bandwidth InP-based modulator photonic IC (PIC) for a 
130-Gbaud-class coherent driver modulator (CDM). The CDM has an EO bandwidth of over 80-GHz and an 
insertion loss of less than 7.0 dB, including applied bias absorption for a 2.0-V Vπ [1]. In this paper, we introduce the 
next generation of InP-based modulator PIC based on an n-i-p-n heterostructure [2,3]. By modifying the RF 
electrode design and optical waveguide structure, we were able to extend the bandwidth and reduce Vπ without 
increasing the optical loss. The 3-dB EO bandwidth of the twin-IQ modulator exceeded 100 GHz, which means that  
it can support 200 Gbaud operations and beyond. 
2.  Modulator PIC 
Figure 1(a) shows a schematic diagram and images of a fabricated twin-IQ modulator PIC. The overall layout is 
almost the same as the one described in our previous report [4]. The symmetrical channel layout provides better IQ 
phase stability and low-loss microwave feeding into the modulation region. The RF modulation region combines an 
inverted-trapezoidal ridge waveguide with an n-i-p-n heterostructure and capacitance-loaded traveling-wave 
electrode (CL-TWE). For the DC phase adjustment, we employ a thermo-optic heater in which the half-wave shift 
power (Pπ) is less than 20 mW. Figure 1(b) shows the extinction characteristics obtained by single-arm driving in the 
RF modulation region. We also measured optical absorption characteristics and bias voltage characteristics of Vπ. 
Moreover, we estimated the optical absorption dependence of Vπ from these results. As shown in Fig. 1(c), the Vπ of 
1.4 V could be adjustable if up to 1.0 dB of additional loss is tolerated. 
 
Fig. 1. (a) Modulator PIC outline, (b) extinction characteristics (single-arm drive), and (c) optical absorption loss due to applied bias for 
required Vπ 
Tu2D.7
OFC 2024 © Optica Publishing Group 2024
Disclaimer: Preliminary paper, subject to publisher revision
Authorized licensed use limited to: Stanford University. Downloaded on July 06,2026 at 04:29:42 UTC from IEEE Xplore.  Restrictions apply. 

<!-- page 2 -->
2.  DC characteristics 
We measured the optical insertion losses and extinction characteristics under two different Vπ conditions determined 
by the applied bias voltage. We set a low Vπ of 1.5 V, for which the applied voltage was around -10 V in the entire 
C band. We conducted PIC measurements using 4.5 mΦ lensed fiber alignment systems. Figure 2(a) shows the 
optical insertion losses for each and both polarization conditions. The loss of each polarization was less than 11.5 dB, 
which includes the absorption loss due to the applied bias and fiber coupling losses (2.3 dB/facet [4]). As shown in 
Fig. 2(b), the extinction ratio was over 25 dB in the entire C band for all child- and parent-Mach-Zehnder 
interferometer circuits. For reference, we also measured the extinction characteristics under actual push-pull drive 
conditions (Fig. 2(c)). Next, we investigated the wide-wavelength-range capabilities under the 2.0-V Vπ condition 
because the optical absorption due to the applied bias was sufficiently low in the entire C+L band. For wide-
wavelength-range operations, we fabricated two PICs, one with a multi-mode interferometer (MMI) waveguide, the 
other with a taper cross waveguide. Figure 3(a) shows the insertion losses of the two different designs. the maximum 
transmittance of the MMI type was slightly better than that of the taper type in C- or L-band operation. Although 
degradation due to the wavelength dependence of the MMI coupler can’t be ignored in C+L-band operation, we 
could keep the extinction ratio over 25 dB throughout the C+L band, as shown in Fig. 3(b). On the other hand, the 
wavelength dependence was improved by the wavelength-insensitive taper cross waveguide. 
  
    
Fig. 2.  DC characteristics under 1.5-V Vπ condition (a) insertion loss, (b) extinction ratio, and (c) extinction curve (push-pull drive) 
 
 
Fig. 3.  DC characteristics under 2.0-V Vπ condition (a) insertion losses of MMI- and taper-type cross waveguides, and (b) extinction ratio of 
MMI-type cross waveguide 
3.  High-frequency characteristics 
Figure 4 shows the EO responses of three different modulator PICs which had the same length of CL- traveling-
wave electrode (3.6 mm) and characteristic impedances (Z0) designed to be around 60 Ω. For reference, Fig. 4(a) 
depicts the EO response of a 130-Gbaud-class commercial PIC. The 3-dB EO bandwidth was around 70 GHz, which 
was mainly limited by microwave loss and the parasitic capacitance of the TWE. Thus, we modified the electrode 
design and fabrication process, in which we shortened the period length of the capacitance-loaded TWE from 150 to 
120 m to increase the roll-off frequency determined by Bragg reflection and RC time constant. Figure 4(b) shows 
the results of the modifications. Although the 3-dB bandwidth reached 90 GHz, it was still not sufficient for 200-
Gbaud-class operation. Next, we modified the structure of the optical waveguide because it strongly impacts the 
capacitance of the RF circuit. Thanks to this modification, we reduced the capacitance, which resulted in a further 
increase in bandwidth. As shown in Fig. 4(c), the 3-dB and 6-dB EO bandwidths exceeded 100 and 110 GHz, 
respectively. Owing to the reduction in capacitance, Z0 of the CL-TWE was increased, which in turn enhanced the 
Tu2D.7
OFC 2024 © Optica Publishing Group 2024
Disclaimer: Preliminary paper, subject to publisher revision
Authorized licensed use limited to: Stanford University. Downloaded on July 06,2026 at 04:29:42 UTC from IEEE Xplore.  Restrictions apply. 

<!-- page 3 -->
EO response in the low-frequency region. We can flexibly control the response by adjusting the value of the RF 
termination resistor [5] or/and Z0 of the CL-TWE. For example, we have room to increase the resister value when a 
higher Z0 is required for co-designing with an analog IC. Moreover, we could decrease the Z0 of the CL-TWE by 
employing a lower-microwave-loss electrode to increase the bandwidth further. 
 
Fig. 4.  EO responses of three variant PICs: (a) 130-GBaud-class product (reference), 
(b) electrode modification, and (c) electrode and optical waveguide modifications 
 
4.  Conclusion 
We described our recent work on next-generation InP modulator PICs that can operate above 200 Gbaud. Here, we 
investigated not only the symbol rate but also wavelength scalabilities. The specifications are summarized in Table 1. 
By modifying both the electrode design and optical waveguide structure, we can extend the EO bandwidth without 
degrading the optical properties. These PICs will be ready for mass production in the near future. In addition, we 
expect that higher speed modulation can be achieved by further optimization of the optical waveguide and CL-TWE 
designs and by integration with a semiconductor optical amplifier (SOA).  
 
Table 1.  Modulator PIC spec. summary (Typ. value) 
 
5.  References   
[1] J. Ozaki et al., “Over-85-GHz-Bandwidth InP-Based Coherent Driver Modulator Capable of 1-Tb/s/λ-Class Operation,” J. Lightw. Technol., 
vol. 41, no. 11, pp. 3290–3296 (2023). 
[2] Y. Ogiso et al., “Over 67 GHz Bandwidth and 1.5 V Vπ InP-Based Optical IQ Modulator With n-i-p-n Heterostructure,” J. Lightw. Technol., 
vol. 35, no. 8, pp. 1450–1455 (2017). 
[3] Y. Ogiso et al., “80-GHz bandwidth and 1.5-V Vπ InP-based IQ modulator,” J. Lightw. Technol., vol. 38, no. 2, pp. 249–255 (2020). 
[4] Y. Ogiso et al., “High-Bandwidth InP MZ/IQ Modulator PIC Ready for Practical Use,” ECOC2022, paper Mo3F.3 (2022). 
[5] X. Liu et al., “Capacitively-Loaded Thin-Film Lithium Niobate Modulator With Ultra-Flat Frequency Response,” Photon. Technol. Lett., vol. 
34, no. 16, pp. 854–857 (2022). 
 
Tu2D.7
OFC 2024 © Optica Publishing Group 2024
Disclaimer: Preliminary paper, subject to publisher revision
Authorized licensed use limited to: Stanford University. Downloaded on July 06,2026 at 04:29:42 UTC from IEEE Xplore.  Restrictions apply. 

