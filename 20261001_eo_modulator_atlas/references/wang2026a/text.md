---
paper_id: wang2026a
source_url: https://doi.org/10.1364/ofc.2026.w3e.5
doi: 10.1364/ofc.2026.w3e.5
license: 
sha256: 97fd3049b1cc3c0abdcaee93c3607d28960a63ed85dc1fa117594b1b83f667d9
pages: 3
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
 
 
Toward 400 G/Lane Silicon Differential-Drive Mach-
Zehnder Modulator with > 80 GHz Bandwidth for Optical 
Interconnects  
 
Haibo Wang1,*, Qianni Zhang1, Yanming Zhang1, Pengfei Guo1, Jindian Wei1,  
Wenxu Gu1, Ting Yu Teo1, Haiyang Zheng1, Chao Li1, Patrick Guo-Qiang Lo1 
1. Advanced Micro Foundry Pte Ltd, 117685, Singapore. 
                 * wanghb@advmf.com 
 
Abstract: A differential-drive Silicon Mach–Zehnder modulator is demonstrated, achieving a 
3-dB bandwidth of 81.8 GHz. Despite limitations in the eye-diagram measurement setup, a 
PAM-8 eye-diagram was obtained at 100 Gbaud. 
 
1.  Introduction 
The rapid growth of workloads in AI, IoT, big data analytics, and cloud computing continues to drive the demand 
for ultra-high-speed optical interconnects [1]. The industry is already building and deploying 800 G systems based 
on the IEEE standards (IEEE P802.3dj), and future systems are inevitably evolving toward 1.6 T optical links and 
beyond [2]. To enable compact and energy-efficient transceivers for such high-volume data transmission without 
excessive lanes, each optical channel is required to support a data rate of 400 Gb/s [2]. These impose the 
requirement of modulator bandwidth and efficiency. Among various material platforms, silicon (Si) modulators 
are particularly attractive for their CMOS compatibility, low cost and heterogenous integration [3]. Recent 
demonstrations of Si MZMs have achieved 363 Gb/s (O-band, PAM8) [4] and 350 Gb/s (C-band, PAM4) [5] 
operation with 47 GHz and 42 GHz 3-dB bandwidth, respectively. Although slow-light modulator already 
achieved bandwidth higher than 110 GHz, the narrow transmission window characteristics and poor fabrication 
tolerance make it difficult to be implemented in 400 G/lane applications [6]. 
     In this paper, we designed a differential-drive high-speed Si MZM.  The device was fabricated at Advanced 
Micro Foundry (AMF) using the standard High-Performance platform with LPCVD Silicon Nitride (SiN) on 200-
mm SOI wafer. The modulator was designed with 50 Ω characteristic impedance for both metal trace and on-chip 
termination resistor. The MZM exhibits a 3-dB bandwidth of 81.8 GHz in the C-band. Despite the constraints of 
the current testing setup, a PAM-8 eye-diagram was obtained at 100 GBaud (300 Gbps/lane). To further improve 
the eye-diagram quality at higher baud rate, modulation efficiency improvement is needed. A high-efficiency 
MZM was designed and fabricated for this purpose, achieving PAM-8 eye-diagram at 112 Gbaud (336 G/lane). 
The results demonstrate strong potential for 400 G/Lane optical interconnect applications.   
 
Fig. 1. (a) 3D schematic of the designed Si MZM. (b) 200 mm SOI wafer fabricated at AMF using LPCVD platform. (c) Microscope image 
of the fabricated device with differential-drive input and 50 Ω terminator. (d) Measured electro-optical response showing 3-dB bandwidth 
of 81.8 GHz. (e) Eye-diagram testing setup. (f) PAM-8 eye diagram at 100 Gbaud. 
         
 
 
 
 
 
   
          
   
    
      
  
   
   
      
        
      
       
      
       
      
 
-
               
 
 
                  
                 
             
              
   
   
   
            
               
   
   
W3E.5.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 2 -->
 
 
2. Design, fabrication, and performance 
The demonstrated differential-drive modulator adopts a travelling-wave electrode (TWE) configuration. The 
device performance is fundamentally limited by several factors, including RC delay, carrier response of the PN 
junction, and microwave propagation effects, such as impedance mismatch, RF loss, and velocity mismatch. To 
achieve broad electro-optic bandwidth, the electrode geometry and doping profile were optimized to ensure 50 Ω 
impedance matching, low RF attenuation, and phase velocity alignment with the optical mode. The slab-type 
waveguide is employed to reduce insertion loss, while the PN junction is created through P- and N-type 
implantation. To balance RC delay and optical insertion loss, medium-level doping (N+ and P+) is utilized between 
the N and N++ (N contact) and P and P++ (P contact). The total length of MZM is 2.5 mm with effective modulation 
length of 2.25 mm. The doping concentration and junction profile were optimized to balance the trade-off between 
insertion loss (IL) and product of half-wave voltage and length (Vπ·L). Fig. 1. (a) shows the 3D image of the 
modulator with two-layer metal line and differential-drive structure. The device was fabricated with standard 
LPCVD SiN-on-SOI technology in AMF, and Fig. 1. (b) shows the image of a 200 mm SOI wafer. Fig. 1. (c) is 
the overall microscope image of the modulator with differential-drive input and 50 Ω terminator. For modulation 
efficiency characterization, the optical transmission spectrum was measured under different bias voltages and the 
phase shift versus bias voltage relationship was extracted. The resulting Vπ·L was determined to be 2.3 V·cm. The 
electro-optical (EO) performance was characterized using Lightwave Components Analyzer (LCA, Keysight 
N4372B), and the extracted 3-dB bandwidth reached 81.8 GHz as shown in Fig.1. (d). The high-speed eye diagram 
was also evaluated using the setup illustrated in Fig. 1. (e). A continuous-wave (CW) laser was coupled into the 
device, while the RF signal was generated by a Keysight M8199A arbitrary waveform generator (AWG) with 
sampling rate of 128 GSa/s and amplified by a driver before being applied to the modulator. The modulated optical 
signal was amplified by an PDFA and directly captured by a 56 GHz Photodetector (PD) and oscilloscope. The 
low sampling rate of AWG and low bandwidth of PD used in this setup limit the maximum data rate and the 
observed eye-opening. In addition, the driving voltage from AWG is only around 2.5 V for data rate of 100 Gbaud 
or higher. As a result, only a 100 Gbaud PAM-8 eye diagram could be recorded under these non-optimized 
conditions, as shown in Fig. 1. (f). Another factor limiting the eye-diagram performance is the relatively large 
Vπ·L, corresponding to a half-wave voltage (Vπ) of approximately 10.2 V under DC test condition. As the 
frequency increases, the effective Vπ will further rise, which consequently degrades the eye quality and limits the 
achievable data rate. 
     To enhance the modulation performance at higher baud rates, higher modulation efficiency (lower Vπ·L) is 
required towards 400 G per lane. Therefore, a differential-drive modulator with improved efficiency (Vπ·L of 1.36 
V·cm) was studied for verification. The Vπ·L was lowered through junction engineering and optimized optical 
confinement for reduced insertion loss. The optical transmission spectrum was measured under different bias 
voltage in Fig. 2. (a) and Vπ·L was extracted to be 1.36 V·cm, corresponding to a Vπ of approximately 6 V. The 
high-speed EO performance was characterized using LCA. Even though with lower bandwidth around 60 GHz 
[Fig. 2. (b)], a measured eye (PAM-4 at 100 Gbaud, 4.8E-4 SER) remains open with an extinction ratio of 4.2 dB 
and a TDECQ of 2.6 dB [Fig. 2. (c)], while PAM-8 eye diagrams at 100 Gbaud and 112 Gbaud were recorded in 
Fig. 2. (d-e). A preliminary 8-channel circuit was also developed for co-evaluation for data center networks [Fig. 
 
Fig. 2. (a) The optical transmission spectrum. (b) The measured bandwidth. (c-e) The eye diagram of PAM-4 at 100 Gbaud, PAM-8 at 
100 Gbaud, and PAM-8 at 112 Gbaud. (f) the microscope image of the fabricated DR8 transceiver circuit. 
   
   
   
   
   
   
              
              
              
W3E.5.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

<!-- page 3 -->
 
 
2. (f) ]. Despite the lower bandwidth, PAM-8 eye diagram at 112 Gbaud was achieved as compared to the higher 
bandwidth device (81.8 GHz), which indicates that data rate of the 81.8 GHz modulator can be further improved 
with a lower Vπ·L. In future work, we plan to further reduce the Vπ·L and enhance the measurement setup 
capability by applying higher bandwidth source and digital signal processing, to achieve higher-quality eye 
diagrams and validating higher data rates.  A comparative benchmark in Table I was conducted in both the O-band 
and C-band operation windows. The proposed modulator exhibits the high 3-dB bandwidth among reported 
devices in both bands. In the O-band, it also demonstrates the low Vπ·L while maintaining low insertion loss.   
3. Conclusion 
In this work, we demonstrated a high-speed Si MZM fabricated on a standard silicon photonics platform. The EO 
response exhibited a 3-dB bandwidth of 81.8 GHz. PAM-8 eye-diagram at 100 Gbaud was achieved despite of an 
unoptimized testing setup. The demonstrated performance indicates strong potential for data rates beyond 400 
G/Lane optical interconnects. 
4. References 
[1] Jacques M, “200 Gbit/s net rate transmission over 2 km with a silicon photonic segmented MZM. ” 45th European Conference on 
Optical Communication (ECOC 2019). IET, 2019. 
[2] IEEE P802.3dj 200 Gb/s, 400 Gb/s, 800 Gb/s, and 1.6 Tb/s Ethernet Task Force, http://www.ieee802.org/3/dj/, accessed on 23 Oct. 2023 
[3] J. Witzens, “High-speed silicon photonics modulators,” Proc. IEEE, vol. 106, no. 12, pp. 2158–2182, Dec. 2018. 
[4] M. S. Alam, et. al., “Net 300 Gbps/λ Transmission Over 2 km of SMF With a Silicon Photonic Mach-Zehnder Modulator,” IEEE 
Photonics Technol. Lett. 33, 1391-1394 (2021).  
[5] A. Ostrovskis et al., “Traveling-Wave Silicon Photonics Mach-Zehnder Modulator for Beyond 350 Gb/s Transmission in C-band,” OFC, 
2025, pp. 1-3.  
[6] Changhao Han et al., “Slow-light silicon modulator with 110-GHz bandwidth. ” Sci. Adv. 9,eadi5339(2023)  
[7] Dong P et al., “High-speed low-voltage single-drive push-pull silicon Mach-Zehnder modulators,” Opt. Express 20, 6163-6169 (2012). 
[8] J. Fujikata et al., “25 GHz operation of silicon optical modulator with projection MOS structure,” Optical Fiber Communication 
Conference, paper OMI3 (2010).  
[9] L. Liao et al., “40 Gbit/s silicon optical modulator for high speed applications,” Electron. Lett. 43, 1196–1197 (2007).  
[10] F. Y. Gardes et al., “40 Gb/s silicon photonics modulator for TE and TM polarisations,” Opt. Express 19, 11804–11814 (2011).  
[11] D. Thomson et al., “50Gbit/s silicon optical modulator,” IEEE Photon. Technol. Lett. 24, 234–236 (2012).  
[12] D. J. Thomson et al., “High contrast 40Gbit/s optical modulation in silicon,” Opt. Express 19(12), 11507–11516 (2011).  
[13] L. Chen et al., “Monolithic silicon chip with 10 modulator channels at 25 Gbps and 100-GHz spacing,” in 37th European Conference 
and Exposition on Optical Communications, (Optical Society of America, 2011), paper Th.13.A.1.  
[14] Miaofeng Li et al., “Silicon intensity Mach–Zehnder Modulator for single lane 100 Gb/s applications, ” Photon. Res. 6, 109-116 (2018). 
[15] A. Samani et al., “Silicon photonic Mach–Zehnder Modulator architectures for on chip PAM-4 signal generation,” J. Lightw. Technol. 
37(13), 2989–2999 (2019). 
[16] M. Jacques et al., “Net 212.5 Gbit/s transmission in O-band with a SiP MZM, one driver and linear equalization,” in Proc. Opt. Fiber 
Commun., San Diego, CA, Mar. 2020, Paper Th4A.3. 
[17] J. X. B. Sia et al., “Wafer-Scale Demonstration of Low-Loss (∼0.43 dB/cm), High-Bandwidth (>38 GHz), Silicon Photonics Platform 
Operating at the C-Band,” IEEE Photonics Journal 14, 1-9 (2022). 
Table 1. The benchmark of Si based MZM in terms of VpiL, insertion loss and high-speed performance.  
 
Speed
(Gb/s)
3-dB bandwidth
(GHz)
Insertion loss 
(dB)
VpiL
(V*cm)
Length 
(mm)
Single
/differential 
O/C
50 (NRZ)
-
4.1
2.4
2
Single
O
[7]
-
25
2.5
0.5
0.12
Single
O
[8]
40(NRZ)
30
4
4
1
Single
O
[9]
40 (NRZ)
~20
15
11
1.35
Single
O
[10]
40 (NRZ)
-
3.7
2.8
1
Single
O
[11]
40 (NRZ)
-
15
2.7
3.5
Differential
O
[12]
30 (NRZ)
-
4.1
2.4
2.4
Single
O
[13]
128 (PAM4)
58
5.4
1.4
2
Single
O
[14]
112 (PAM4)
35
3.4
-
1.5
Single
O
[15]
200 (PAM8)
47
5.7
1.728
3.2
Single
O
[16]
336 (PAM8)
44
4.6
1.1
3
Single
O
[4]
362 (PAM6)
42
3.6
1.7
1.5
Differential
C
[5]
38
0.43 dB/cm
single
C
[17]
300 (PAM8)
81.8
2.6
2.3
2.5
Differential
C
This work
336 (PAM8)
59.5
2.45
1.36
2.5
Differential
O
W3E.5.pdf
Optical Fiber Communication Conference (OFC)  ©
2026 Optica Publishing Group

