---
paper_id: derose2012
source_url: https://doi.org/10.1109/oic.2012.6224486
doi: 10.1109/oic.2012.6224486
license: 
sha256: a12b26e5dbef9d259ee8a9a02a998037305b746ec3be96621aaf48018b20d221
pages: 2
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
High Speed Travelling Wave Carrier Depletion Silicon 
Mach-Zehnder Modulator 
 
Christopher T. DeRose1, Douglas C. Trotter1, William A. Zortman1 and Michael R. Watts2 
1Applied Photonic Microsystems 
Sandia National Labs, P.O. Box 5800, Albuquerque, NM 8718, USA 
2Massachusetts Institute of Technology Research Laboratory of Electronics  
77 Massachusetts Avenue, Cambridge, Massachusetts 02139, USA 
cderose@sandia.gov 
 
Abstract: We present the first demonstration of a travelling wave carrier depletion Mach-Zehnder 
modulator impedance matched to 50 This device has a bandwidth of 24 GHz and a halfwave 
voltage length product of 0.7 V-cm, placing it among the best in its class.   
 
1. Introduction  
Over the last 25 years optical links have grown from a solution for long haul telecommunication to a solution for 
data communication over distances of a few tens to hundreds of meters.  As optical data communications systems 
become the solution for tens of meter and shorter distance links, the cost per Gbps will ultimately become the 
driving performance metric.  Due to its high yield and the high volume manufacturing infrastructure, silicon is an 
excellent candidate for future low cost optical data communications device components.  Here we present a high 
speed, travelling wave silicon carrier depletion Mach-Zehnder modulator (MZM). 
 
2. Device Fabrication and Theory 
The MZM was fabricated on a silicon-on-insulator (SOI) wafer with 3 m of buried oxide and an initial 250 nm 
thick silicon layer.  The devices were defined using an ASML deep ultra-violet (DUV) laser scanner and silicon 
etch.  Arsenic and Phosphorous implants were used to achieve an n-type and p-type doping level of ~5×1018/cm3 in 
the sp-n junction.  Electrical contact from the device to aluminum bus lines was made through tungsten vias.  A 
cross-section of the active part of the device can be seen in Fig. 1.   
 
Fig. 1 a) Schematic of the capacitively loaded slot line b) push-pull carrier depletion MZM with no applied RF signal, the depletion region is 
represented by white, a contour plot of the Ex field for the mode of each arm of the device is also shown c) with an applied RF voltage, the 
effective index in one arm of the device is increased (depletion region widens) while it is decreased in the other (depletion region shrinks). 
 
    There have been several previous demonstrations of silicon MZMs [1-3].  Previously reported devices have had 
low impedance due to the high capacitance of the p-n junction and relatively high V-L products.  The MZM we 
present here was designed as a push-pull segmented travelling wave device.  Using this approach we were able to 
able to achieve both impedance matching to 50  and velocity matching to the waveguide mode for the first time.   
     In a segmented design, the active p-n junctions are attached periodically to a high impedance travelling wave 
electrode.  By periodically adding the capacitive p-n junctions both impedance and velocity matching can be 
135
WC6 (Contributed Oral)
15:15 – 00:00
978-1-4577-1619-5/12/$26.00 ©2012 IEEE

<!-- page 2 -->
achieved simultaneously [4].  The simultaneous impedance and velocity matching can be seen with a simple 
analysis.  The impedance and microwave velocity index of an unloaded electrode are given by 
 
 
 
 
  

	 and, 
  	 
 
 
 
 
 (1) 
where, Z0 is the impedance of the unloaded line, n0 is the microwave velocity index, L0 is the inductance of the 
unloaded line, C0 is the capacitance of the unloaded line and c is the speed of light.  Adding additional capacitance 
results in 
 
 
 
        

	 and, 
    	 
 
 
 
 (2) 
where, ZL is the impedance of the loaded line, nL is the microwave velocity index of the loaded line and CL is the 
loading capacitance.  By requiring 
  
 where ZL is 50 and nL is the group index of the optical mode 
achieves impedance matching and by simultaneously enforcing   

  


 velocity matching is 
achieved.   
     The unloaded travelling wave electrode was a slot line with an impedance of 95  and was fabricated in metal1 
which is a 1m Ti/TiN/AlCu/TiN stack the slot line had a microwave index of 2.3.  The p-n junctions which were 
connected in series had a capacitance of 0.41 fF/m at 0V bias .  In order to achieve velocity matching to the optical 
mode which has a group index of 4.5 we fabricated our device with 50 m active segments and a fill factor of 0.6.   
 
 
3. Experimental Results 
The electrical S-parameters and modulator bandwidth for MZMs with effective active lengths of 0.5, and 1.5 mm 
were measured with an Agilent E8364B vector network analyzer.  We found 3dB bandwidths of 24 GHz and 14 
GHz for the 0.5mm and 1.5 mm long modulators respectively.  We found that the devices had better than 20 dB 
return loss up to 30 GHz, showing excellent impedance matching to 50 Furthermore, a half-wave voltage of 0.7 
V-cm was measured for a 2 mm long modulator.   
 
Fig. 2 a) measured bandwidth of 1.5 and 0.5 mm capacitively loaded MZM b) electrical S11 of same modulators showing better thand 20 dB 
return loss up to 30 GHz.   
 
4. Conclusions 
We have demonstrated a capacitively loaded push-pull travelling wave carrier depletion silicon MZM which was 
impedance matched to 50 for the first time.  We measured a bandwidth of 24 GHz for a 0.5 mm modulator and 14 
GHz for a 1.5 mm modulator placing the bandwidth of this device among the best in its class.  Furthermore, a VL 
of 0.7 V-cm was measured for a 2 mm modulator, which to the best of the authors knowledge is the best yet reported 
for this class of modulator.  Finally, the device bandwidth is currently limited by series resistance in the p-n junction 
and can be improved in future designs.   
 
Sandia National Laboratories is a multi-program laboratory managed and operated by Sandia Corporation, a wholly owned subsidiary of 
Lockheed Martin Corporation, for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-AC04-
94AL85000. 
 
5. References 
[1] A.S. Liu, et al., “A high –speed silicon optical modulator based on a metal-oxide semiconductor capcitor,”Nature, 427 615-618 (2004). 
[2] A. Liu, et al., “High-speed optical modulation based on carrier depletion in a silicon waveguide”, Opt. Expr.15  660-668 (2007). 
[3] M. R. Watts, et al., “Low-Voltage, Compact, Depletion-Mode, Silicon Mach-Zehnder Modulator,” JSTQE 16  159-164 (2010). 
[4] G.L. Li, et al., “Analysis of Segmented Traveling-Wave Modulators,” J. Lightwave Technol. 22 1789-1796 (2004). 
a) 
b) 
136

