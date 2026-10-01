---
paper_id: sayem2026c
source_url: https://arxiv.org/abs/2604.09825
doi: 
license: CC-BY-4.0
sha256: 466131bc6c31c464e89796790538cad059c331b4fbcc84f7e2b66344f065bd87
pages: 6
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
High bandwidth traveling wave electro-optic modulator at 1µm on thin-film lithium
tantalate
Ayed Al Sayem, Shiekh Zia Uddin, Ting-Chen Hu, Alaric Tate, Mark Cappuzzo, Rose Kopf, Mark Earnshaw
Nokia Bell Labs, NJ, USA
(Dated: April 14, 2026)
We present the first experimental demonstration of a high-bandwidth thin-film lithium tantalate
(TFLT) electro-optic modulator operating at 1 µm, with a Vπ of 2.4 V, and less than 2 dB electro-
optic roll-off upto 50 GHz and stable DC bias operation.
INTRODUCTION
The recent advancement of hollow-core fibers now
opens up the opportunity to utilize a wide range of
wavelengths, such as the visible wavelength band and
also the non-conventional near IR wavelengths bands,
such as 780 nm, 850 nm and 1 µm with lower optical
losses than achievable with standard glass core fiber [1–7].
Hollow-core fibers can also operate at much higher optical
power due to very weak nonlinearity [2, 8, 9], potentially
enabling the operation of the optical transceiver with
very high optical power and potentially eliminating or
significantly reducing the number of line amplifiers used
many communication networks. To take full advantage
of hollow-core fibers, one requires a material system with
low optical loss, strong electro-optic effect, high power-
handling capability, and stable operation. Unfortunately,
the most explored and commercially utilized materials,
such as Silicon (Si) or Indium phosphide (InP), have
a band-gap of 1.1 µm and 0.9 µm respectively, which
makes these platforms not ideal to take advantage
of the transparency window of the hallow core fibers
below the O-band.
Among the transparent materials
with a wide optical window, lithium niobate (LN) and
lithium tantalate (LT) are particularly promising because
they offer broadband transparency along with a strong
electro-optic effect [10, 11]. The cut-off wavelengths for
LN and LT are 350 nm [10], and 316 nm respectively
[11].
Thin-film lithium niobate (TFLN) has already
been widely explored in the visible and near-infrared
range for applications such as generating efficient second-
harmonics [12? , 13], entangled photon pairs [14], optical
parametric generation [15], and low-loss resonators [16],
and so on. TFLN electro-optic modulators in the visible
and near-infrared regimes have been widely explored.
For example,
TFLN modulators have already been
demonstrated at 532 nm [17], 780 nm [18], 850 nm [19],
and at 1064 nm [20].
The fundamental advantage of
operating in such a wavelength range comes directly
from the tighter mode confinement, which allows closer
electrode spacing,
which in turn reduces the drive
voltage [19, 21].
The second fundamental advantage
originates purely from physics, as the phase accumulation
per length is inversely proportional to the operating
wavelength.
Unfortunately, TFLN suffers significantly
from the well-known photo-refractive (PR) effect [22–
26].
Because of the PR effect, TFLN is not DC-bias
stable [27, 28], requiring high power thermal control,
which limits its practical utility for large-scale photonic
circuits.
Critically, the PR effects are more drastic at
visible and near-infrared wavelengths as carriers from
defect-related trap centers such as Fe2+, and Fe3+ ions
as well as other intrinsic defects can be easily photo-
excited at these wavelength bands, which have been
extensively studied for bulk LN [29–32]. For a practical
communication system, one needs a material system
that not only offers high performance but also maintains
stable performance. We recently showed that thin-film
lithium tantalate (TFLT) is a highly stable material
platform for optical power handling and outperforms
most material platforms, such as Si, LN, Ta2O5, etc.,
in the C-band. For short-wavelength operation, we also
expect TFLT to be a better choice of material platform
than TFLN [21, 33] due to the stronger PR effect at the
shorter wavelength bands [34]. Unfortunately, TFLT is
not as explored as TFLN, and there have been very few
experimental demonstrations [11, 35, 36], especially at
wavelengths beyond the telecommunication band [21, 33].
Here, we show the first experimental demonstration of
a high-speed (> 50 GHz electro-optic bandwidth with
< 2 dB roll-off) modulator with a Vπ of 2.4 V on a
standard 600 nm TFLT 4-inch wafer platform with a
wafer-level fabrication process flow. We also show that
our device is DC-bias stable even at 1 µm and can
generate sharp pulses without distortion even with very
wide pulses.
DEVICE GEOMETRY
In Fig.1, we show the schematic of the traveling
wave electro-optic modulator used in this article. The
modulator consists of three cascaded Mach-Zehnder
interferometers (MZIs). The first and the third MZI act
as tunable splitters so that we can operate the modulator
at a wide range of wavelengths from 1.5 µm to near 1 µm.
Fig.1(b) shows the cross-section of different sections
of the device, the directional coupler (DC), modulator
section, and heater section, respectively. Fig.1(c) shows
the optical microscope image of the modulator.
The
arXiv:2604.09825v1  [physics.optics]  10 Apr 2026

<!-- page 2 -->
2
FIG. 1. (a) Schematic of the modulator device. G: ground, S: signal (b) Cross-section of the modulator in three different
sections of the device, i) the directional coupler (DC), ii) MZI section, iii) waveguide with NiCr heater for thermal tuning. Here,
LMZI = 7 mm, , wDC = 0.8 µm, g = 0.8 µm, tf = 600 nm, td =∼240 nm, tm = 1 µm, ws = 17 µm, wMZI = 1.6 µm, wNiCr = 10 µm.
(c) Optical microscope image of the device. (d) Extinction ratio of the modulator as a function of the heater power with
identical heating power for two heaters.
(e) Two-dimensional (2D) color of extinction as a function of heater power.
(f)
Normalized transmission of the modulator as a function of applied triangular voltage at 100 Hz.
device has been fabricated on a 4-inch 600 nm thick TFLT
wafer with 4.7µm thick insulating oxide underneath,
commercially available from NanoLN. We use on-chip
50 Ωtermination for traveling-wave operation.
Details
of the device fabrication can be found in our previous
articles [37, 38]. By controlling the phase of the first and
the third MZI, e.g., the tunable splitters, we can control
the beam splitting ratio over a wide range of wavelengths.
Fig.1(d) shows the extinction ratio of the modulator as
a function of the heater power when identical heating
power is applied to both heaters.
Fig.1(e) shows the
two-dimensional color map of the extinction ratio when
the heater power was swept for both heaters. It can be
observed from Fig.1(e) that close to 30 dB extinction can
be achieved, which is limited by the dynamic range of the
photo-detector (PD) used in the experiment. In Fig.1(f),
we show the normalized transmission of the device as a
function of applied voltage. All these measurements are
performed at λ = 1071 nm.
MODULATOR PERFORMANCE
We characterize the modulator performance at relevant
wavelengths for a systematic comparison. The schematic
of the measurement setup is shown in Fig.2(a).
Light
from either a diode laser or a tunable laser is sent
to the device under test (DUT) through a fiber-
based polarization controller and a lensed fiber.
We
use different lasers to test the modulator at different
wavelengths. For λ = 984 nm, we use a fixed wavelength
semicondutor laser. For λ = 1071 nm and λ = 1551 nm,

<!-- page 3 -->
3
FIG. 2. Transmission as a function of applied voltage for four different wavelengths of operation, λ = 984 nm, λ = 1071 nm,
λ = 1311 nm, and λ = 1551 nm. (b) VπL as a function of operating wavelength.
we use tunable lasers from Santec (Santec-570), and
for λ = 1311 nm, we use a tunable laser from EXFO
(model). The output light from the device is collected
by another lensed fiber and is then sent to a slow power-
meter (PM) and a fast photo-detector (PD). Fig.2(b)
shows the normalized transmission as a function of
applied voltage for the same modulator at different
wavelengths.
In Fig.2(b), we plot the voltage-length
(VπL) product as a function of wavelength.
Clear
wavelength scaling can be observed in Fig.2(c).
Here,
we use the same device with an electrode gap, G = 5 µm.
For shorter wavelengths, it’s possible to use a smaller
electrode gap and improve Vπ (VπL) [21].
We also
completed a radio-frequency (RF) characterization of the
modulator. In Fig.3(a), we plot the RF reflection, S11
of the modulator.
We measure the RF transmission
spectra of the modulator on a separate device without
termination resistors but with identical geometry.
In
Fig.3(b), we show the calculated microwave phase index
from the RF transmission measurements along with
the optical group index at different wavelengths.
In
Fig.3(c), we plot the simulated optical phase index,
no, and group index, ng as a function of wavelength
for TFLT waveguides for the geometry used in this
paper.
From Fig.3(c), we can observe that TFLT
waveguides offer a flatter dispersion, which indicates
the possibility of high-bandwidth operation over a wide
range of wavelengths simultaneously.
Due to the lack
of high-speed photo-detectors, we measure the electro-
optic (EO) bandwidth of the modulator at 1 µm using
an optical spectrum analyzer (OSA). Fig.3(d) shows the
schematic of the measurement setup and the measured
EO response as a function of the RF drive frequency at
two different wavelengths, e.g., at 1071 nm and 1551 nm.
Almost identical EO roll-off can be observed for both
wavelengths, proving the ultra-flat dispersion of LT. Less
than 2 dB roll-off can be observed for RF frequency up
to 50 GHz.
We
next
investigate
the
stability
of
the
TFLT
modulators near λ = 1 µm. TFLN is widely known for
its DC bias instability [27, 39–41], and TFLT modulators
have already been shown to have much better DC bias
stability for both waveguide and resonant modulators
[28, 38]. Most of these experiments have been performed
near the C-band.
The PR effect is much stronger at
shorter wavelengths, such as in the visible and near-
infrared, hence we study the DC-bias stability and pulse
generation at λ = 1 µm. Fig.4(a) shows the output power
of the TFLT modulator as a function of time when the
modulator is biased at the quadrature point. Negligible
drift is observed over more than one hour with an on-chip
power of ∼0 dBm, showing excellent DC bias stability of
the modulator.
Sharp pulse generation is another key
characteristic of a stable modulator. Because of the PR
effect and defect-assisted charge dynamics, with TFLN
modulators, it’s difficult to generate sharp pulses [42].

<!-- page 4 -->
4
FIG. 3. (a) Experimental S11 of the TFLT modulator. (b) Extracted RF phase index as a function of RF frequency. (c)
Simulated optical phase and group index of TFLT waveguides as a function of wavelength.
Here, the waveguide width is
1.6 µm.
(d) Schematic of the measurement setup and electro-optic (EO) response of the modulator as a function of RF
frequency at two operating wavelengths.
Fig.4(b)–(d) show the applied voltage and corresponding
output pulse from the modulator when square-wave
signals at 10 Hz, 100 Hz, and 1 kHz, respectively, are sent
the modulator. The output waveform faithfully tracks
the input waveform, preserving sharp pulse edges without
observable distortion, which suggests that charge-related
effects are negligible in the device. These results confirm
that our modulator can generate sharp pulses at 1 µm at
low frequencies where charge related distortions are most
pronounced, detailed results on achieving high extinction
ratio pulses in TFLT devices will be presented in a future
work.
ACKNOWLEDGMENT
We thank Santec Corporation for providing us the
TSL-570 at 1 µm.
AUTHOR CONTRIBUTION
A.S. designed the photonic devices and developed the
fabrication process flow with T.H, A.T, M.C., and R. K..
T.H, A.T, M.C., and R. K. fabricated the devices. A.S
and S.U performed the measurements. A.S, S.U wrote
the paper with technical feedback from M.E.
FUNDING
Nokia Corporation of America.
[1] Marco Petrovich, Eric Numkam Fokoua, Y. Chen, et al.
Broadband optical fibre with an attenuation lower than
0.1 decibel per kilometre.
Nature Photonics, 19:1203–
1208, 2025.

<!-- page 5 -->
5
FIG. 4. (a) Normalized Transmission (NT) of the MZI modulator as a function of time when the modulator is biased at the
quadrature point. Applied voltage and corresponding output pulse from the modulator for square-wave signals at (b) Ω= 10 Hz,
(c) 100 Hz, and (d) 1 KHz.
[2] Jing
Shi,
Binyu
Rao,
Zilun
Chen,
Zefeng
Wang,
Guangrong Sun, Zhen Huang, Min Fu, Xin Tian, Baolai
Yang, Jian Zhang, Zhiyue Zhou, Tianyu Li, Chenxin
Gao, Jinbao Chen, Zuying Xu, Biao Shui, Peng Li, Zihan
Dong, and Lei Zhang. All-fiber highly efficient delivery
of 2 kw laser over 2.45 km hollow-core fiber.
Nature
Communications, 16:8965, 2025.
[3] Shoufei Gao, Hao Chen, Yizhi Sun, Yifan Xiong, Zijie
Yang, Rui Zhao, Wei Ding, and Yingying Wang. Fourfold
truncated double-nested antiresonant hollow-core fiber
with ultralow loss and ultrahigh mode purity.
Optica,
12(1):56–61, 2025.
[4] Yukun Wan, Min Xia, Zhehan Wang, Li Xia, Peng Li,
Lei Zhang, and Wei Li. Anti-resonant hollow core fiber
with excellent bending resistance in the visible spectral
range. Optics Express, 32(8):14659–14673, 2024.
[5] Dawei Ge, Siyuan Liu, Peng Li, Qiang Guo, Yifan
Xiong, Mingqing Zuo, Dong Wang, Shoufei Gao, Dechao
Zhang, Da Liu, Yingying Wang, Lei Zhang, Wei Ding,
Jie Luo, Hongqiang Zou, Han Li, Zhangyuan Chen, and
Xiaodong Duan.
Field trial of real-time 128tb/s co-
frequency co-time full-duplex transmission over deployed
20km ar-hcfs in urban duct network.
In Optical
Fiber Communication Conference (OFC) 2025, Technical
Digest. Optica Publishing Group, 2025. Paper W1C.4.
[6] Dawei Ge, Yifan Xiong, Yan Wu, Yizhi Sun, Yancai
Luan, Dong Wang, Shoufei Gao, Dechao Zhang, Liang
Mei, Yingying Wang, Wei Ding, Han Li, and Zhangyuan
Chen. First penalty-free real-time co-frequency co-time
full-duplex optical fiber transmission with 202.1tb/s net
capacity enabled by hollow-core 5-element nanf.
In
Optical Fiber Communication Conference (OFC) 2024,
Technical Digest. Optica Publishing Group, 2024. Paper
M3J.2.
[7] Kazunori Mukasa and Takeshi Takagi. Hollow core fiber
cable technologies. Optical Fiber Technology, 80:103447,
2023.
[8] M. A. Cooper, J. Wahlen, S. Yerolatsitis, D. Cruz-
Delgado, D. Parra, B. Tanner, P. Ahmadi, O. Jones,
Md. S. Habib,
I. Divliansky,
J. E. Antonio-Lopez,
A. Sch¨ulzgen, and R. Amezcua Correa. 2.2 kw single-
mode narrow-linewidth laser delivery through a hollow-
core fiber. Optica, 10(10):1253–1259, 2023.
[9] Mattia Michieletto, Jens K. Lyngsø, Christian Jakobsen,
Jesper Lægsgaard, Ole Bang, and Thomas T. Alkeskjold.
Hollow-core fibers for high power pulse delivery. Optics
Express, 24(7):7103–7119, 2016.
[10] Di Zhu, Linbo Shao, Mengjie Yu, Rebecca Cheng, Boris
Desiatov, CJ Xin, Yaowen Hu, Jeffrey Holzgrafe, Soumya
Ghosh, Amirhassan Shams-Ansari, et al.
Integrated
photonics on thin-film lithium niobate.
Advances in
Optics and Photonics, 13(2):242–352, 2021.
[11] Chengli Wang, Zihan Li, Johann Riemensberger, Grigory
Lihachev,
Mikhail
Churaev,
Wil
Kao,
Xinru
Ji,
Junyin Zhang, Terence Blesin, Alisa Davydova, et al.
Lithium tantalate photonic integrated circuits for volume
manufacturing. Nature, 629(8013):784–790, 2024.
[12] Cheng Wang, Carsten Langrock, Alireza Marandi, Marc
Jankowski, Mian Zhang, Boris Desiatov, Martin M Fejer,
and Marko Lonˇcar.
Ultrahigh-efficiency wavelength
conversion in nanophotonic periodically poled lithium
niobate waveguides. Optica, 5(11):1438–1441, 2018.
[13] Ayed Al Sayem, Yubo Wang, Juanjuan Lu, Xianwen
Liu, Alexander W Bruch, and Hong X Tang. Efficient
and
tunable
blue
light
generation
using
lithium
niobate nonlinear photonics.
Applied Physics Letters,
119(23):231104, 2021.
[14] Zhaohui Ma, Jia-Yang Chen, Zhan Li, Chao Tang,
Yong Meng Sua,
Heng Fan,
and Yu-Ping Huang.
Ultrabright quantum photon sources on chip. Physical
Review Letters, 125(26):263602, 2020.
[15] Rajveer Nehra, Ryoto Sekine, Luis Ledezma, Qiushi Guo,
Robert M Gray, Arkadev Roy, and Alireza Marandi.
Few-cycle vacuum squeezing in nanophotonics. Science,
377(6612):1333–1337, 2022.
[16] Mian Zhang, Cheng Wang, Rebecca Cheng, Amirhassan
Shams-Ansari, and Marko Lonˇcar.
Monolithic ultra-
high-q lithium niobate microring resonator.
Optica,
4(12):1536–1537, 2017.
[17] Chijun Li, Bin Chen, Ziliang Ruan, Pengxin Chen,
Kaixuan Chen, Changjian Guo, and Liu Liu.
High
modulation efficiency and large bandwidth thin-film
lithium niobate modulator for visible light.
Optics
Express, 30(20):36394–36402, 2022.

<!-- page 6 -->
6
[18] Oguz Tolga Celik, Christopher J. Sarabalis, Felix M.
Mayor,
Hubert S.
Stokowski,
Jason
F.
Herrmann,
Timothy P. McKenna, Nathan R. A. Lee, Wentao Jiang,
Kevin K. S. Multani, and Amir H. Safavi-Naeini. High-
bandwidth CMOS-voltage-level electro-optic modulation
of 780 nm light in thin-film lithium niobate.
Optics
Express, 30(13):23177–23186, 2022.
[19] Daniel Assumpcao, Dylan Renaud, Amirhassan Shams-
Ansari, and Marko Loncar. High-speed short-wavelength
communications
utilizing
thin-film
lithium
niobate.
Optics Letters, 50(5):1473–1475, 2025.
[20] Navarun Jagatpal, Andrew J. Mercante, Abu Naim R.
Ahmed, and Dennis W. Prather.
Thin film lithium
niobate electro-optic modulator for 1064 nm wavelength.
IEEE Photonics Technology Letters, 33(5):271–274, 2021.
[21] Keith
Powell,
Dylan
Renaud,
Xudong
Li,
Daniel
Assumpcao, CJ Xin, Neil Sinclair, and Marko Lonˇcar.
A
sub-volt
near-ir
lithium
tantalate
electro-optic
modulator. APL Photonics, 10(9), 2025.
[22] Yuntao Xu,
Mohan Shen,
Juanjuan Lu,
Joshua B
Surya, Ayed Al Sayem, and Hong X Tang. Mitigating
photorefractive
effect
in
thin-film
lithium
niobate
microring resonators. Optics Express, 29(4):5497–5504,
2021.
[23] R. Ahmed, S. R. Baghdadi, M. Bernadskiy, N. Bowman,
R. Braid, J. Carr, C. Chen, P. Ciccarella, M. Cole,
J. Cooke, K. Desai, C. Dorta, J. Elmhurst, and et al.
Universal photonic artificial intelligence acceleration.
Nature, 640(8058):368–374, 2025.
[24] Xinyi Ren, Chun-Ho Lee, Kaiwen Xue, Shaoyuan Ou,
Yue Yu, Zaijun Chen, and Mengjie Yu. Photorefractive
and
pyroelectric
photonic
memory
and
long-term
stability in thin-film lithium niobate microresonators. npj
Nanophotonics, 2(1):1, 2025.
[25] Mengjie Li, Hanxiao Liang, Rui Luo, Yang He, Haowei
Jiang, and Qiang Lin. Photon-level tuning of photonic
nanocavities. Optica, 6(7):860–863, 2019.
[26] Tummas Napoleon Arge, Seongmin Jo, Huy Quang
Nguyen,
Francesco
Lenzini,
Emma
Lomonte,
Jens
Arnbak
Holbøll
Nielsen,
Renato
R
Domeneguetti,
Jonas Schou Neergaard-Nielsen, Wolfram Pernice, Tobias
Gehring, et al. Demonstration of a squeezed light source
on thin-film lithium niobate with modal phase matching.
Optica Quantum, 3(5):467–473, 2025.
[27] Mengyue Xu, Mingbo He, Hongguang Zhang, Jian Jian,
Ying Pan, Xiaoyue Liu, Lifeng Chen, Xiangyu Meng,
Hui Chen, Zhaohui Li, et al. High-performance coherent
optical modulators based on thin-film lithium niobate
platform. Nature communications, 11(1):3911, 2020.
[28] Keith Powell, Xudong Li, Daniel Assumpcao, Let´ıcia
Magalh˜aes, Neil Sinclair, and Marko Lonˇcar.
DC-
stable electro-optic modulators using thin-film lithium
tantalate. Optics Express, 32(25):44115–44122, 2024.
[29] F. Jermann and J. Otten. Light-induced charge transport
in linbo3:fe at high light intensities. Journal of the Optical
Society of America B, 10(11):2085–2092, 1993.
[30] M. Goulkov and Th. Woike. Photoelectric response in
photorefractive linbo3:fe versus fe2+/fe3+ ratio studied
by pils method. Journal of the Optical Society of America
B, 31(5):1071–1077, 2014.
[31] Y. Furukawa, K. Kitamura, A. Alexandrovski, R. K.
Route, M. M. Fejer, and G. Foulon.
Green-induced
infrared absorption in mgo doped linbo3. Applied Physics
Letters, 78(14):1970–1972, 2001.
[32] Thomas Volk and Manfred W¨ohlecke. Lithium Niobate:
Defects,
Photorefraction and Ferroelectric Switching.
Springer, 2009.
[33] Changjian Guo, Xingjie Li, Xiaofeng Wu, Jiajie Deng,
Wenchang Yang, Weilong Ma, Ziliang Ruan, Kaixuan
Chen, Sailing He, and Liu Liu. Robust and active visible-
light integrated photonics on thin-film lithium tantalate
for underwater optical wireless communications. arXiv
preprint arXiv:2603.14346, 2026.
[34] Boris Desiatov, Amirhassan Shams-Ansari, Mian Zhang,
Cheng
Wang,
and
Marko
Lonˇcar.
Ultra-low-loss
integrated
visible
photonics
using
thin-film
lithium
niobate. Optica, 6(3):380–384, 2019.
[35] Junyin Zhang, Chengli Wang, Connor Denney, Johann
Riemensberger, Grigory Lihachev, Jianqi Hu, Wil Kao,
Terence Bl´esin, Nikolai Kuznetsov, Zihan Li, et al.
Ultrabroadband integrated electro-optic frequency comb
in lithium tantalate. Nature, 637(8048):1096–1103, 2025.
[36] Chengli
Wang,
Dengyang
Fang,
Junyin
Zhang,
Alexander Kotz, Grigory Lihachev, Mikhail Churaev,
Zihan Li, Adrian Schwarzenberger, Xin Ou, Christian
Koos, et al. Ultrabroadband thin-film lithium tantalate
modulator for high-speed communications.
Optica,
11(12):1614–1620, 2024.
[37] Ayed
Al
Sayem,
Heqing
Huang,
Ting-Chen
Hu,
Mark Cappuzzo, Alaric Tate, Rose Kopf, and Mark
Earnshaw. Multi-stage racetrack mach zehnder coupling
interferometer on tfln with thermal and electro-optic
modulation.
In CLEO: Applications and Technology,
page AA124 8. Optica Publishing Group, 2025.
[38] Ayed Sayem, Shiekh Zia Uddin, Ting-Chen Hu, Alaric
Tate, Mark Cappuzzo, Rose Kopf, and Mark Earnshaw.
High-power handling and bias stability of thin-film
lithium tantalate microring and coupling resonators.
arXiv preprint arXiv:2602.00922, 2026.
[39] Guanbao
Zhao,
Luohan
Peng,
and
Jinbiao
Xiao.
Research on DC-drift of TFLN modulator.
In Proc.
SPIE 13806, Optoelectronic Materials and Devices, page
1380602, 2025.
[40] Oguz Tolga Celik,
Nancy Yousry Ammar,
Taewon
Park,
Hubert S. Stokowski,
Kevin K. S. Multani,
Yudan Hwang, Martin M. Fejer, and Amir H. Safavi-
Naeini.
Roles of temperature, materials, and domain
inversion in high-performance, low-bias-drift thin-film
lithium niobate blue light modulators. Optics Express,
32(21):36160–36178, 2024.
[41] Jeffrey Holzgrafe, Eric Puma, Rebecca Cheng, Hana
Warner, Amirhassan Shams-Ansari, Raji Shankar, and
Marko Lonˇcar. Relaxation of the electro-optic response
in thin-film lithium niobate modulators. Optics Express,
32(3):3619–3631, 2024.
[42] Yuan Shen, Xiaoqian Shu, Lingmei Ma, Shaoliang Yu,
Gengxin Chen, Liu Liu, Renyou Ge, Bigeng Chen, and
Yunjiang Rao. Ultra-high extinction ratio optical pulse
generation with a thin film lithium niobate modulator
for distributed acoustic sensing.
Photonics Research,
12(1):40–50, 2023.

