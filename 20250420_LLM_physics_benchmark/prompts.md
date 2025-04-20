

# Galileo's ramp


write a python script to plot and generate a gif animation that shows Galileo's ramp thought experiment, i.e., there are four panels, each panel is one subplot showing one setting of the ramp experiment. The four panels has the same downward slope at the beginning, the balls start from the top left of the downward ramp and rolls down because of gravity. The upward ramp would have different slope in the four different panels. One panel is flat and does not have an upward ramp. Be physically as precise as possible. Ignore non-ideal factors such as friction and air drag etc. Save the final gif as output.gif.



#### Task  
Write a Python script that generates `output.gif`, an animation with **four sub‑plots** depicting Galileo’s inclined‑plane experiment.  
* Each panel starts with an identical **down‑slope** segment of length \(L_{\mathrm{down}}\) and angle \(\theta_{\mathrm{down}}\).  
* A rigid sphere (point mass) is released from rest at the top left; gravitational acceleration \(g=9.80665\;\text{m s}^{-2}\).  
* The **up‑slope** that follows has a different angle \(\theta_{i}\) in each panel:  
  \[
    \theta_{1}=+\tfrac{\pi}{6},\quad
    \theta_{2}=+\tfrac{\pi}{12},\quad
    \theta_{3}=0,\quad
    \theta_{4}=-\tfrac{\pi}{12}.
  \]  
  Neglect friction and air drag.  
* Compute the motion analytically with  
  \[
    a = g\sin\theta,\qquad
    v(t) = v_{0}+at,\qquad
    x(t) = x_{0}+v_{0}t+\tfrac12at^{2}.
  \]  
* Plot true spatial trajectories, synchronise time across panels, label axes in metres, and display frame rate clearly.

#### Description  
This task probes classical‑mechanics reasoning (constant‑acceleration kinematics, matching boundary conditions at the slope junctions) and basic scientific visualisation skills (subplot layouts, animation, file output).

#### Evaluation Points  
- **Physics fidelity**: correct accelerations for each slope; continuous position/velocity where slopes join.  
- **Code quality**: modular functions, clear parameters, precise units; no magic numbers.  
- **Visual clarity**: labelled axes, equal aspect ratio, consistent time scale, legend for slopes.  
- **Output**: `output.gif` present, smooth animation (≥ 30 fps), four correctly arranged panels.


# Maxwell’s Displacement Current in a Charging Capacitor

#### Task  
Derive the analytic fields and generate `output.gif`, a two‑panel animation over one period \(T=2\pi/\omega\):

1. **Electric field (uniform)**  
   \[
     E(r,t)=\frac{I_{0}}{\pi R^{2}\varepsilon_{0}\,\omega}\,\sin(\omega t).
   \]

2. **Azimuthal magnetic field**  
   \[
     B(r,t)=
     \begin{cases}
       \dfrac{\mu_{0} I_{0} r}{2\pi R^{2}}\cos(\omega t), & 0\le r\le R,\\[6pt]
       0, & r>R.
     \end{cases}
   \]

* **Parameters** \(R=5\;\text{cm},\;d=2\;\text{mm},\;I_{0}=1\;\text{A},\;\omega=2\pi\times10^{5}\;\text{rad s}^{-1}\).  
* Plot \(E(r,t)\) and \(B(r,t)\) versus \(r\) (mid‑plane, 0–5 cm) for 200 equally spaced time‑steps; identical time axis in both sub‑plots; SI labels; ignore edge fringing.  

#### Description  
Assesses understanding of the displacement‑current term in Maxwell–Ampère’s law, uniform‑field approximation, and quantitative field visualisation.

#### Evaluation Points  
- **Complete derivation** showing where \(\partial E/\partial t\) enters.  
- **Numeric accuracy** using the supplied constants.  
- **Synchronous animation** – two panels advance together, axes and units clear.  
- **Brief physical note** in titles/captions explaining \(B\propto r\) behaviour.








# Quantum Double‑Slit Wave‑Packet


#### Task  
Simulate an electron Gaussian wave‑packet incident on a double‑slit barrier with the time‑dependent Schrödinger equation and deliver:

* `double_slit.gif` – colour‑map animation of \(|\psi(x,t)|^{2}\) from launch (\(x=-6\,\mu\text{m}\)) to arrival at the detection plane (\(x=+10\,\mu\text{m}\)).  
* A static plot overlaying the **time‑averaged** intensity at the detection plane with the Fraunhofer prediction  
  \[
    I_{\text{theory}}(y)
      \propto
      \cos^{2}\!\Bigl(\tfrac{\pi d y}{\lambda L}\Bigr)\;
      \operatorname{sinc}^{2}\!\Bigl(\tfrac{\pi a y}{\lambda L}\Bigr).
  \]

* **Physical parameters**  
  * Electron mass \(m_{e}\).  
  * Initial packet: width \(\sigma=0.5\,\mu\text{m}\), centre momentum \( \hbar k_{0}, \;k_{0}=5\times10^{6}\,\text{m}^{-1}\).  
  * Barrier at \(x=0\): height \(10\,\text{eV}\); two rectangular slits width \(a=0.3\,\mu\text{m}\) centred at \(y=\pm1.0\,\mu\text{m}\) (centre‑to‑centre \(d=2.0\,\mu\text{m}\)).  
  * Large transverse extent so reflections are negligible; absorbing layers at grid edges.  

* **Numerics** – split‑operator FFT or Crank–Nicolson in 2‑D (preferred) or reduced 1‑D Fresnel approximation; choose grid so \(\Delta x,\Delta t\) satisfy the stability criteria.  

#### Description  
Exercises quantum dynamics, numerical PDEs, absorbing boundaries, and comparison between simulation and analytic interference.

#### Evaluation Points  
- **Probability conservation** loss < 1 %.  
- **Fringe spacing agreement** with analytic curve within 5 %.  
- **Clear figures** – colour bar, time stamps, slit/barrier overlay.  
- **Modular, documented Python code** with adjustable grid and physical parameters.






# Gravitational Redshift from the Sun

#### Task  
1. **Derivation**  
   From the Schwarzschild line element  
   \[
     \mathrm{d}\tau^{2}= 
       \Bigl(1-\frac{2GM}{rc^{2}}\Bigr)c^{2}\mathrm{d}t^{2}
       -\Bigl(1-\frac{2GM}{rc^{2}}\Bigr)^{-1}\mathrm{d}r^{2}-r^{2}\mathrm{d}\Omega^{2},
   \]  
   show that for a photon emitted at radius \(r\) and detected at infinity  
   \[
     \frac{\Delta\lambda}{\lambda}\;\approx\;\frac{GM}{rc^{2}}
     \quad (\text{first order in }GM/rc^{2}).
   \]

2. **Numerical evaluation**  
   * Use \(M_\odot = 1.9885\times10^{30}\,\text{kg}\), \(R_\odot = 6.9634\times10^{8}\,\text{m}\).  
   * Reference line: H‑α, \(\lambda_{0}=656.281\,\text{nm}\).  
   * Output shifted wavelength and parts‑per‑million (ppm) shift.  

3. **Comment** – one or two sentences on additional perturbations (solar rotation, pressure shifts).

#### Description  
Tests competence with GR weak‑field approximation, algebraic manipulation of metrics, and accurate astrophysical calculation.

#### Evaluation Points  
- **Correct algebra** distinguishing proper and coordinate time.  
- **Numeric shift** ≈ 0.636 Å (0.064 nm) ≈ 96 ppm, within 2 %.  
- **Concise discussion** of non‑GR contributions (< 40 words).  
- **Well‑presented constants and units**; clear final boxed answer.



