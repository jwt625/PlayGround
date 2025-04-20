

#%%
"""
galileo_ramp_animation.py
Generate an animated GIF of Galileo’s two‑ramp thought experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ------------------------- Physical parameters -------------------------
g = 9.81               # m s⁻², gravitational acceleration
theta1 = np.deg2rad(30)  # fixed downward slope angle
L1     = 5.0            # m, length of the downward ramp
theta2_deg = [60, 30, 15, 0]           # upward‑ramp slopes for the four panels
theta2     = np.deg2rad(theta2_deg)

# Derived quantities (valid for all panels)
H         = L1 * np.sin(theta1)            # height drop
v_bottom  = np.sqrt(2 * g * H)             # speed at ramp junction
a1        =  g * np.sin(theta1)            # accel. down the slope
t1        =  np.sqrt(2 * L1 / a1)          # time to reach the junction
x1        =  L1 * np.cos(theta1)           # junction coordinates
y1        =  0.0

# One‑time pre‑compute of each panel’s upward ramp length and stop time
L2, t2 = [], []
for th in theta2:
    if th > 1e-6:                              # non‑flat ramp
        L2_i = H / np.sin(th)                  # distance needed to regain height
        t2_i = v_bottom / (g * np.sin(th))     # time until momentary rest
    else:                                      # flat extension (θ = 0)
        L2_i = 8.0                             # arbitrary visible length
        t2_i = L2_i / v_bottom                 # constant‑velocity travel time
    L2.append(L2_i)
    t2.append(t2_i)

T_max   = t1 + max(t2)      # total simulated duration
dt      = 0.04              # s, frame time step (≈25 fps)
nframes = int(np.ceil(T_max / dt)) + 1

# ------------------------- Figure / axes setup -------------------------
fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
axarr = axarr.ravel()
titles = [f"Upward ramp = {d}°" if d else "Flat extension"
          for d in theta2_deg]

# Pre‑draw the ramps and create a point artist for each ball
balls, xmax = [], 0.0
for ax, th2, L2_i, title in zip(axarr, theta2, L2, titles):
    # Segment end‑points
    x2 = x1 + L2_i * np.cos(th2) if th2 > 1e-6 else x1 + L2_i
    y2 = y1 + L2_i * np.sin(th2) if th2 > 1e-6 else y1

    ax.plot([0,  x1], [H, y1], 'k-', lw=2)   # down‑slope
    ax.plot([x1, x2], [y1, y2], 'k-', lw=2)  # up/flat slope
    ball, = ax.plot([], [], 'ro', ms=8)
    balls.append(ball)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.5, x2 + 0.5)
    ax.set_ylim(-0.5, H + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    xmax = max(xmax, x2)

# Make x‑limits uniform
for ax in axarr:
    ax.set_xlim(-0.5, xmax + 0.5)

# ------------------------- Kinematics helper -------------------------
def position(t, th2, L2_i):
    """
    Return (x, y) of the ball at time *t* for an upward‑slope of angle th2.
    """
    if t < t1:       # descending leg
        s = 0.5 * a1 * t**2
        return s * np.cos(theta1), H - s * np.sin(theta1)

    # upward or flat leg
    t_rel = t - t1
    if th2 > 1e-6:   # ascending ramp with deceleration
        a2 = -g * np.sin(th2)
        # still climbing?
        if t_rel <= -v_bottom / a2:
            s = v_bottom * t_rel + 0.5 * a2 * t_rel**2
        else:        # reached peak
            s = L2_i
        x = x1 + s * np.cos(th2)
        y = y1 + s * np.sin(th2)
    else:            # flat ramp, constant velocity
        s = min(v_bottom * t_rel, L2_i)
        x = x1 + s
        y = y1
    return x, y

# ------------------------- Animation functions -------------------------
def init():
    for b in balls:
        b.set_data([], [])
    return balls

def animate(frame):
    t = frame * dt
    for i, b in enumerate(balls):
        x, y = position(t, theta2[i], L2[i])
        b.set_data([x], [y])  
        # b.set_data(*position(t, theta2[i], L2[i]))
    return balls

ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=nframes, interval=dt*1000, blit=True
)

# ------------------------- Save to GIF -------------------------
ani.save("output.gif", writer='pillow', fps=int(round(1 / dt)))
print("Saved: output.gif")

# %%
