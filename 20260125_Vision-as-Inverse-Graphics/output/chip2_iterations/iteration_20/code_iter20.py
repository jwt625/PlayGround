import gdsfactory as gf

# Iteration 20: Based on COMPLETE COMPONENT INVENTORY from VLM
# KEY INSIGHT: NO grating couplers! The right-side structures are ELECTRODE PADS!
# - 3 ring resonators
# - 3 electrode rings (one per resonator)
# - 5 electrode pads on right side
# - 5 routing traces from pads to electrode rings
# - 12 straight waveguides + 6 bends for coupling

c = gf.Component("chip2_iter20")

# Parameters
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.3
ring_spacing = 13.0
bus_width = 0.5

# Create 3 horizontal waveguides (top, middle, bottom)
wg_top = c << gf.components.straight(length=80, width=bus_width)
wg_top.movey(25)

wg_middle = c << gf.components.straight(length=80, width=bus_width)
wg_middle.movey(0)

wg_bottom = c << gf.components.straight(length=80, width=bus_width)
wg_bottom.movey(-25)

# Create 3 rings with coupling to middle waveguide
rings = []
electrode_rings = []

for i in range(3):
    # Ring resonator
    ring = c << gf.components.ring(radius=ring_radius, width=ring_width)
    x_pos = 15 + i * ring_spacing
    y_pos = wg_middle.center[1] + coupling_gap + ring_radius
    ring.move((x_pos, y_pos))
    rings.append(ring)
    
    # ONE electrode ring per resonator (not 9!)
    elec_ring = c << gf.components.ring(
        radius=ring_radius + 1.5,
        width=0.5,
        layer=(2, 0)
    )
    elec_ring.move(ring.center)
    electrode_rings.append(elec_ring)

# Add 5 electrode pads on right side (vertically aligned)
pad_x = 75
pad_start_y = -20
pad_spacing = 10
pads = []

for i in range(5):
    pad = c << gf.components.pad(size=(10, 10), layer=(2, 0))
    pad_y = pad_start_y + i * pad_spacing
    pad.move((pad_x, pad_y))
    pads.append(pad)

# Add 5 routing traces from pads to electrode rings
# VLM said: "Mostly vertical and horizontal, forming a grid-like routing"

# Trace 1: Top pad to top of right electrode ring
trace1 = c << gf.components.rectangle(size=(2.0, 15), layer=(2, 0))
trace1.move((pad_x + 5, pads[0].center[1]))

# Trace 2: Second pad to middle electrode ring
trace2_h = c << gf.components.rectangle(size=(20, 2.0), layer=(2, 0))
trace2_h.move((pad_x - 20, pads[1].center[1]))

trace2_v = c << gf.components.rectangle(size=(2.0, 10), layer=(2, 0))
trace2_v.move((pad_x - 20, pads[1].center[1]))

# Trace 3: Third pad to left electrode ring
trace3_h = c << gf.components.rectangle(size=(40, 2.0), layer=(2, 0))
trace3_h.move((pad_x - 40, pads[2].center[1]))

trace3_v = c << gf.components.rectangle(size=(2.0, 15), layer=(2, 0))
trace3_v.move((pad_x - 40, pads[2].center[1]))

# Trace 4: Fourth pad routing
trace4 = c << gf.components.rectangle(size=(2.0, 20), layer=(2, 0))
trace4.move((pad_x + 3, pads[3].center[1]))

# Trace 5: Fifth pad routing
trace5 = c << gf.components.rectangle(size=(2.0, 25), layer=(2, 0))
trace5.move((pad_x + 1, pads[4].center[1]))

# Add vertical waveguides connecting horizontal waveguides to rings
# VLM said: "6 vertical segments... at x≈150, 250, 350, 450, 550, 650"
for i, ring in enumerate(rings):
    # Vertical waveguide from top horizontal to ring
    wg_v_top = c << gf.components.straight(length=10, width=bus_width)
    wg_v_top.rotate(90)
    wg_v_top.move((ring.center[0], wg_top.center[1]))
    
    # Vertical waveguide from ring to bottom horizontal
    wg_v_bottom = c << gf.components.straight(length=10, width=bus_width)
    wg_v_bottom.rotate(90)
    wg_v_bottom.move((ring.center[0], wg_bottom.center[1]))

# Add 6 waveguide bends for coupling (2 per ring: input and output)
bend_radius = 5.0
for i, ring in enumerate(rings):
    # Top bend
    bend_top = c << gf.components.bend_circular(radius=bend_radius, angle=90)
    bend_top.move((ring.center[0] - bend_radius, ring.center[1] + ring_radius + coupling_gap))
    
    # Bottom bend
    bend_bottom = c << gf.components.bend_circular(radius=bend_radius, angle=90)
    bend_bottom.move((ring.center[0] - bend_radius, ring.center[1] - ring_radius - coupling_gap))

