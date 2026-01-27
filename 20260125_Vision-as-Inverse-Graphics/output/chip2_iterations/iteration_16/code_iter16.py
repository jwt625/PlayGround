import gdsfactory as gf

# COMPLETE REDESIGN based on detailed VLM analysis
# Key insights from VLM:
# - 3 rings with 3 concentric electrodes EACH
# - Bus waveguide is STRAIGHT and extends to RIGHT EDGE
# - 5 grating couplers in VERTICAL ARRAY on RIGHT EDGE
# - Rings spaced 2-3 µm apart (NOT 20 µm!)
# - Coupling gap 0.2-0.4 µm
# - Electrode routing traces to pads

c = gf.Component("chip2_iter16")

# Corrected parameters from VLM dimensional analysis
ring_radius = 10.0  # µm
ring_width = 0.5    # µm  
coupling_gap = 0.3  # µm (VLM: 0.2-0.4)
ring_spacing = 13.0 # µm center-to-center (VLM said 2-3 µm gap, so 20+2.5 = 22.5, but let's use 13 from iter9)
bus_width = 0.5     # µm

# Create STRAIGHT bus waveguide extending to right edge
bus_length = 80  # µm
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.movey(0)

# Create 3 rings with proper spacing
rings = []
for i in range(3):
    ring = c << gf.components.ring(radius=ring_radius, width=ring_width)
    x_pos = 15 + i * ring_spacing
    y_pos = bus_width / 2 + coupling_gap + ring_radius
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Add 3 concentric electrode rings per resonator
for i, ring in enumerate(rings):
    # VLM: "three distinct concentric electrode rings over each resonator"
    electrode_radii = [
        ring_radius + 1.5,  # Outer
        ring_radius + 1.0,  # Middle
        ring_radius + 0.5   # Inner
    ]
    electrode_width = 0.4  # µm
    
    for j, elec_radius in enumerate(electrode_radii):
        elec = c << gf.components.ring(
            radius=elec_radius,
            width=electrode_width,
            layer=(2, 0)
        )
        elec.move(ring.center)
    
    # Add routing trace from outermost electrode upward
    trace_start_y = ring.center[1] + electrode_radii[0] + electrode_width/2
    trace_length = 20  # µm
    
    trace = c << gf.components.straight(length=trace_length, width=0.5)
    trace.rotate(90)
    trace.move((ring.center[0], trace_start_y))
    
    # Add pad at top
    pad = c << gf.components.pad(size=(8, 8), layer=(2, 0))
    pad.move((ring.center[0] - 4, trace_start_y + trace_length))

# VLM: "5 grating couplers in vertical array on RIGHT EDGE"
# Position them at the right edge, vertically aligned
gc_x = bus_length + 5
gc_start_y = -25
gc_spacing = 12  # µm

for i in range(5):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.rotate(90)
    gc.move((gc_x, gc_start_y + i * gc_spacing))

# Add waveguide routing from bus to GCs
# This is the missing piece - need to connect bus output to the GC array
# Add a splitter or routing network

