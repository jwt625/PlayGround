import gdsfactory as gf

# Create main component
c = gf.Component("chip2_iter15")

# Parameters - MAJOR CORRECTIONS based on detailed VLM analysis
ring_radius = 10.0  # µm (confirmed correct)
ring_width = 0.5    # µm (target is 0.5, not 1.5)
coupling_gap = 0.3  # µm (target is 0.2-0.4, not 9-10!)
ring_spacing = 20.0 # µm (center-to-center, NOT edge-to-edge)
bus_width = 0.5     # µm (target is 0.5, not 1.5)

# Create continuous STRAIGHT bus waveguide (VLM: should be straight, not split)
bus_length = 100  # µm - extend to right edge
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.movey(0)

# Create three rings with PROPER SPACING (not overlapping!)
rings = []
for i in range(3):
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    
    # Position: center-to-center spacing of 20 µm
    x_pos = 15 + i * ring_spacing
    # Ring bottom edge should be coupling_gap away from bus top edge
    y_pos = bus_width / 2 + coupling_gap + ring_radius
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Add THREE concentric electrode rings over EACH resonator (VLM: missing 2 inner rings!)
# VLM says target has 3 rings per resonator
for i, ring in enumerate(rings):
    # Three concentric rings per resonator - make them more visible
    electrode_radii = [
        ring_radius + 2.0,  # Outer ring
        ring_radius + 1.0,  # Middle ring
        ring_radius + 0.3   # Inner ring (just outside resonator edge)
    ]
    electrode_width = 0.5  # µm - make wider for visibility
    
    for elec_radius in electrode_radii:
        elec = c << gf.components.ring(
            radius=elec_radius,
            width=electrode_width,
            layer=(2, 0)  # Metal layer
        )
        elec.move(ring.center)
    
    # Add routing trace from outermost electrode to pad (VLM: missing routing!)
    # Route from top of outermost electrode ring upward to a pad
    trace_start_y = ring.center[1] + electrode_radii[0]
    trace_length = 15  # µm

    # Create trace on metal layer
    trace = c << gf.components.straight(length=trace_length, width=0.5)
    trace.rotate(90)
    trace.move((ring.center[0], trace_start_y))

    # Add contact pad at end of trace
    pad = c << gf.components.pad(size=(10, 10), layer=(2, 0))
    pad.move((ring.center[0] - 5, trace_start_y + trace_length))

# VLM: ALL 5 grating couplers should be at OUTPUT (right side), NOT at input!
# Create vertical array of 5 GCs on the right side
gc_x = bus_length + 10  # Position to the right of bus end
gc_start_y = -30
gc_spacing = 15  # µm (VLM: uniform spacing ~20-25 µm)

gcs = []
for i in range(5):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.rotate(90)  # Orient for vertical fiber coupling
    gc.move((gc_x, gc_start_y + i * gc_spacing))
    gcs.append(gc)

