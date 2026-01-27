import gdsfactory as gf

# Create main component
c = gf.Component("chip2_iter03")

# Parameters from detailed analysis
ring_radius = 10.0  # µm
ring_width = 0.5    # µm
coupling_gap = 0.25 # µm (critical!)
ring_spacing = 30.0 # µm (center-to-center)
bus_width = 0.5     # µm

# Create continuous bus waveguide running horizontally
bus_length = ring_spacing * 2 + 40  # enough to span all rings plus margins
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.movey(0)  # position at y=0

# Create three identical circular rings positioned above the bus
rings = []
for i in range(3):
    # Create circular ring
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    
    # Position ring: x = 20 + i*30, y = ring_radius + coupling_gap + bus_width/2
    x_pos = 20 + i * ring_spacing
    y_pos = ring_radius + coupling_gap + bus_width / 2
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Add metal electrodes adjacent to each ring (2 per ring = 6 total)
electrode_width = 2.0  # µm
electrode_length = 15.0  # µm
electrode_gap = 1.5  # µm from ring edge

for i, ring in enumerate(rings):
    # Left electrode
    elec_left = c << gf.components.rectangle(
        size=(electrode_width, electrode_length),
        layer=(2, 0)  # Metal layer
    )
    elec_left.move((ring.center[0] - ring_radius - electrode_gap - electrode_width, 
                    ring.center[1] - electrode_length/2))
    
    # Right electrode
    elec_right = c << gf.components.rectangle(
        size=(electrode_width, electrode_length),
        layer=(2, 0)  # Metal layer
    )
    elec_right.move((ring.center[0] + ring_radius + electrode_gap, 
                     ring.center[1] - electrode_length/2))

# Add grating couplers at outputs
gc_spacing = 10.0  # spacing between grating couplers
num_gcs = 5
gc_start_x = bus.xmax + 15
gc_start_y = -20

for i in range(num_gcs):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc_start_x, gc_start_y + i * gc_spacing))

# Add input/output ports
c.add_port("input", port=bus.ports["o1"])
c.add_port("output", port=bus.ports["o2"])

