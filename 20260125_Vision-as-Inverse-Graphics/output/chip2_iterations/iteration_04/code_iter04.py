import gdsfactory as gf

# Create main component
c = gf.Component("chip2_iter04")

# Parameters from detailed analysis
ring_radius = 10.0  # µm
ring_width = 0.5    # µm
coupling_gap = 0.25 # µm (critical!)
ring_spacing = 12.0 # µm (center-to-center, reduced to match target better)
bus_width = 0.5     # µm

# Create continuous bus waveguide running horizontally
bus_length = 80  # adjusted length
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.movey(0)  # position at y=0

# Create three identical circular rings positioned above the bus
# Position them so the bottom of each ring is coupling_gap away from top of bus
rings = []
for i in range(3):
    # Create circular ring
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    
    # Position ring: bottom edge at coupling_gap from bus top edge
    x_pos = 15 + i * ring_spacing
    y_pos = coupling_gap + bus_width / 2 + ring_radius
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Add metal electrodes adjacent to rings
# Based on target: electrodes are on left/right sides of rings
electrode_width = 1.5  # µm (adjusted)
electrode_length = 12.0  # µm (adjusted)
electrode_gap = 1.0  # µm from ring edge (adjusted)

for i, ring in enumerate(rings):
    # Top electrode (above ring)
    elec_top = c << gf.components.rectangle(
        size=(electrode_length, electrode_width),
        layer=(2, 0)  # Metal layer
    )
    elec_top.move((ring.center[0] - electrode_length/2, 
                   ring.center[1] + ring_radius + electrode_gap))

# Add output routing from bus end
# Create splitter region
bus_end_x = bus.xmax
splitter_start = bus_end_x + 2

# Add bends and routing to grating couplers
# Simplified: add grating couplers in a fan-out pattern
gc_spacing = 8.0  # spacing between grating couplers
num_gcs = 5
gc_start_x = splitter_start + 15
gc_center_y = 0

for i in range(num_gcs):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc_y = gc_center_y + (i - 2) * gc_spacing  # center around y=0
    gc.move((gc_start_x, gc_y))
    gc.rotate(90)  # rotate for proper orientation

# Add input/output ports
c.add_port("input", port=bus.ports["o1"])
c.add_port("output", port=bus.ports["o2"])

