import gdsfactory as gf

# Create main component
c = gf.Component("chip2_iter06")

# Parameters from detailed analysis and topology clarification
ring_radius = 10.0  # µm
ring_width = 0.5    # µm
coupling_gap = 0.25 # µm (critical!)
ring_spacing = 15.0 # µm (center-to-center)
bus_width = 0.5     # µm

# Create continuous bus waveguide running horizontally
bus_length = 70  # adjusted length
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.movey(0)  # position at y=0

# Create three identical circular rings positioned to couple with bus
# CRITICAL FIX: Position rings so the gap between ring inner edge and bus top edge is exactly coupling_gap
rings = []
for i in range(3):
    # Create circular ring
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    
    # Position ring: inner edge of ring at coupling_gap from bus top edge
    # Ring center y = bus_top + coupling_gap + (ring_radius - ring_width/2)
    x_pos = 12 + i * ring_spacing
    y_pos = bus_width / 2 + coupling_gap + (ring_radius - ring_width / 2)
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Add metal electrodes OVER each ring (concentric structures)
# Tighter gap for better modulation efficiency
electrode_radius_outer = ring_radius + 1.5  # µm (reduced gap)
electrode_radius_inner = ring_radius - 1.5  # µm (reduced gap)
electrode_width = 1.0  # µm

for i, ring in enumerate(rings):
    # Outer electrode ring
    elec_outer = c << gf.components.ring(
        radius=electrode_radius_outer,
        width=electrode_width,
        layer=(2, 0)  # Metal layer
    )
    elec_outer.move(ring.center)
    
    # Inner electrode ring
    elec_inner = c << gf.components.ring(
        radius=electrode_radius_inner,
        width=electrode_width,
        layer=(2, 0)  # Metal layer
    )
    elec_inner.move(ring.center)

# Add input grating coupler at left end
gc_input = c << gf.components.grating_coupler_elliptical_te()
gc_input.connect('o1', bus.ports['o1'])

# Add bus splitter at the end using Y-splitter
splitter = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=10, length_mmi=10)
splitter.connect('o1', bus.ports['o2'])

# Add routing from splitter to grating couplers
bend_radius = 10.0  # increased to match target

# Upper output
straight_up1 = c << gf.components.straight(length=5, width=bus_width)
straight_up1.connect('o1', splitter.ports['o2'])

bend_up = c << gf.components.bend_circular(radius=bend_radius, angle=90)
bend_up.connect('o1', straight_up1.ports['o2'])

straight_up2 = c << gf.components.straight(length=10, width=bus_width)
straight_up2.connect('o1', bend_up.ports['o2'])

gc_out1 = c << gf.components.grating_coupler_elliptical_te()
gc_out1.connect('o1', straight_up2.ports['o2'])

# Lower output
straight_down1 = c << gf.components.straight(length=5, width=bus_width)
straight_down1.connect('o1', splitter.ports['o3'])

bend_down = c << gf.components.bend_circular(radius=bend_radius, angle=-90)
bend_down.connect('o1', straight_down1.ports['o2'])

straight_down2 = c << gf.components.straight(length=10, width=bus_width)
straight_down2.connect('o1', bend_down.ports['o2'])

gc_out2 = c << gf.components.grating_coupler_elliptical_te()
gc_out2.connect('o1', straight_down2.ports['o2'])

# Add 2 more grating couplers in vertical array (total 5: 1 input + 2 outputs + 2 extra)
gc_spacing = 8.0
for i in range(2):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc_out1.center[0] + 10, gc_out1.center[1] - 10 - i * gc_spacing))

