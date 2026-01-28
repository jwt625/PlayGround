import gdsfactory as gf

# Create main component
c = gf.Component("chip2_iter10")

# Parameters from detailed analysis and verifier feedback
ring_radius = 10.0  # µm
ring_width = 0.5    # µm
coupling_gap = 0.25 # µm (back to target spec)
ring_spacing = 13.0 # µm (reduced from 15 to match target better)
bus_width = 0.5     # µm

# Create continuous bus waveguide running horizontally
bus_length = 70  # µm
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.movey(0)  # position at y=0

# Create three identical circular rings positioned to couple with bus
rings = []
for i in range(3):
    # Create circular ring
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    
    # Position ring: bottom of ring close to top of bus
    x_pos = 12 + i * ring_spacing
    # Ring center y = bus top + coupling_gap + ring_radius
    y_pos = bus_width / 2 + coupling_gap + ring_radius
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Add multiple concentric metal electrode rings OVER each ring
# Very dense packing with 0.1 µm steps
electrode_radii = [11.5, 11.4, 11.3, 11.2, 11.1, 11.0, 10.9, 10.8, 10.7, 10.6, 10.5, 10.4, 10.3, 10.2, 10.1, 10.0, 9.9, 9.8, 9.7, 9.6, 9.5, 9.4, 9.3, 9.2, 9.1, 9.0, 8.9, 8.8, 8.7, 8.6, 8.5]  # µm
electrode_width = 0.2  # µm (very narrow for dense packing)

for i, ring in enumerate(rings):
    for elec_radius in electrode_radii:
        elec = c << gf.components.ring(
            radius=elec_radius,
            width=electrode_width,
            layer=(2, 0)  # Metal layer
        )
        elec.move(ring.center)

# Add input grating coupler at left end
gc_input = c << gf.components.grating_coupler_elliptical_te()
gc_input.connect('o1', bus.ports['o1'])

# Add bus splitter at the end using MMI 1x2
splitter = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=10, length_mmi=10)
splitter.connect('o1', bus.ports['o2'])

# Add routing from splitter to grating couplers
bend_radius = 10.0  # µm

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

# Add 2 more grating couplers in vertical array below gc_out2 (total 5)
gc_spacing = 8.0
for i in range(2):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc_out2.center[0], gc_out2.center[1] - 10 - i * gc_spacing))

