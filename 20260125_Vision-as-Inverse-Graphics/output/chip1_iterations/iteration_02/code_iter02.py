import gdsfactory as gf

# Create main component
c = gf.Component("chip1_iter02")

# Iteration 2: Adjusted based on verifier feedback
# - Reduce bend_radius from 72.5 to 68 µm
# - Reduce straight_length from 22.5 to 18 µm

# Parameters
bend_radius = 68.0  # µm (adjusted from 72.5)
straight_length = 18.0  # µm (adjusted from 22.5)
waveguide_width = 0.5  # µm (standard single-mode)

# Create left grating coupler
gc_left = c << gf.components.grating_coupler_elliptical_te()

# Create straight section from left grating
straight_left = c << gf.components.straight(length=straight_length, width=waveguide_width)
straight_left.connect("o1", gc_left.ports["o1"])

# Create semicircular bend (180 degrees)
bend = c << gf.components.bend_circular(radius=bend_radius, angle=180, width=waveguide_width)
bend.connect("o1", straight_left.ports["o2"])

# Create straight section to right grating
straight_right = c << gf.components.straight(length=straight_length, width=waveguide_width)
straight_right.connect("o1", bend.ports["o2"])

# Create right grating coupler
gc_right = c << gf.components.grating_coupler_elliptical_te()
gc_right.connect("o1", straight_right.ports["o2"])

# Add ports at the grating couplers for reference
c.add_port("input", port=gc_left.ports["o2"])
c.add_port("output", port=gc_right.ports["o2"])
