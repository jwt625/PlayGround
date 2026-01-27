import gdsfactory as gf

# Create main component
c = gf.Component("chip2_iter02")

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
# Each ring is positioned with coupling_gap from the bus
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

# Add input/output ports from bus waveguide
c.add_port("input", port=bus.ports["o1"])
c.add_port("output", port=bus.ports["o2"])
