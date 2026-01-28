import gdsfactory as gf

# Create main component
c = gf.Component("chip2_iter01")

# Parameters from detailed analysis
ring_radius = 10.0  # µm
coupling_gap = 0.25 # µm (critical!)
ring_spacing = 30.0 # µm (center-to-center)

# Create three ring resonators with integrated bus coupling
rings = []
for i in range(3):
    ring = c << gf.components.ring_single(
        gap=coupling_gap,
        radius=ring_radius,
        length_x=0.1,
        length_y=0.1
    )
    ring.movex(i * ring_spacing)
    rings.append(ring)

# Add ports from first and last ring
c.add_port("input", port=rings[0].ports["o1"])
c.add_port("output", port=rings[-1].ports["o2"])
