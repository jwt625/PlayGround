import gdsfactory as gf

# CORRECT TOPOLOGY based on detailed VLM analysis:
# - 1 input GC on left
# - Main bus runs horizontally ABOVE the 3 rings
# - Each ring has individual output waveguide to separate GC (3 drop GCs)
# - Bus continues to right and splits to 2 more GCs (through/monitor)
# - Total: 1 input + 3 ring drops + 2 bus outputs = 6 GCs

c = gf.Component("chip2_iter17")

# Parameters
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.3
ring_spacing = 13.0
bus_width = 0.5

# Input GC on left
gc_input = c << gf.components.grating_coupler_elliptical_te()
gc_input.rotate(180)
gc_input.move((-10, 15))

# Main bus waveguide running horizontally (above rings)
bus_start = c << gf.components.straight(length=5, width=bus_width)
bus_start.connect('o1', gc_input.ports['o1'])

bus_main = c << gf.components.straight(length=70, width=bus_width)
bus_main.connect('o1', bus_start.ports['o2'])

# Create 3 rings BELOW the bus
rings = []
for i in range(3):
    ring = c << gf.components.ring(radius=ring_radius, width=ring_width)
    x_pos = 10 + i * ring_spacing
    # Position rings BELOW bus with coupling gap
    y_pos = bus_main.center[1] - coupling_gap - ring_radius
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Add 3 concentric electrodes per ring
for ring in rings:
    electrode_radii = [ring_radius + 1.5, ring_radius + 1.0, ring_radius + 0.5]
    for elec_radius in electrode_radii:
        elec = c << gf.components.ring(radius=elec_radius, width=0.4, layer=(2, 0))
        elec.move(ring.center)
    
    # Electrode routing to pad
    trace = c << gf.components.straight(length=15, width=0.5)
    trace.rotate(90)
    trace.move((ring.center[0], ring.center[1] + ring_radius + 2))
    
    pad = c << gf.components.pad(size=(8, 8), layer=(2, 0))
    pad.move((ring.center[0] - 4, ring.center[1] + ring_radius + 17))

# Add individual output waveguides from each ring to separate GCs (drop ports)
gc_drop_x = 80
gc_drop_start_y = 10
gc_drop_spacing = 12

for i, ring in enumerate(rings):
    # Waveguide from ring drop port to GC
    # Route from bottom of ring downward, then right to GC
    drop_y = ring.center[1] - ring_radius - 1
    
    wg_down = c << gf.components.straight(length=abs(drop_y - (gc_drop_start_y + i * gc_drop_spacing)), width=bus_width)
    wg_down.rotate(90)
    wg_down.move((ring.center[0], drop_y))
    
    wg_right = c << gf.components.straight(length=gc_drop_x - ring.center[0], width=bus_width)
    wg_right.move((ring.center[0], gc_drop_start_y + i * gc_drop_spacing))
    
    gc_drop = c << gf.components.grating_coupler_elliptical_te()
    gc_drop.rotate(90)
    gc_drop.move((gc_drop_x, gc_drop_start_y + i * gc_drop_spacing))

# Bus continues to right and splits to 2 monitoring GCs
bus_end_x = bus_main.ports['o2'].center[0]
bus_end_y = bus_main.ports['o2'].center[1]

# Split bus into 2 branches
splitter = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=5, length_mmi=10)
splitter.connect('o1', bus_main.ports['o2'])

# Upper branch to GC
wg_up = c << gf.components.straight(length=5, width=bus_width)
wg_up.connect('o1', splitter.ports['o2'])

bend_up = c << gf.components.bend_circular(radius=10, angle=90)
bend_up.connect('o1', wg_up.ports['o2'])

wg_up2 = c << gf.components.straight(length=10, width=bus_width)
wg_up2.connect('o1', bend_up.ports['o2'])

gc_mon1 = c << gf.components.grating_coupler_elliptical_te()
gc_mon1.connect('o1', wg_up2.ports['o2'])

# Lower branch to GC
wg_down = c << gf.components.straight(length=5, width=bus_width)
wg_down.connect('o1', splitter.ports['o3'])

bend_down = c << gf.components.bend_circular(radius=10, angle=-90)
bend_down.connect('o1', wg_down.ports['o2'])

wg_down2 = c << gf.components.straight(length=10, width=bus_width)
wg_down2.connect('o1', bend_down.ports['o2'])

gc_mon2 = c << gf.components.grating_coupler_elliptical_te()
gc_mon2.connect('o1', wg_down2.ports['o2'])

