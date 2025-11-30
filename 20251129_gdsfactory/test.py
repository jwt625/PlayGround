

import gdsfactory as gf
import inspect
from gdsfactory.component import Component
from gdsfactory.add_pins import add_pins_container


c_chip = gf.Component("chip")
# xsec = gf.cross_section.strip
# p_spiral = gf.path.spiral_archimedean(min_bend_radius=10, separation=3, number_of_loops=10, npoints=5000)
c_strip = gf.components.straight(length=1, cross_section='strip').copy()
c_spiral = gf.components.spiral(length=0, bend='bend_euler', straight='straight', 
                                cross_section='strip', spacing=3, n_loops=6).copy()


c_mzi = gf.components.mzi1x2_2x2(
    cross_section='strip',
    length_y=0.0,  # Increased arm length (both arms)
    length_x=150.0,  # Increased arm length (both arms)
    delta_length=0.0,  # Keep arms same length (no difference)
).copy()

# print(inspect.signature(gf.components.mzi1x2_2x2))
# check all port locations of the mzi
for ind_port in range(len(c_mzi.ports)):
    print(c_mzi.ports[ind_port])

ref_WG = c_chip << c_strip
ref_spiral = c_chip << c_spiral
ref_spiral.rotate(180).movey(-60)
ref_mzi = c_chip << c_mzi

# Connect o2 of ref_mzi to o1 of ref_WG
ref_mzi.connect("o2", ref_WG.ports["o2"])

# Route from spiral o1 to waveguide o1 with Euler bends
route = gf.routing.route_single(
    c_chip,
    port1=ref_spiral.ports["o2"],
    port2=ref_WG.ports["o1"],
    cross_section='strip',
    bend='bend_euler'
)

# print all components
# print(c_spiral)
# print(c_mzi)
# print(c_chip)

# Option 1: Add pins with port names and text labels (RECOMMENDED)
# c_chip_with_pins = add_pins_container(c_chip)
# c_chip_with_pins.write_gds("test.gds")

# Option 2: Just draw port markers on their layers (without text labels)
c_chip.draw_ports()
c_chip.write_gds("test.gds")

# Show in KLayout
# c_chip_with_pins.show()

