

import gdsfactory as gf
from gdsfactory.component import Component


c_chip = gf.Component("chip")
# xsec = gf.cross_section.strip
# p_spiral = gf.path.spiral_archimedean(min_bend_radius=10, separation=3, number_of_loops=10, npoints=5000)
p_spiral = gf.components.spiral(length=0, bend='bend_euler', straight='straight', cross_section='strip', spacing=3, n_loops=6).copy()
f = p_spiral.plot()
c_chip << p_spiral

c_chip.write_gds("test.gds")

