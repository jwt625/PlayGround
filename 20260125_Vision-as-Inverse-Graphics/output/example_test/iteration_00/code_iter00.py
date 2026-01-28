
import gdsfactory as gf

# Create a simple MZI
c = gf.Component("test_mzi")
mzi = c << gf.components.mzi(delta_length=10)
c.add_ports(mzi.ports)
