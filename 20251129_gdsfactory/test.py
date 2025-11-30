

import gdsfactory as gf
import inspect
from gdsfactory.component import Component
from gdsfactory.add_pins import add_pins_container


@gf.cell
def spiral_mzi_circuit(n_loops: int = 6, mzi_length_x: float = 150.0) -> Component:
    """
    Create a circuit with spiral, waveguide, and MZI connected together.

    Args:
        n_loops: Number of loops in the spiral
        mzi_length_x: Length of the MZI arms in x direction

    Returns:
        Component with three ports: o1 (spiral input), o2 (MZI output 1), o3 (MZI output 2)
    """
    c = gf.Component()

    # Create sub-components
    c_strip = gf.components.straight(length=1, cross_section='strip')
    c_spiral = gf.components.spiral(
        length=0,
        bend='bend_euler',
        straight='straight',
        cross_section='strip',
        spacing=3,
        n_loops=n_loops
    )
    c_mzi = gf.components.mzi1x2_2x2(
        cross_section='strip',
        length_y=0.0,
        length_x=mzi_length_x,
        delta_length=0.0,
    )

    # Add references
    ref_WG = c << c_strip
    ref_spiral = c << c_spiral
    ref_spiral.rotate(180).movey(-60)
    ref_mzi = c << c_mzi

    # Connect components
    ref_mzi.connect("o2", ref_WG.ports["o2"])

    # Route from spiral o2 to waveguide o1 with Euler bends
    gf.routing.route_single(
        c,
        port1=ref_spiral.ports["o2"],
        port2=ref_WG.ports["o1"],
        cross_section='strip',
        bend='bend_euler'
    )

    # Export unused ports
    c.add_port("o1", port=ref_spiral.ports["o1"])  # Spiral input
    c.add_port("o2", port=ref_mzi.ports["o1"])     # MZI output 1
    c.add_port("o3", port=ref_mzi.ports["o3"])     # MZI output 2

    return c


# Create the main chip component
c_chip = gf.Component("chip")

# Instantiate the spiral-MZI circuit with custom parameters
circuit = c_chip << spiral_mzi_circuit(n_loops=6, mzi_length_x=150.0)

# Add the circuit's ports to the chip
c_chip.add_ports(circuit.ports)

# Print port information
print("Circuit ports:")
c_chip.pprint_ports()

# Option 1: Add pins with port names and text labels (RECOMMENDED)
# c_chip_with_pins = add_pins_container(c_chip)
# c_chip_with_pins.write_gds("test.gds")

# Option 2: Just draw port markers on their layers (without text labels)
c_chip.draw_ports()
c_chip.write_gds("test.gds")

# Show in KLayout
# c_chip_with_pins.show()

