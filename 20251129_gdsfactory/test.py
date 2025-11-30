

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

# Create three short waveguides for alignment at y=0, separated by 500 um along x
wg_positions = [0, 300, 600]  # x positions in um
waveguides = []

for i, x_pos in enumerate(wg_positions):
    wg = c_chip << gf.components.straight(length=1, cross_section='strip')
    wg.move((x_pos, 0))  # Position at (x_pos, 0)
    waveguides.append(wg)

# Create three spiral-MZI circuits with n_loops = 6, 7, 8
n_loops_list = [9, 10, 11]
circuits = []

for i, n_loops in enumerate(n_loops_list):
    circuit = c_chip << spiral_mzi_circuit(n_loops=n_loops, mzi_length_x=150.0)
    circuits.append(circuit)

    # Connect the MZI output (circuit's o2) to the waveguide's o1
    circuit.connect("o2", waveguides[i].ports["o1"])

# Connect waveguide output to next spiral input (chain the circuits)
for i in range(len(circuits) - 1):
    # Route from waveguide i output (o2) to next circuit's spiral input (o1)
    gf.routing.route_single(
        c_chip,
        port1=waveguides[i].ports["o2"],
        port2=circuits[i+1].ports["o1"],
        cross_section='strip',
        bend='bend_euler'
    )

# Export the chain's input and output ports, plus MZI outputs
c_chip.add_port("input", port=circuits[0].ports["o1"])  # First spiral input (chain input)
c_chip.add_port("output", port=waveguides[-1].ports["o2"])  # Last waveguide output (chain output)

# Export all MZI o3 outputs (the unused outputs from each MZI)
for i, circuit in enumerate(circuits):
    c_chip.add_port(f"mzi_{i+1}_output", port=circuit.ports["o3"])

# Print port information
print("Chip ports:")
c_chip.pprint_ports()

# Option 1: Add pins with port names and text labels (RECOMMENDED)
# c_chip_with_pins = add_pins_container(c_chip)
# c_chip_with_pins.write_gds("test.gds")

# Option 2: Just draw port markers on their layers (without text labels)
c_chip.draw_ports()
c_chip.write_gds("test.gds")

# Show in KLayout
# c_chip_with_pins.show()

