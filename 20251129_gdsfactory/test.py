

import gdsfactory as gf
import inspect
from functools import partial
from gdsfactory.component import Component
from gdsfactory.add_pins import add_pins_container


# Create custom MZI 1x2_2x2 with phase shifter (heater in top arm)
mzi1x2_2x2_phase_shifter = partial(
    gf.components.mzi,
    combiner='mmi2x2',
    port_e1_combiner='o3',
    port_e0_combiner='o4',
    straight_x_top='straight_heater_metal',
    length_x=200  # default length
)

# Create custom MZI 1x2_1x2 with phase shifter (heater in top arm)
# This has 1x2 splitter and 1x2 combiner (only one output)
mzi1x2_1x2_phase_shifter = partial(
    gf.components.mzi,
    splitter='mmi1x2',
    combiner='mmi1x2',
    straight_x_top='straight_heater_metal',
    length_x=200  # default length
)


@gf.cell
def spiral_mzi_circuit(n_loops: int = 6, mzi_length_x: float = 150.0) -> Component:
    """
    Create a circuit with spiral, waveguide, and MZI with integrated heater connected together.

    Args:
        n_loops: Number of loops in the spiral
        mzi_length_x: Length of the MZI arms in x direction

    Returns:
        Component with optical ports: o1 (spiral input), o2 (MZI output 1), o3 (MZI output 2)
        and electrical ports from the MZI heater
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
    c_mzi = mzi1x2_2x2_phase_shifter(
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

    # Export optical ports
    c.add_port("o1", port=ref_spiral.ports["o1"])  # Spiral input
    c.add_port("o2", port=ref_mzi.ports["o1"])     # MZI output 1
    c.add_port("o3", port=ref_mzi.ports["o3"])     # MZI output 2

    # Export electrical ports from MZI heater
    for port in ref_mzi.ports:
        if 'e' in port.name or port.port_type == 'electrical':
            c.add_port(f"mzi_{port.name}", port=port)

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

# Create four additional MZI 1x2_1x2 circuits stacked along x
# Get MZI dimensions for spacing
test_mzi = mzi1x2_1x2_phase_shifter(
    cross_section='strip',
    length_y=0.0,
    length_x=150.0,
    delta_length=0.0,
)
mzi_width = test_mzi.xmax - test_mzi.xmin
mzi_height = test_mzi.ymax - test_mzi.ymin

# Create four MZIs stacked along x with consistent y offset to prevent overlap
x_start = -280  # Starting x position
x_spacing = mzi_width + 10  # Add 10 um spacing between MZIs along x
y_start = 280  # Starting y position
y_shift = mzi_height / 2 + 5  # Half height plus 5 um buffer for consistent y shift
y_shift = -y_shift
stacked_mzis = []
for i in range(4):
    mzi = c_chip << mzi1x2_1x2_phase_shifter(
        cross_section='strip',
        length_y=0.0,
        length_x=150.0,
        delta_length=0.0,
    )
    # Position the MZI: stack along x, with consistent y offset for each
    x_pos = x_start + i * x_spacing
    y_pos = y_start + i * y_shift  # Consistent downward shift
    mzi.move((x_pos, y_pos))
    stacked_mzis.append(mzi)

# Create four vertical heaters with same spacing as the cells
heater_length = 100  # Length of heated section
heaters = []

# Get the ports we want to connect to
ports_to_connect = [
    circuits[0].ports["o1"],  # First spiral input
    circuits[0].ports["o3"],  # MZI 1 output
    circuits[1].ports["o3"],  # MZI 2 output
    circuits[2].ports["o3"],  # MZI 3 output
]

# Create and position heaters vertically
for i, port in enumerate(ports_to_connect):
    heater = c_chip << gf.components.straight_heater_metal(
        length=heater_length,
        cross_section='strip'
    )
    # Rotate to make it vertical (90 degrees)
    heater.rotate(90)

    # Position the heater: place it offset from the port
    # Move heater so its o1 port is offset vertically from the target port
    offset_y = 50  # Offset distance in um
    heater.movex(port.x - 50).movey(port.y + offset_y)

    heaters.append(heater)

# Route from each port to its corresponding heater
for i, port in enumerate(ports_to_connect):
    gf.routing.route_single(
        c_chip,
        port1=port,
        port2=heaters[i].ports["o1"],
        cross_section='strip',
        bend='bend_euler'
    )

# Route from vertical heater outputs to stacked MZI inputs
# Based on x-ordering: leftmost heater to leftmost MZI, etc.
heater_to_mzi_connections = [
    (heaters[0].ports["o2"], stacked_mzis[0].ports["o1"]),  # heater_1 → stacked_mzi_1
    (heaters[1].ports["o2"], stacked_mzis[1].ports["o1"]),  # heater_2 → stacked_mzi_2
    (heaters[2].ports["o2"], stacked_mzis[2].ports["o1"]),  # heater_3 → stacked_mzi_3
    (heaters[3].ports["o2"], stacked_mzis[3].ports["o1"]),  # heater_4 → stacked_mzi_4
]

for heater_port, mzi_port in heater_to_mzi_connections:
    gf.routing.route_single(
        c_chip,
        port1=heater_port,
        port2=mzi_port,
        cross_section='strip',
        bend='bend_euler'
    )

# Export the chain's input and output ports
c_chip.add_port("input", port=heaters[0].ports["o1"])  # Input through first heater (changed from o2 to o1)
c_chip.add_port("output", port=waveguides[-1].ports["o2"])  # Last waveguide output (chain output)

# Export heater outputs (o2 ports)
for i, heater in enumerate(heaters):
    if i == 0:
        c_chip.add_port(f"heater_{i+1}_output", port=heater.ports["o2"])
    else:
        c_chip.add_port(f"mzi_{i}_heater_output", port=heater.ports["o2"])

# Export electrical ports for all standalone heaters
for i, heater in enumerate(heaters):
    # Each heater has electrical ports for the metal contacts
    for port in heater.ports:
        if 'e' in port.name:  # Electrical ports
            c_chip.add_port(f"heater_{i+1}_{port.name}", port=port)

# Export electrical ports from MZI heaters in each circuit
for i, circuit in enumerate(circuits):
    for port in circuit.ports:
        if 'mzi_e' in port.name:  # MZI heater electrical ports
            c_chip.add_port(f"circuit_{i+1}_{port.name}", port=port)

# Export electrical ports from stacked MZIs
for i, mzi in enumerate(stacked_mzis):
    for port in mzi.ports:
        if port.port_type == 'electrical':
            c_chip.add_port(f"stacked_mzi_{i+1}_{port.name}", port=port)

# Export optical ports from stacked MZIs
for i, mzi in enumerate(stacked_mzis):
    for port in mzi.ports:
        if port.port_type == 'optical':
            c_chip.add_port(f"stacked_mzi_{i+1}_{port.name}", port=port)

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

