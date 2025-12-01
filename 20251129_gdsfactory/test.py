

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
n_loops_list = [11, 10, 9]
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

# Extend stacked MZI o2 ports to x=710 um and add fan-in
target_x = 710.0
fanin_input_spacing = 28.5  # Same as the y spacing between stacked MZIs
fanin_output_spacing = 1.25  # 1.25 um output spacing (matches 4x2 MMI input port spacing)

# First, extend each stacked MZI o2 port to x=710 using straight waveguides
extended_ports = []
for i, mzi in enumerate(stacked_mzis):
    o2_port = mzi.ports['o2']

    # Calculate the length needed to reach x=710
    current_x = o2_port.x
    extension_length = target_x - current_x

    if extension_length > 0:
        # Create a straight waveguide to extend to x=710
        extension_wg = c_chip << gf.components.straight(
            length=extension_length,
            cross_section='strip'
        )
        extension_wg.connect('o1', o2_port)
        extended_ports.append(extension_wg.ports['o2'])
    else:
        # Already past x=710, just use the existing port
        extended_ports.append(o2_port)

# Now create the fan-in starting at x=710
# Get the y positions of the extended ports
y_positions = [port.y for port in extended_ports]

# Calculate the center y position for the fan-in output
y_center = sum(y_positions) / len(y_positions)
y_start_output = y_center - (len(extended_ports) - 1) * fanin_output_spacing / 2

# Create input waveguides for the fan-in (short straights at x=710)
fanin_input_wgs = []
for i, port in enumerate(extended_ports):
    wg = c_chip << gf.components.straight(length=30, cross_section='strip')
    wg.connect('o1', port)
    fanin_input_wgs.append(wg)

# Add a 4x2 MMI - we'll connect it directly to the fan-in S-bend outputs
mmi4x2 = c_chip << gf.components.mmi(inputs=4, outputs=2)

# Position the MMI after the fan-in input waveguides
# The fan-in input waveguides end at x=740, so place MMI further out
mmi4x2.movex(target_x + 30 + 200)  # Position after input wgs + some gap

# Calculate the target y positions for fan-in outputs (FLIPPED order)
# These will be the positions where the S-bend outputs should end up
fanin_output_y_positions = []
for i in range(len(extended_ports)):
    # Reversed order: start from highest y and go down
    y = y_start_output + (len(extended_ports) - 1 - i) * fanin_output_spacing
    fanin_output_y_positions.append(y)

# Center the MMI vertically with the calculated fan-in output positions
fanin_y_center = sum(fanin_output_y_positions) / len(fanin_output_y_positions)

# Get MMI input port y positions to calculate its center
mmi_input_ports = [mmi4x2.ports[f'o{i+1}'] for i in range(4)]
mmi_input_y_positions = [p.y for p in mmi_input_ports]
mmi_y_center = sum(mmi_input_y_positions) / len(mmi_input_y_positions)

# Move MMI to align centers
mmi4x2.movey(fanin_y_center - mmi_y_center)

# Now route directly from fan-in inputs to MMI inputs using S-bends
# The S-bends will handle the pitch transition AND connect to the MMI
# Connect with FLIPPED order to MMI inputs
fanin_routes = gf.routing.route_bundle_sbend(
    component=c_chip,
    ports1=[wg.ports['o2'] for wg in fanin_input_wgs],
    ports2=[mmi4x2.ports[f'o{i+1}'] for i in range(3, -1, -1)],  # Reversed: o4, o3, o2, o1
    cross_section='strip'
)

# Export optical ports from stacked MZIs (o1 ports only, since o2 are now connected)
for i, mzi in enumerate(stacked_mzis):
    c_chip.add_port(f"stacked_mzi_{i+1}_o1", port=mzi.ports['o1'])

# Add fan-out after 4x2 MMI to separate the waveguides more
# Fan-out with 5 um separation and max 20 um length along x
fanout_output_spacing = 5.0  # 5 um separation
fanout_length_x = 10.0  # 10 um along x for the S-bend transition

# Get MMI output ports directly
mmi_output_ports = [mmi4x2.ports['o5'], mmi4x2.ports['o6']]

# Calculate center y position for fan-out outputs
mmi_output_y_positions = [p.y for p in mmi_output_ports]
fanout_y_center = sum(mmi_output_y_positions) / len(mmi_output_y_positions)
fanout_total_output_span = (len(mmi_output_ports) - 1) * fanout_output_spacing
fanout_y_start_output = fanout_y_center - fanout_total_output_span / 2

# Create output waveguides for the fan-out (no extra straight)
# REVERSED order: start from highest y and go down to match MMI output order
fanout_output_wgs = []
for i in range(len(mmi_output_ports)):
    wg = c_chip << gf.components.straight(length=5, cross_section='strip')
    wg.movex(mmi4x2.ports['o5'].x + fanout_length_x)  # Position after fan-out transition
    # Reversed: highest y first (i=0 gets highest y position)
    wg.movey(fanout_y_start_output + (len(mmi_output_ports) - 1 - i) * fanout_output_spacing)
    fanout_output_wgs.append(wg)

# Create the fan-out using S-bend routing (max 20 um along x)
fanout_routes = gf.routing.route_bundle_sbend(
    component=c_chip,
    ports1=mmi_output_ports,
    ports2=[wg.ports['o1'] for wg in fanout_output_wgs],
    cross_section='strip'
)

# Export the fan-out output ports
c_chip.add_port("mmi4x2_output_1", port=fanout_output_wgs[0].ports['o2'])
c_chip.add_port("mmi4x2_output_2", port=fanout_output_wgs[1].ports['o2'])

# Add a set of 8 grating couplers at x=1100 um along y direction
# Waveguide output should be facing -x (180 degrees)
gc_array = c_chip << gf.components.grating_coupler_array(
    n=8,
    pitch=127,  # Standard pitch of 127 um
    rotation=-90  # Default rotation
)

# Position the grating coupler array at x=1100
# The array is created centered at origin with ports at 90 degrees (facing +y)
# We need to rotate it so ports face -x (180 degrees)
# Rotation from 90° to 180° requires +90° rotation
gc_array.rotate(90)
gc_array.movex(1100)
# Move the grating array along +y by one pitch (127 um)
gc_array.movey(127)

# Export grating coupler ports
for i in range(8):
    c_chip.add_port(f"gc_{i}", port=gc_array.ports[f'o{i}'])

# Route fan-out outputs to grating couplers
# After moving GC array by +127 um, the new positions are:
# gc_4: y = 190.5
# gc_5: y = 317.5
# CORRECTED: Fan-out output 0 (higher, y=239.75) → gc_5 (y=317.5), Fan-out output 1 (lower, y=234.75) → gc_4 (y=190.5)
gf.routing.route_single(
    c_chip,
    port1=fanout_output_wgs[0].ports['o2'],  # Higher fan-out output (y=239.75)
    port2=gc_array.ports['o5'],  # gc_5 (y=317.5)
    cross_section='strip',
    bend='bend_euler'
)

gf.routing.route_single(
    c_chip,
    port1=fanout_output_wgs[1].ports['o2'],  # Lower fan-out output (y=234.75)
    port2=gc_array.ports['o4'],  # gc_4 (y=190.5)
    cross_section='strip',
    bend='bend_euler'
)

# Route GC3 to the third MZI-spiral circuit's o2 port (the unused MZI output)
gf.routing.route_single(
    c_chip,
    port1=gc_array.ports['o3'],  # gc_3
    port2=circuits[2].ports['o2'],  # Third MZI-spiral circuit o2 (unused output)
    cross_section='strip',
    bend='bend_euler'
)

# Add loopback between gc_0 and gc_1
gf.routing.route_single(
    c_chip,
    port1=gc_array.ports['o0'],  # gc_0
    port2=gc_array.ports['o1'],  # gc_1
    cross_section='strip',
    bend='bend_euler'
)

# Add loopback between gc_6 and gc_7
gf.routing.route_single(
    c_chip,
    port1=gc_array.ports['o6'],  # gc_6
    port2=gc_array.ports['o7'],  # gc_7
    cross_section='strip',
    bend='bend_euler'
)

# ============================================================================
# CREATE BOND PADS FOR ELECTRICAL ROUTING
# ============================================================================

# Get current chip extent
chip_xmin = c_chip.xmin
chip_xmax = c_chip.xmax
chip_ymin = c_chip.ymin
chip_ymax = c_chip.ymax

print(f"\n=== Chip Extent (before bond pads) ===")
print(f"X: [{chip_xmin:.1f}, {chip_xmax:.1f}] um")
print(f"Y: [{chip_ymin:.1f}, {chip_ymax:.1f}] um")

# Bond pad parameters with 500 um buffer from pattern extent
edge_buffer = 500.0  # 500 um buffer from pattern extent
left_edge_x = chip_xmin - edge_buffer  # Left edge x-coordinate for bond pads
bottom_edge_y = chip_ymin - edge_buffer  # Bottom edge y-coordinate for bond pads
pad_pitch = 100.0  # 100 um pitch between bond pads
pad_size = 80.0  # 80 um x 80 um bond pads

print(f"\n=== Bond Pad Edge Positions (with {edge_buffer} um buffer) ===")
print(f"LEFT edge x: {left_edge_x:.1f} um")
print(f"BOTTOM edge y: {bottom_edge_y:.1f} um")

# Group heater ports by heater and end (left/right)
# Each heater has 2 ends, each end has 4 ports that should be merged into 1 bond pad

heater_groups = {
    # Standalone heaters (4 heaters x 2 ends = 8 bond pads)
    'heater_1_left': ['heater_1_l_e1', 'heater_1_l_e2', 'heater_1_l_e3', 'heater_1_l_e4'],
    'heater_1_right': ['heater_1_r_e1', 'heater_1_r_e2', 'heater_1_r_e3', 'heater_1_r_e4'],
    'heater_2_left': ['heater_2_l_e1', 'heater_2_l_e2', 'heater_2_l_e3', 'heater_2_l_e4'],
    'heater_2_right': ['heater_2_r_e1', 'heater_2_r_e2', 'heater_2_r_e3', 'heater_2_r_e4'],
    'heater_3_left': ['heater_3_l_e1', 'heater_3_l_e2', 'heater_3_l_e3', 'heater_3_l_e4'],
    'heater_3_right': ['heater_3_r_e1', 'heater_3_r_e2', 'heater_3_r_e3', 'heater_3_r_e4'],
    'heater_4_left': ['heater_4_l_e1', 'heater_4_l_e2', 'heater_4_l_e3', 'heater_4_l_e4'],
    'heater_4_right': ['heater_4_r_e1', 'heater_4_r_e2', 'heater_4_r_e3', 'heater_4_r_e4'],

    # Circuit MZI heaters (3 circuits x 2 ends = 6 bond pads)
    # Each MZI has 2 heater sections (e1,e3,e6,e8 and e2,e4,e5,e7)
    'circuit_1_mzi_1': ['circuit_1_mzi_e1', 'circuit_1_mzi_e3', 'circuit_1_mzi_e6', 'circuit_1_mzi_e8'],
    'circuit_1_mzi_2': ['circuit_1_mzi_e2', 'circuit_1_mzi_e4', 'circuit_1_mzi_e5', 'circuit_1_mzi_e7'],
    'circuit_2_mzi_1': ['circuit_2_mzi_e1', 'circuit_2_mzi_e3', 'circuit_2_mzi_e6', 'circuit_2_mzi_e8'],
    'circuit_2_mzi_2': ['circuit_2_mzi_e2', 'circuit_2_mzi_e4', 'circuit_2_mzi_e5', 'circuit_2_mzi_e7'],
    'circuit_3_mzi_1': ['circuit_3_mzi_e1', 'circuit_3_mzi_e3', 'circuit_3_mzi_e6', 'circuit_3_mzi_e8'],
    'circuit_3_mzi_2': ['circuit_3_mzi_e2', 'circuit_3_mzi_e4', 'circuit_3_mzi_e5', 'circuit_3_mzi_e7'],

    # Stacked MZI heaters (4 MZIs x 2 ends = 8 bond pads)
    'stacked_mzi_1_1': ['stacked_mzi_1_e1', 'stacked_mzi_1_e3', 'stacked_mzi_1_e6', 'stacked_mzi_1_e8'],
    'stacked_mzi_1_2': ['stacked_mzi_1_e2', 'stacked_mzi_1_e4', 'stacked_mzi_1_e5', 'stacked_mzi_1_e7'],
    'stacked_mzi_2_1': ['stacked_mzi_2_e1', 'stacked_mzi_2_e3', 'stacked_mzi_2_e6', 'stacked_mzi_2_e8'],
    'stacked_mzi_2_2': ['stacked_mzi_2_e2', 'stacked_mzi_2_e4', 'stacked_mzi_2_e5', 'stacked_mzi_2_e7'],
    'stacked_mzi_3_1': ['stacked_mzi_3_e1', 'stacked_mzi_3_e3', 'stacked_mzi_3_e6', 'stacked_mzi_3_e8'],
    'stacked_mzi_3_2': ['stacked_mzi_3_e2', 'stacked_mzi_3_e4', 'stacked_mzi_3_e5', 'stacked_mzi_3_e7'],
    'stacked_mzi_4_1': ['stacked_mzi_4_e1', 'stacked_mzi_4_e3', 'stacked_mzi_4_e6', 'stacked_mzi_4_e8'],
    'stacked_mzi_4_2': ['stacked_mzi_4_e2', 'stacked_mzi_4_e4', 'stacked_mzi_4_e5', 'stacked_mzi_4_e7'],
}

# Calculate average position for each group and determine which edge to route to
bond_pad_info = []
for group_name, port_names in heater_groups.items():
    # Get the ports
    ports = [c_chip.ports[name] for name in port_names if name in c_chip.ports]
    if not ports:
        continue

    # Calculate average position
    avg_x = sum(p.x for p in ports) / len(ports)
    avg_y = sum(p.y for p in ports) / len(ports)

    # Decide which edge: LEFT if x < 100, BOTTOM otherwise
    edge = 'LEFT' if avg_x < 100 else 'BOTTOM'

    bond_pad_info.append({
        'name': group_name,
        'ports': ports,
        'avg_x': avg_x,
        'avg_y': avg_y,
        'edge': edge
    })

# Sort by position
left_pads = sorted([p for p in bond_pad_info if p['edge'] == 'LEFT'], key=lambda p: p['avg_y'])
bottom_pads = sorted([p for p in bond_pad_info if p['edge'] == 'BOTTOM'], key=lambda p: p['avg_x'])

print(f"\n=== Bond Pad Assignment ===")
print(f"Total heater groups: {len(bond_pad_info)}")
print(f"LEFT edge bond pads: {len(left_pads)}")
print(f"BOTTOM edge bond pads: {len(bottom_pads)}")

# Create bond pads along LEFT edge
# Move down by 200 um to be closer to bottom-left corner
left_start_y = -200.0  # Starting y position for left edge pads

for i, pad_info in enumerate(left_pads):
    # Create a rectangular bond pad
    pad = c_chip << gf.components.rectangle(size=(pad_size, pad_size), layer='M3')
    pad_y = left_start_y + i * pad_pitch
    pad.move((left_edge_x, pad_y))

    # Add port to the bond pad for routing
    c_chip.add_port(f"bondpad_{pad_info['name']}",
                    center=(left_edge_x + pad_size/2, pad_y + pad_size/2),
                    width=40.0, orientation=0, layer='M3', port_type='electrical')

    # Store mapping for later routing
    pad_info['bondpad_name'] = f"bondpad_{pad_info['name']}"

print(f"Created {len(left_pads)} bond pads on LEFT edge")
print(f"  Y range: [{left_start_y:.1f}, {left_start_y + (len(left_pads)-1)*pad_pitch:.1f}]")

# Create bond pads along BOTTOM edge
# Move left by 200 um to be closer to bottom-left corner
bottom_start_x = 0.0  # Starting x position for bottom edge pads

for i, pad_info in enumerate(bottom_pads):
    # Create a rectangular bond pad
    pad = c_chip << gf.components.rectangle(size=(pad_size, pad_size), layer='M3')
    pad_x = bottom_start_x + i * pad_pitch
    pad.move((pad_x, bottom_edge_y))

    # Add port to the bond pad for routing
    c_chip.add_port(f"bondpad_{pad_info['name']}",
                    center=(pad_x + pad_size/2, bottom_edge_y + pad_size/2),
                    width=40.0, orientation=90, layer='M3', port_type='electrical')

    # Store mapping for later routing
    pad_info['bondpad_name'] = f"bondpad_{pad_info['name']}"

print(f"Created {len(bottom_pads)} bond pads on BOTTOM edge")
print(f"  X range: [{bottom_start_x:.1f}, {bottom_start_x + (len(bottom_pads)-1)*pad_pitch:.1f}]")

# ============================================================================
# ELECTRICAL ROUTING: Connect heater ports to bond pads
# ============================================================================

print(f"\n=== Electrical Routing ===")

# Routing parameters
metal_width = 15.0  # 15 um wide metal traces
metal_layer = (49, 0)  # M3 layer

# Route LEFT edge pads
# Strategy: Maintain Y-order to avoid crossings and overlaps
# 1. Collect all merge points and sort by Y
# 2. Assign each a unique Y-channel at intermediate X (spaced by metal_width)
# 3. Route: merge -> intermediate_x at unique Y -> bond pad's Y -> bond pad
print(f"Routing {len(left_pads)} groups to LEFT edge...")

# Define intermediate X position for LEFT edge routing (between ports and pads)
# This should be to the left of all heater ports
intermediate_x_left = left_edge_x + 200.0  # 200 um to the right of bond pads

# First pass: calculate all merge points and assign unique Y channels
left_routing_info = []
for i, pad_info in enumerate(left_pads):
    ports = pad_info['ports']
    bondpad_name = pad_info['bondpad_name']
    bondpad_port = c_chip.ports[bondpad_name]

    # Calculate merge point at average position
    merge_point_x = sum(p.x for p in ports) / len(ports)
    merge_point_y = sum(p.y for p in ports) / len(ports)

    left_routing_info.append({
        'pad_info': pad_info,
        'ports': ports,
        'bondpad_port': bondpad_port,
        'merge_x': merge_point_x,
        'merge_y': merge_point_y,
        'channel_y': bondpad_port.y,
        'sort_key': merge_point_y  # Sort by merge point Y
    })

# Sort by Y position to assign non-overlapping channels
left_routing_info.sort(key=lambda x: x['sort_key'])

# Assign unique Y positions at intermediate_x (spaced by metal_width + gap)
channel_spacing = metal_width + 5.0  # 5 um gap between traces
for idx, info in enumerate(left_routing_info):
    # Assign Y position at intermediate X, evenly spaced
    info['intermediate_y'] = info['merge_y'] + idx * channel_spacing - (len(left_routing_info) - 1) * channel_spacing / 2

# Second pass: do the actual routing
for info in left_routing_info:
    ports = info['ports']
    merge_point_x = info['merge_x']
    merge_point_y = info['merge_y']
    intermediate_y = info['intermediate_y']
    channel_y = info['channel_y']
    bondpad_port = info['bondpad_port']

    # Step 1: Route each port to merge point (short local connections)
    for port in ports:
        points = [
            (port.x, port.y),
            (merge_point_x, merge_point_y)
        ]
        path = gf.Path(points)
        trace = c_chip << path.extrude(width=metal_width, layer=metal_layer)

    # Step 2: Route from merge point to bond pad with Manhattan routing
    # Path: merge -> (merge_x, intermediate_y) -> (intermediate_x, intermediate_y) -> (intermediate_x, channel_y) -> bond pad
    route_points = [
        (merge_point_x, merge_point_y),        # Start at merge point
        (merge_point_x, intermediate_y),        # Go VERTICAL to intermediate Y level
        (intermediate_x_left, intermediate_y),  # Go HORIZONTAL to intermediate X
        (intermediate_x_left, channel_y),       # Go VERTICAL to bond pad's Y channel
        (bondpad_port.x, channel_y)             # Go HORIZONTAL to bond pad
    ]

    path = gf.Path(route_points)
    trace = c_chip << path.extrude(width=metal_width, layer=metal_layer)

print(f"  ✓ Routed {len(left_pads)} groups to LEFT edge")

# Route BOTTOM edge pads
# Strategy: Maintain X-order to avoid crossings and overlaps
# 1. Collect all merge points and sort by X
# 2. Assign each a unique X-channel at intermediate Y (spaced by metal_width)
# 3. Route: merge -> intermediate_y at unique X -> bond pad's X -> bond pad
print(f"Routing {len(bottom_pads)} groups to BOTTOM edge...")

# Define intermediate Y position for BOTTOM edge routing (between ports and pads)
# This should be below all heater ports
intermediate_y_bottom = bottom_edge_y + 200.0  # 200 um above bond pads

# First pass: calculate all merge points and assign unique X channels
bottom_routing_info = []
for i, pad_info in enumerate(bottom_pads):
    ports = pad_info['ports']
    bondpad_name = pad_info['bondpad_name']
    bondpad_port = c_chip.ports[bondpad_name]

    # Calculate merge point at average position
    merge_point_x = sum(p.x for p in ports) / len(ports)
    merge_point_y = sum(p.y for p in ports) / len(ports)

    bottom_routing_info.append({
        'pad_info': pad_info,
        'ports': ports,
        'bondpad_port': bondpad_port,
        'merge_x': merge_point_x,
        'merge_y': merge_point_y,
        'channel_x': bondpad_port.x,
        'sort_key': merge_point_x  # Sort by merge point X
    })

# Sort by X position to assign non-overlapping channels
bottom_routing_info.sort(key=lambda x: x['sort_key'])

# Assign unique X positions at intermediate_y (spaced by metal_width + gap)
channel_spacing = metal_width + 5.0  # 5 um gap between traces
for idx, info in enumerate(bottom_routing_info):
    # Assign X position at intermediate Y, evenly spaced
    info['intermediate_x'] = info['merge_x'] + idx * channel_spacing - (len(bottom_routing_info) - 1) * channel_spacing / 2

# Second pass: do the actual routing
for info in bottom_routing_info:
    ports = info['ports']
    merge_point_x = info['merge_x']
    merge_point_y = info['merge_y']
    intermediate_x = info['intermediate_x']
    channel_x = info['channel_x']
    bondpad_port = info['bondpad_port']

    # Step 1: Route each port to merge point (short local connections)
    for port in ports:
        points = [
            (port.x, port.y),
            (merge_point_x, merge_point_y)
        ]
        path = gf.Path(points)
        trace = c_chip << path.extrude(width=metal_width, layer=metal_layer)

    # Step 2: Route from merge point to bond pad with Manhattan routing
    # Path: merge -> (intermediate_x, merge_y) -> (intermediate_x, intermediate_y) -> (channel_x, intermediate_y) -> bond pad
    route_points = [
        (merge_point_x, merge_point_y),         # Start at merge point
        (intermediate_x, merge_point_y),        # Go HORIZONTAL to intermediate X level
        (intermediate_x, intermediate_y_bottom), # Go VERTICAL to intermediate Y
        (channel_x, intermediate_y_bottom),     # Go HORIZONTAL to bond pad's X channel
        (channel_x, bondpad_port.y)             # Go VERTICAL to bond pad
    ]

    path = gf.Path(route_points)
    trace = c_chip << path.extrude(width=metal_width, layer=metal_layer)

print(f"  ✓ Routed {len(bottom_pads)} groups to BOTTOM edge")
print(f"\n✓ Electrical routing complete!")
print(f"  Total traces: {len(left_pads) + len(bottom_pads)}")
print(f"  Metal width: {metal_width} μm")
print(f"  Metal layer: M3 ({metal_layer[0]}/{metal_layer[1]})")

# Print port information
print("\n=== Chip Ports ===")
c_chip.pprint_ports()

# Option 1: Add pins with port names and text labels (RECOMMENDED)
# c_chip_with_pins = add_pins_container(c_chip)
# c_chip_with_pins.write_gds("test.gds")

# Option 2: Just draw port markers on their layers (without text labels)
# c_chip.draw_ports()
c_chip.write_gds("test.gds")

# Show in KLayout
# c_chip_with_pins.show()

