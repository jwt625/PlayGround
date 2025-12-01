"""
Example of Fan-In and Fan-Out using gdsfactory

This demonstrates how to create pitch transitions for waveguide arrays,
commonly used in phased arrays, fiber coupling, and similar applications.
"""

import gdsfactory as gf


def create_fanin_sbend(
    n_waveguides: int = 8,
    input_pitch: float = 20.0,
    output_pitch: float = 5.0,
    length: float = 150.0,
    wg_length: float = 30.0,
    name: str = None
) -> gf.Component:
    """
    Create a fan-in using S-bends (smooth pitch transition).

    Args:
        n_waveguides: Number of waveguides in the array
        input_pitch: Input spacing between waveguides (um)
        output_pitch: Output spacing between waveguides (um)
        length: Length of the transition region (um)
        wg_length: Length of straight waveguides at input/output (um)
        name: Optional custom name for the component

    Returns:
        Component with fan-in transition
    """
    if name is None:
        name = f"fanin_sbend_{n_waveguides}wg_{input_pitch}to{output_pitch}"
    c = gf.Component(name)
    
    # Create input waveguides with wide spacing
    input_wgs = []
    for i in range(n_waveguides):
        wg = c << gf.components.straight(length=wg_length, cross_section='strip')
        wg.movey(i * input_pitch)
        input_wgs.append(wg)
    
    # Create output waveguides with tight spacing (centered)
    y_center = (n_waveguides - 1) * input_pitch / 2
    y_start = y_center - (n_waveguides - 1) * output_pitch / 2
    
    output_wgs = []
    for i in range(n_waveguides):
        wg = c << gf.components.straight(length=wg_length, cross_section='strip')
        wg.movex(length).movey(y_start + i * output_pitch)
        output_wgs.append(wg)
    
    # Get the ports
    input_ports = [wg.ports['o2'] for wg in input_wgs]
    output_ports = [wg.ports['o1'] for wg in output_wgs]
    
    # Route with S-bends for smooth transition
    routes = gf.routing.route_bundle_sbend(
        component=c,
        ports1=input_ports,
        ports2=output_ports,
        cross_section='strip'
    )
    
    # Add ports for the component
    for i, wg in enumerate(input_wgs):
        c.add_port(f"in_{i}", port=wg.ports['o1'])
    for i, wg in enumerate(output_wgs):
        c.add_port(f"out_{i}", port=wg.ports['o2'])
    
    return c


def create_fanout_sbend(
    n_waveguides: int = 8,
    input_pitch: float = 5.0,
    output_pitch: float = 20.0,
    length: float = 150.0,
    wg_length: float = 30.0,
    name: str = None
) -> gf.Component:
    """
    Create a fan-out using S-bends (smooth pitch transition).

    This is just a fan-in with reversed input/output pitches.
    """
    if name is None:
        name = f"fanout_sbend_{n_waveguides}wg_{input_pitch}to{output_pitch}"
    return create_fanin_sbend(
        n_waveguides=n_waveguides,
        input_pitch=input_pitch,
        output_pitch=output_pitch,
        length=length,
        wg_length=wg_length,
        name=name
    )


if __name__ == "__main__":
    # Example 1: Fan-in (wide to narrow)
    print("Creating fan-in example (20 um -> 5 um pitch)...")
    fanin = create_fanin_sbend(
        n_waveguides=8,
        input_pitch=20.0,
        output_pitch=5.0,
        length=150.0
    )
    fanin.write_gds("fanin_clean.gds")
    print(f"  Saved to: fanin_clean.gds")
    print(f"  Ports: {len(fanin.ports)} total")
    
    # Example 2: Fan-out (narrow to wide)
    print("\nCreating fan-out example (5 um -> 20 um pitch)...")
    fanout = create_fanout_sbend(
        n_waveguides=8,
        input_pitch=5.0,
        output_pitch=20.0,
        length=150.0
    )
    fanout.write_gds("fanout_clean.gds")
    print(f"  Saved to: fanout_clean.gds")
    
    # Example 3: Phased array application
    print("\nCreating phased array example...")
    c_phased = gf.Component("phased_array_example")
    
    # Fan-out from single fiber to array
    fanout_ref = c_phased << fanout
    
    # Add some phase shifters (represented by longer waveguides)
    phase_shifters = []
    for i in range(8):
        ps = c_phased << gf.components.straight(length=100, cross_section='strip')
        ps.connect('o1', fanout_ref.ports[f'out_{i}'])
        phase_shifters.append(ps)
    
    # Fan-in back to tight spacing for emission
    fanin_ref = c_phased << fanin
    fanin_ref.movex(fanout_ref.xmax + 100)
    
    c_phased.write_gds("phased_array_example.gds")
    print(f"  Saved to: phased_array_example.gds")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("gdsfactory provides excellent fan-in/fan-out capabilities:")
    print("  • gf.routing.route_bundle_sbend() - Clean S-bend transitions")
    print("  • gf.routing.route_bundle() - Manhattan routing with bends")
    print("  • gf.routing.fanout2x2() - Specific for 2x2 components")
    print("\nFor phased arrays, use route_bundle_sbend() for smooth,")
    print("low-loss pitch transitions without waveguide crossings.")

