"""
Demux + Modulation Module

This module creates a reusable demux and modulation unit that will be repeated 4 times
in the transceiver chip. Each module:
- Takes 1 input waveguide carrying 8 multiplexed wavelengths
- Demultiplexes using 8 add-drop ring resonators with PIN
- Modulates each wavelength using 8 PIN modulators
- Outputs 8 separate modulated waveguides

Layout:
- 8 rings in 1 row × 8 columns (horizontal arrangement)
- 8 modulators in 1 row × 8 columns (horizontal arrangement, below rings)
- 110 µm horizontal pitch (matching PIN modulator length)
- 3 µm cumulative Y offset between rings for routing clearance
- 7 µm cumulative Y offset between modulators for routing clearance
"""

import gdsfactory as gf
from ring_double_pin import ring_double_pin


@gf.cell
def demux_modulation_module(
    base_x: float = 0.0,
    base_y: float = 0.0,
    ring_pitch_x: float = 110.0,
    ring_y_offset: float = 7.0,
    modulator_pitch_x: float = 110.0,
    modulator_y_offset: float = 7.0,
    ring_to_modulator_dy: float = 50.0,
) -> gf.Component:
    """
    Create a demux + modulation module with 8 rings and 8 modulators.
    
    Args:
        base_x: Base X position for the module
        base_y: Base Y position for the module (Ring 0 position)
        ring_pitch_x: Horizontal spacing between rings (center-to-center)
        ring_y_offset: Cumulative Y offset between adjacent rings (each ring is offset by this amount)
        modulator_pitch_x: Horizontal spacing between modulators
        modulator_y_offset: Cumulative Y offset between adjacent modulators (each modulator is offset by this amount)
        ring_to_modulator_dy: Vertical separation from rings to modulators
        
    Returns:
        Component with:
        - Input port: bus_input (right side, for input waveguide)
        - Output ports: mod_0_out to mod_7_out (modulator outputs)
    """
    c = gf.Component("Demux_Modulation_Module")
    
    # ========================================================================
    # PLACE RING ARRAY (1×8 horizontal)
    # ========================================================================

    rings = []
    for i in range(8):
        ring = c << ring_double_pin(
            gap=0.2,
            radius=10.0,
            length_x=20.0,
            length_y=50.0,
            via_stack_width=10.0,
            pin_on_left=True
        )

        # Calculate position with cumulative Y offset
        ring_x = base_x + i * ring_pitch_x
        ring_y = base_y + i * ring_y_offset

        ring.move((ring_x, ring_y))
        rings.append(ring)

    # ========================================================================
    # PLACE MODULATOR ARRAY (1×8 horizontal)
    # ========================================================================

    modulators = []
    for i in range(8):
        mod = c << gf.components.straight_pin(length=100.0)

        # Calculate position with cumulative Y offset
        mod_x = base_x + i * modulator_pitch_x
        mod_y = base_y - ring_to_modulator_dy + i * modulator_y_offset

        mod.move((mod_x, mod_y))
        modulators.append(mod)
    
    # ========================================================================
    # ROUTING: BUS WAVEGUIDE (TOP BUS, RIGHT TO LEFT)
    # ========================================================================

    # Input from right side connects to Ring 7 o4
    # Path: Input → Ring 7 o4 → Ring 7 o3 → Ring 6 o4 → ... → Ring 0 o3

    # Create input port on the right side (aligned with Ring 7 o4)
    ring_7_o4 = rings[7].ports["o4"]
    c.add_port("bus_input", port=ring_7_o4)

    # Route through all rings using S-bends for Y offset
    for i in range(7, 0, -1):  # Ring 7 to Ring 1
        src_port = rings[i].ports["o3"]
        dst_port = rings[i-1].ports["o4"]

        # Calculate S-bend size (horizontal and vertical distance)
        dx = dst_port.center[0] - src_port.center[0]
        dy = dst_port.center[1] - src_port.center[1]

        # Use S-bend to connect between rings with Y offset
        sbend = c << gf.components.bend_s(
            size=(dx, dy),
            cross_section="strip",
        )
        sbend.connect("o1", src_port)
    
    # ========================================================================
    # ROUTING: DROP WAVEGUIDES (RING TO MODULATOR)
    # ========================================================================
    
    # Each ring's drop port (o1 or o2) connects to corresponding modulator input
    # Need to determine which port to use based on ring configuration
    
    for i in range(8):
        # Use o2 port (bottom-right, through port) for drop connection
        ring_drop_port = rings[i].ports["o2"]
        mod_input_port = modulators[i].ports["o1"]
        
        # Route with waypoints to go down from ring to modulator
        gf.routing.route_single(
            c,
            ring_drop_port,
            mod_input_port,
            cross_section="strip",
        )
    
    # ========================================================================
    # EXTEND MODULATOR OUTPUTS TO ALIGNED X POSITION
    # ========================================================================

    # Find the maximum X coordinate from all modulator output ports
    x_max = max(mod.ports["o2"].center[0] for mod in modulators)
    x_ports = x_max + 10.0  # Add 10 µm margin

    # Add straight waveguides to extend all outputs to the same X position
    extended_ports = []
    for i in range(8):
        mod_out_port = modulators[i].ports["o2"]
        extension_length = x_ports - mod_out_port.center[0]

        if extension_length > 0:
            # Create straight waveguide extension
            extension = c << gf.components.straight(
                length=extension_length,
                cross_section="strip"
            )
            extension.connect("o1", mod_out_port)
            extended_ports.append(extension.ports["o2"])
        else:
            # No extension needed, just use the port directly
            extended_ports.append(mod_out_port)

    # ========================================================================
    # FAN-IN: ROUTE TO ALIGNED OUTPUT PORTS
    # ========================================================================

    # Add short straight waveguides after the aligned ports for fan-in input
    fanin_input_length = 30.0
    fanin_input_wgs = []
    for i in range(8):
        wg = c << gf.components.straight(length=fanin_input_length, cross_section="strip")
        wg.connect("o1", extended_ports[i])
        fanin_input_wgs.append(wg)

    # Calculate target positions for fan-in outputs (tighter spacing)
    # Get the Y positions of the fan-in inputs
    y_positions = [wg.ports["o2"].center[1] for wg in fanin_input_wgs]
    y_center = sum(y_positions) / len(y_positions)

    # Target output spacing (1 µm)
    fanin_output_spacing = 1.0
    fanin_total_output_span = (len(fanin_input_wgs) - 1) * fanin_output_spacing
    fanin_y_start_output = y_center - fanin_total_output_span / 2

    # Create output waveguides for the fan-in
    fanin_output_wgs = []
    fanin_output_x = x_ports + fanin_input_length + 50.0  # Position after fan-in transition
    for i in range(8):
        wg = c << gf.components.straight(length=5, cross_section="strip")
        wg.movex(fanin_output_x)
        wg.movey(fanin_y_start_output + i * fanin_output_spacing)
        fanin_output_wgs.append(wg)

    # Route using S-bend bundle
    gf.routing.route_bundle_sbend(
        component=c,
        ports1=[wg.ports["o2"] for wg in fanin_input_wgs],
        ports2=[wg.ports["o1"] for wg in fanin_output_wgs],
        cross_section="strip"
    )

    # Expose the fan-in output ports
    for i in range(8):
        c.add_port(f"mod_{i}_out", port=fanin_output_wgs[i].ports["o2"])

    return c


if __name__ == "__main__":
    print("Creating Demux + Modulation Module...")
    print("=" * 80)
    
    c = demux_modulation_module(
        base_x=100.0,
        base_y=500.0,
    )
    
    # Write GDS
    gds_file = "demux_modulation_module.gds"
    c.write_gds(gds_file)
    print(f"GDS file written: {gds_file}")
    
    # Print summary
    bbox = c.bbox()
    print(f"\nModule dimensions: {bbox.width():.1f} × {bbox.height():.1f} µm")
    
    print(f"\nPorts:")
    print(f"  - Input: bus_input")
    print(f"  - Outputs: mod_0_out to mod_7_out")
    print(f"  - Total ports: {len(c.ports)}")
    
    # Show in viewer
    try:
        c.show()
    except Exception as e:
        print(f"\nViewer error (GDS file is OK): {e}")
    
    print("\n" + "=" * 80)
    print("MODULE COMPLETE")
    print("=" * 80)

