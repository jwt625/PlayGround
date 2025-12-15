"""
Demux + Modulation + AWG Module

This module creates a reusable demux, modulation, and recombination unit that will be
repeated 4 times in the transceiver chip. Each module:
- Takes 1 input waveguide carrying 8 multiplexed wavelengths
- Demultiplexes using 8 add-drop ring resonators with PIN
- Modulates each wavelength using 8 PIN modulators
- Recombines all 8 modulated wavelengths using an AWG
- Outputs 1 waveguide carrying all 8 modulated wavelengths

Layout:
- 8 rings in 1 row × 8 columns (horizontal arrangement)
- 8 modulators in 1 row × 8 columns (horizontal arrangement, below rings)
- 110 µm horizontal pitch (matching PIN modulator length)
- 7 µm cumulative Y offset between rings for routing clearance
- 7 µm cumulative Y offset between modulators for routing clearance
- AWG positioned to the right of modulators for recombination
"""

import gdsfactory as gf
from ring_double_pin import ring_double_pin
from custom_awg_debug import awg_manual_route


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
    Create a demux + modulation + AWG module with 8 rings, 8 modulators, and 1 AWG.

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
        - Input port: bus_input (right side, for input waveguide carrying 8 wavelengths)
        - Output port: awg_out (AWG output carrying all 8 recombined modulated wavelengths)
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
    # ROUTING: BUS WAVEGUIDE INPUT (TOP BUS, WITH UPWARD BEND AND S-BEND)
    # ========================================================================

    # Route Ring 7 o4 upward and to the left for chip-level routing ease
    # Path: bus_input → S-bend (left and down) → 180° bend up → Ring 7 o4

    ring_7_o4 = rings[7].ports["o4"]

    # Add 180° Euler bend to turn upward from Ring 7 o4
    bend_radius = 10.0
    bend_180 = c << gf.components.bend_euler(radius=bend_radius, angle=180)
    bend_180.connect("o1", ring_7_o4)

    # After 180° bend, the output port is pointing left (180°) and positioned above Ring 7 o4
    # Calculate S-bend parameters to route left and down to align with Ring 0 level

    # Vertical offset: Ring 7 is at base_y + 7*ring_y_offset, need to go down to base_y level
    # Plus the height added by the 180° bend (2 * bend_radius)
    vertical_offset = 7 * ring_y_offset + 2 * bend_radius - 20  # 49 + 20 = 69 µm down

    # Horizontal distance: route to the left side (align with Ring 0 or further left)
    # Ring 7 is at base_x + 7*ring_pitch_x, route to base_x - some margin
    horizontal_distance = -(7 * ring_pitch_x + 50.0)  # -(770 + 50) = -820 µm (negative = left)

    # S-bend to route left and down from the 180° bend output
    # The bend output is pointing left (180°), so S-bend goes left and down
    sbend_input = c << gf.components.bend_s(
        size=(horizontal_distance, -vertical_offset),
        cross_section="strip",
    )
    sbend_input.connect("o1", bend_180.ports["o2"])

    # Expose the S-bend output as the module input port
    c.add_port("bus_input", port=sbend_input.ports["o2"])

    # ========================================================================
    # ROUTING: BUS WAVEGUIDE (TOP BUS, RIGHT TO LEFT)
    # ========================================================================

    # Route through all rings using S-bends for Y offset
    # Path: Ring 7 o4 → Ring 7 o3 → Ring 6 o4 → ... → Ring 0 o3
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
    # AWG: RECOMBINE MODULATED WAVELENGTHS
    # ========================================================================

    # Create AWG with 8 inputs (for 8 wavelengths) and 20 arms
    awg = c << awg_manual_route(
        arms=20,
        outputs=8,
        fpr_spacing=50.0,
        delta_length=10.0
    )

    # Rotate and mirror AWG so E ports point left (180°) to face modulator outputs
    awg.rotate(90)
    awg.mirror()  # Mirror along Y-axis to make E ports point left
    awg.mirror((1, 0))  # Mirror along X-axis (horizontal line) to flip port order vertically

    # Position AWG to the right of modulator outputs (max 50 µm away)
    awg_x_offset = 50.0  # Distance from modulator outputs to AWG

    # Calculate modulator output center
    mod_output_y_center = sum(port.center[1] for port in extended_ports) / len(extended_ports)
    mod_output_x = extended_ports[0].center[0]

    # Calculate AWG E port center
    awg_e_port_y_center = sum(awg.ports[f"E{i}"].center[1] for i in range(8)) / 8
    awg_e_port_x = awg.ports["E0"].center[0]

    # Position AWG: align Y centers and place to the right
    awg.move((
        mod_output_x + awg_x_offset - awg_e_port_x,
        mod_output_y_center - awg_e_port_y_center
    ))

    # ========================================================================
    # ROUTING: MODULATOR OUTPUTS TO AWG INPUTS
    # ========================================================================

    # Route directly from modulator outputs to AWG inputs using S-bend bundle
    awg_ports = [awg.ports[f"E{i}"] for i in range(8)]

    gf.routing.route_bundle_sbend(
        component=c,
        ports1=extended_ports,
        ports2=awg_ports,
        cross_section="strip"
    )

    # ========================================================================
    # EXPOSE OUTPUT PORT
    # ========================================================================

    # Expose AWG output port (o1) as module output
    c.add_port("awg_out", port=awg.ports["o1"])

    return c


if __name__ == "__main__":
    print("Creating Demux + Modulation + AWG Module...")
    print("=" * 80)

    c = demux_modulation_module(
        base_x=100.0,
        base_y=500.0,
    )

    # Write GDS
    gds_file = "demux_modulation_module.gds"
    c.write_gds(gds_file)
    print(f"\nGDS file written: {gds_file}")

    # Print summary
    bbox = c.bbox()
    print(f"\nModule dimensions: {bbox.width():.1f} × {bbox.height():.1f} µm")

    print(f"\nPorts:")
    print(f"  - Input: bus_input (8 wavelengths)")
    print(f"  - Output: awg_out (8 recombined modulated wavelengths)")
    print(f"  - Total ports: {len(c.ports)}")

    print(f"\nPort details:")
    for port in c.ports:
        print(f"  {port.name}: center=({port.center[0]:.1f}, {port.center[1]:.1f}), orientation={port.orientation}°")

    # Show in viewer
    try:
        c.show()
    except Exception as e:
        print(f"\nViewer error (GDS file is OK): {e}")

    print("\n" + "=" * 80)
    print("MODULE COMPLETE")
    print("=" * 80)

