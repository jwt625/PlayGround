"""
Test script to connect demux_modulation_module output to AWG input.

This script tests the connection between:
- Demux module fan-in outputs (mod_0_out to mod_7_out, 1 µm pitch)
- AWG output ports (E0 to E7)

We need to determine the correct AWG orientation (rotation/mirror) to connect properly.
"""

import gdsfactory as gf
from demux_modulation_module import demux_modulation_module
from custom_awg_debug import awg_manual_route


@gf.cell
def test_demux_awg_connection() -> gf.Component:
    """Test connection between demux module and AWG."""
    c = gf.Component("Test_Demux_AWG_Connection")
    
    # Create demux module
    demux = c << demux_modulation_module(
        base_x=100.0,
        base_y=500.0,
    )
    
    print("\n" + "=" * 80)
    print("DEMUX MODULE OUTPUT PORTS")
    print("=" * 80)
    for i in range(8):
        port = demux.ports[f"mod_{i}_out"]
        print(f"  mod_{i}_out: center=({port.center[0]:.1f}, {port.center[1]:.1f}), orientation={port.orientation}°")
    
    # Create AWG (default orientation)
    awg = c << awg_manual_route(
        arms=20,
        outputs=8,
        fpr_spacing=50.0,
        delta_length=10.0
    )

    print("\n" + "=" * 80)
    print("AWG PORT ANALYSIS (Default Orientation)")
    print("=" * 80)
    print(f"\nAWG input port (o1):")
    print(f"  center: ({awg.ports['o1'].center[0]:.1f}, {awg.ports['o1'].center[1]:.1f})")
    print(f"  orientation: {awg.ports['o1'].orientation}°")

    print(f"\nAWG output ports (E0-E7) before rotation:")
    for i in range(8):
        port = awg.ports[f"E{i}"]
        print(f"  E{i}: center=({port.center[0]:.1f}, {port.center[1]:.1f}), orientation={port.orientation}°")

    # Rotate AWG 90° counterclockwise so E ports are vertical
    print("\n⚠️  Rotating AWG 90° counterclockwise...")
    awg.rotate(90)

    print(f"\nAWG output ports (E0-E7) after 90° rotation:")
    for i in range(8):
        port = awg.ports[f"E{i}"]
        print(f"  E{i}: center=({port.center[0]:.1f}, {port.center[1]:.1f}), orientation={port.orientation}°")

    # Mirror AWG along Y-axis so E ports point left (180°)
    print("\n⚠️  Mirroring AWG along Y-axis to make E ports point left (180°)...")
    awg.mirror()

    print(f"\nAWG output ports (E0-E7) after mirroring:")
    for i in range(8):
        port = awg.ports[f"E{i}"]
        print(f"  E{i}: center=({port.center[0]:.1f}, {port.center[1]:.1f}), orientation={port.orientation}°")

    # Position AWG to the right of demux module
    # Start with a reasonable offset
    awg_x_offset = 100.0  # Distance from demux outputs to AWG

    # Get demux output positions
    demux_output_x = demux.ports["mod_0_out"].center[0]
    demux_output_y_center = sum(demux.ports[f"mod_{i}_out"].center[1] for i in range(8)) / 8

    # Calculate AWG output center (after rotation, E ports are now vertical)
    awg_output_y_center = sum(awg.ports[f"E{i}"].center[1] for i in range(8)) / 8
    awg_output_x = awg.ports["E0"].center[0]

    print(f"\nAWG output center Y: {awg_output_y_center:.1f}")
    print(f"Demux output center Y: {demux_output_y_center:.1f}")

    # Position AWG: align Y centers and place to the right
    awg.move((
        demux_output_x + awg_x_offset - awg_output_x,
        demux_output_y_center - awg_output_y_center
    ))

    print("\n" + "=" * 80)
    print("AWG POSITIONED - Checking port orientations")
    print("=" * 80)
    print(f"\nAWG output ports after positioning:")
    for i in range(8):
        port = awg.ports[f"E{i}"]
        print(f"  E{i}: center=({port.center[0]:.1f}, {port.center[1]:.1f}), orientation={port.orientation}°")

    # Check if we need to mirror/rotate AWG
    # Demux outputs point right (0°), so AWG outputs should point left (180°) to face them
    awg_e0_orientation = awg.ports["E0"].orientation

    print(f"\nOrientation check:")
    print(f"  Demux outputs orientation: 0° (pointing right)")
    print(f"  AWG E ports orientation: {awg_e0_orientation}° (should be 180° to face demux)")

    if awg_e0_orientation == 0:
        print("\n⚠️  AWG needs to be mirrored! E ports are pointing right (0°), should point left (180°)")
        print("  Solution: Mirror AWG along Y-axis")
    elif awg_e0_orientation == 180:
        print("\n✓ AWG orientation is correct! E ports point left (180°) toward demux outputs")

        # Now try to route the connections
        print("\n" + "=" * 80)
        print("ROUTING: Demux outputs to AWG inputs")
        print("=" * 80)

        # Get the ports to connect
        demux_ports = [demux.ports[f"mod_{i}_out"] for i in range(8)]
        awg_ports = [awg.ports[f"E{i}"] for i in range(8)]

        # Route using bundle
        print(f"\nRouting {len(demux_ports)} connections...")
        gf.routing.route_bundle(
            c,
            demux_ports,
            awg_ports,
            cross_section="strip",
        )
        print("✓ Routing complete!")
    
    return c


if __name__ == "__main__":
    print("Testing Demux Module to AWG Connection...")
    print("=" * 80)
    
    c = test_demux_awg_connection()
    
    # Write GDS
    gds_file = "test_demux_awg_connection.gds"
    c.write_gds(gds_file)
    print(f"\n\nGDS file written: {gds_file}")
    
    # Print summary
    bbox = c.bbox()
    print(f"\nTest layout dimensions: {bbox.width():.1f} × {bbox.height():.1f} µm")
    
    # Show in viewer
    try:
        c.show()
    except Exception as e:
        print(f"\nViewer error (GDS file is OK): {e}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

