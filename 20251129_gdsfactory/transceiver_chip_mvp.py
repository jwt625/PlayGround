"""
Silicon Photonics Transceiver Chip - MVP Layout
Mock transceiver with 8 wavelengths × 4 spatial channels (32 total channels)

This MVP places all components without routing connections.
Routing will be added in the next iteration.

Note: Uses custom AWG (awg_manual_route) instead of built-in gf.components.awg()
to ensure consistent path length differences and avoid waveguide crossings.
See silicon_photonics_transceiver_layout.md for AWG debug details.
"""

import gdsfactory as gf
from laser_bar_module import laser_bar, detector
from ring_double_pin import ring_double_pin
from custom_awg_debug import awg_manual_route


@gf.cell
def detector_array(num_detectors: int = 8, pitch: float = 50.0) -> gf.Component:
    """Create an array of detectors."""
    c = gf.Component()
    for i in range(num_detectors):
        det = c << detector(length=150.0)
        det.move((0, i * pitch))
    return c


@gf.cell
def transceiver_chip_mvp() -> gf.Component:
    """
    Create MVP transceiver chip with all components placed.
    No routing in this version - just component placement.
    """
    c = gf.Component("Transceiver_Chip_MVP")
    
    # ========================================================================
    # LEFT SIDE: LASER AND DETECTOR DIES
    # ========================================================================

    # Laser die (top-left)
    # Rotate 180° so outputs face right (default orientation is 180°, facing left)
    laser_die = c << laser_bar(
        num_emitters=8,
        emitter_pitch=50.0,
        emitter_length=300.0,
        add_boundary=False
    )
    laser_die.rotate(180)  # Rotate to face right
    laser_die.move((50, 1850))

    # Monitoring detector array (to the LEFT of laser die, not below)
    # Also rotate 180° so inputs face right (toward lasers)
    monitor_detectors = c << detector_array(num_detectors=8, pitch=50.0)
    monitor_detectors.rotate(180)  # Rotate to face right
    # Position to the left of laser die - laser is at x=50, so put monitors further left
    # But we want them to receive light from lasers, so actually position them to receive taps
    # For now, place them at negative x to be "left" of the laser
    monitor_detectors.move((-200, 1850))

    # RX Detector chips (bottom-left, 4 chips)
    # Rotate 180° so inputs face right (to receive light from AWGs)
    # All chips in same column (x=50) for parallel waveguide routing
    rx_det_positions = [
        (50, 50),    # Chip 1
        (50, 450),   # Chip 2
        (50, 850),   # Chip 3
        (50, 1250),  # Chip 4 - aligned in same column
    ]

    rx_detector_chips = []
    for pos in rx_det_positions:
        det_chip = c << detector_array(num_detectors=8, pitch=50.0)
        det_chip.rotate(180)  # Rotate to face right
        det_chip.move(pos)
        rx_detector_chips.append(det_chip)
    
    # ========================================================================
    # TRANSMIT PATH
    # ========================================================================
    
    # Stage 1: TX Multiplexing - 8 add-drop rings with PIN
    # Each laser couples to the ring from o4 port (top-right, "add" port)
    # Bus waveguide runs through bottom: o1 → o2
    # Arrange in 4x2 grid (4 rows, 2 columns) with 2x laser pitch (100 µm) row spacing
    tx_mux_rings = []

    # Grid configuration
    grid_rows = 4
    grid_cols = 2
    col_spacing = 150.0  # Horizontal spacing between columns

    # Starting position for grid
    grid_start_x = 300
    grid_start_y = 1850  # Align with top laser

    # Variable row spacing to provide more room for routing between rows 1 and 2
    # Row 0 at Y=1850, Row 1 at Y=1759 (91 µm spacing, reduced by 9 µm)
    # Row 2 at Y=1641 (118 µm spacing, increased by 18 µm)
    # Row 3 at Y=1550 (91 µm spacing, reduced by 9 µm)
    row_y_positions = [1850, 1750, 1601, 1500]

    print("\n" + "=" * 80)
    print("TX MUX RING POSITIONS AND PORTS")
    print("=" * 80)

    for i in range(8):
        row = i // grid_cols  # 0, 0, 1, 1, 2, 2, 3, 3
        col = i % grid_cols   # 0, 1, 0, 1, 0, 1, 0, 1

        ring = c << ring_double_pin(
            gap=0.2,
            radius=10.0,
            length_x=20.0,
            length_y=50.0,
            via_stack_width=10.0,
            pin_on_left=True
        )

        # Position in grid with variable row spacing
        ring_x = grid_start_x + col * col_spacing
        ring_y = row_y_positions[row]

        # Add y shift for second column rings
        if col == 1:
            if row % 2 == 0:  # Even rows: shift up
                ring_y += 3.0
            else:  # Odd rows: shift down
                ring_y -= 3.0

        ring.move((ring_x, ring_y))
        tx_mux_rings.append(ring)

        # Print ring position and port coordinates
        print(f"\nRing {i} @ ({ring_x}, {ring_y}) [Row {row}, Col {col}]:")
        for port_name in ['o1', 'o2', 'o3', 'o4']:
            port = ring.ports[port_name]
            print(f"  {port_name}: ({port.center[0]:.1f}, {port.center[1]:.1f}), orientation={port.orientation}°")

    print("=" * 80)
    
    # ========================================================================
    # ROUTING: Lasers to TX Mux Rings (CROSSING-FREE STRATEGY WITH WAYPOINTS)
    # ========================================================================

    # Crossing-free routing with manual waypoints:
    # - Lasers 0,1 route ABOVE rings 0,1 (row 0)
    # - Lasers 2,3 route BELOW rings 2,3 (row 1)
    # - Lasers 4,5 route ABOVE rings 4,5 (row 2)
    # - Lasers 6,7 route BELOW rings 6,7 (row 3)

    # Row 0: Lasers 0,1 → Rings 1,0 (route ABOVE the rings)
    # Ring positions: Row 0 at Y=1850, o3/o4 ports at ~Y=1921.4, o1/o2 ports at ~Y=1814.3
    # Horizontal routing must be ABOVE o3 ports (Y > 1921.4)

    laser_0_port = laser_die.ports["laser_0_o1"]
    ring_1_port = tx_mux_rings[1].ports["o3"]
    laser_1_port = laser_die.ports["laser_1_o1"]
    ring_0_port = tx_mux_rings[0].ports["o3"]

    # Routing parameters
    dx_laser_base = 30 + 125  # Horizontal offset from laser ports (shifted 125 µm right for compactness)
    dx_spacing = 15  # Horizontal spacing between vertical waveguide segments
    dx_ring_approach = 30  # Horizontal offset when approaching ring ports
    dy_spacing = 15  # Vertical spacing between horizontal routes

    # Laser 0 → Ring 1 o3 (crosses over to right, approaches o3 from left)
    dx_0 = dx_laser_base
    gf.routing.route_single(
        c,
        laser_0_port,
        ring_1_port,
        waypoints=[
            (laser_0_port.center[0] + dx_0, laser_0_port.center[1]),
            (laser_0_port.center[0] + dx_0, ring_1_port.center[1]),
        ],
        cross_section="strip",
    )

    # Laser 1 → Ring 0 o3 (goes to left, approaches o3 from left)
    dx_1 = dx_laser_base + dx_spacing
    gf.routing.route_single(
        c,
        laser_1_port,
        ring_0_port,
        waypoints=[
            (laser_1_port.center[0] + dx_1, laser_1_port.center[1]),
            (laser_1_port.center[0] + dx_1, ring_0_port.center[1]),
        ],
        cross_section="strip",
    )

    # Row 1: Lasers 2,3 → Rings 2,3 (route BELOW the rings)
    # Ring positions: Row 1 at Y=1759, o3/o4 at Y=1830.4, o1/o2 at Y=1759
    #                 Row 2 at Y=1641, o3/o4 at Y=1712.4, o1/o2 at Y=1641
    # Horizontal routing must be BELOW row 1's o1/o2 (< 1759) and ABOVE row 2's o3/o4 (> 1712.4)
    # Valid range: 1712.4 < Y < 1759 (46.6 µm of space - much better!)

    laser_2_port = laser_die.ports["laser_2_o1"]
    ring_2_port = tx_mux_rings[2].ports["o2"]
    laser_3_port = laser_die.ports["laser_3_o1"]
    ring_3_port = tx_mux_rings[3].ports["o2"]

    # Row 1 horizontal routing Y positions (between row 2's o3/o4 at 1712.4 and row 1's o1/o2 at 1759)
    # Laser 2 is at Y=1750, Laser 3 is at Y=1700
    # Available space: 1712.4 to 1759 = 46.6 µm
    # Need 20 µm clearance from laser positions and ring ports
    # Laser 2 at 1750: waypoint must be < 1730 (20 µm below laser) or > 1770 (20 µm above laser)
    # Laser 3 at 1700: waypoint must be > 1720 (20 µm above laser) or < 1680 (20 µm below laser)
    waypoint_y_row1_top = 1730  # For laser 2 (20 µm below laser at 1750, 17.6 µm above row 2 o3/o4)
    waypoint_y_row1_bottom = 1720  # For laser 3 (20 µm above laser at 1700, 7.6 µm above row 2 o3/o4)

    # Laser 2 → Ring 2 o2 (goes to left, reuse dx_1 from laser 1)
    gf.routing.route_single(
        c,
        laser_2_port,
        ring_2_port,
        waypoints=[
            (laser_2_port.center[0] + dx_1, laser_2_port.center[1]),
            (laser_2_port.center[0] + dx_1, waypoint_y_row1_top),
            (ring_2_port.center[0] + dx_ring_approach - 10, waypoint_y_row1_top),
            (ring_2_port.center[0] + dx_ring_approach - 10, ring_2_port.center[1]),
        ],
        cross_section="strip",
    )

    # Laser 3 → Ring 3 o2 (crosses over to right, reuse dx_0 from laser 0)
    gf.routing.route_single(
        c,
        laser_3_port,
        ring_3_port,
        waypoints=[
            (laser_3_port.center[0] + dx_0, laser_3_port.center[1]),
            (laser_3_port.center[0] + dx_0, waypoint_y_row1_bottom),
            (ring_3_port.center[0] + dx_ring_approach - 10, waypoint_y_row1_bottom),
            (ring_3_port.center[0] + dx_ring_approach - 10, ring_3_port.center[1]),
        ],
        cross_section="strip",
    )

    # Row 2: Lasers 4,5 → Rings 5,4 (route ABOVE the rings, similar to row 0)
    # Ring positions: Row 2 at Y=1601, o3/o4 at Y=1672.4, o1/o2 at Y=1601
    #                 Row 3 at Y=1500, o3/o4 at Y=1571.4, o1/o2 at Y=1500
    # Horizontal routing must be ABOVE row 2's o3/o4 ports (Y > 1672.4)

    laser_4_port = laser_die.ports["laser_4_o1"]
    ring_5_port = tx_mux_rings[5].ports["o3"]
    laser_5_port = laser_die.ports["laser_5_o1"]
    ring_4_port = tx_mux_rings[4].ports["o3"]

    # Laser 4 → Ring 5 o3 (crosses over to right, approaches o3 from left)
    gf.routing.route_single(
        c,
        laser_4_port,
        ring_5_port,
        waypoints=[
            (laser_4_port.center[0] + dx_0, laser_4_port.center[1]),
            (laser_4_port.center[0] + dx_0, ring_5_port.center[1]),
        ],
        cross_section="strip",
    )

    # Laser 5 → Ring 4 o3 (goes to left, approaches o3 from left)
    gf.routing.route_single(
        c,
        laser_5_port,
        ring_4_port,
        waypoints=[
            (laser_5_port.center[0] + dx_1, laser_5_port.center[1]),
            (laser_5_port.center[0] + dx_1, ring_4_port.center[1]),
        ],
        cross_section="strip",
    )

    # Row 3: Lasers 6,7 → Rings 6,7 (route BELOW the rings, similar to row 1)
    # Ring positions: Row 3 at Y=1500, o3/o4 at Y=1571.4, o1/o2 at Y=1500
    # No row 4 below, so horizontal routing can be freely below row 3's o1/o2 (< 1500)

    laser_6_port = laser_die.ports["laser_6_o1"]
    ring_6_port = tx_mux_rings[6].ports["o2"]
    laser_7_port = laser_die.ports["laser_7_o1"]
    ring_7_port = tx_mux_rings[7].ports["o2"]

    # Row 3 horizontal routing Y positions (below row 3's o1/o2 at 1500)
    # Laser 6 at Y=1550, Laser 7 at Y=1500
    # No row below, so just need > 20 µm clearance from row 3's o1/o2 and lasers
    waypoint_y_row3_top = 1480  # For laser 6 (20 µm below row 3 o1/o2)
    waypoint_y_row3_bottom = 1470  # For laser 7 (30 µm below row 3 o1/o2)

    # Laser 6 → Ring 6 o2 (goes to left, reuse dx_1 from laser 1)
    gf.routing.route_single(
        c,
        laser_6_port,
        ring_6_port,
        waypoints=[
            (laser_6_port.center[0] + dx_1, laser_6_port.center[1]),
            (laser_6_port.center[0] + dx_1, waypoint_y_row3_top),
            (ring_6_port.center[0] + dx_ring_approach - 10, waypoint_y_row3_top),
            (ring_6_port.center[0] + dx_ring_approach - 10, ring_6_port.center[1]),
        ],
        cross_section="strip",
    )

    # Laser 7 → Ring 7 o2 (crosses over to right, reuse dx_0 from laser 0)
    gf.routing.route_single(
        c,
        laser_7_port,
        ring_7_port,
        waypoints=[
            (laser_7_port.center[0] + dx_0, laser_7_port.center[1]),
            (laser_7_port.center[0] + dx_0, waypoint_y_row3_bottom),
            (ring_7_port.center[0] + dx_ring_approach - 10, waypoint_y_row3_bottom),
            (ring_7_port.center[0] + dx_ring_approach - 10, ring_7_port.center[1]),
        ],
        cross_section="strip",
    )

    # ========================================================================
    # ROUTING: Ring Bus Connections (SERPENTINE MULTIPLEXING CHAIN)
    # ========================================================================

    # Serpentine pattern to connect all rings and multiplex wavelengths
    # Path: Ring 1 → Ring 0 → Ring 2 → Ring 3 → Ring 5 → Ring 4 → Ring 6 → Ring 7 → Output

    ring_bus_connections = [
        (1, "o1", 0, "o2"),  # Ring 1 o1 → Ring 0 o2 (right to left, bottom bus)
        (0, "o1", 2, "o3"),  # Ring 0 o1 → Ring 2 o3 (left col, bottom to top, row 0→1)
        (2, "o4", 3, "o3"),  # Ring 2 o4 → Ring 3 o3 (left to right, top bus)
        (5, "o1", 4, "o2"),  # Ring 5 o1 → Ring 4 o2 (right to left, bottom bus)
        (4, "o1", 6, "o3"),  # Ring 4 o1 → Ring 6 o3 (left col, bottom to top, row 2→3)
        (6, "o4", 7, "o3"),  # Ring 6 o4 → Ring 7 o3 (left to right, top bus)
    ]

    for src_ring, src_port, dst_ring, dst_port in ring_bus_connections:
        gf.routing.route_single(
            c,
            tx_mux_rings[src_ring].ports[src_port],
            tx_mux_rings[dst_ring].ports[dst_port],
            cross_section="strip",
        )

    # Special routing for Ring 3 o4 → Ring 5 o2 with x-shift to avoid intersection
    r3_port = tx_mux_rings[3].ports["o4"]
    r5_port = tx_mux_rings[5].ports["o2"]
    x_shift = 25.0  # 25 µm shift to the right
    gf.routing.route_single(
        c,
        r3_port,
        r5_port,
        waypoints=[
            (r3_port.center[0] + x_shift, r3_port.center[1]),
            (r5_port.center[0] + x_shift, r5_port.center[1]),
        ],
        cross_section="strip",
    )

    # ========================================================================
    # TRANSMIT PATH (continued)
    # ========================================================================

    # Stage 2: Spatial splitting - 1x4 splitter
    splitter = c << gf.components.splitter_tree(noutputs=4, spacing=(90, 50))
    splitter.move((800, 2000))
    
    # Stage 3: TX Demultiplexing - 4 groups of 8 add-drop rings (32 total)
    # Each ring drops one wavelength from the bus to a separate waveguide
    tx_demux_groups = []
    demux_y_positions = [2100, 1700, 1300, 900]

    for _, y_base in enumerate(demux_y_positions):
        group_rings = []
        for i in range(8):
            ring = c << ring_double_pin(
                gap=0.2,
                radius=10.0,
                length_x=20.0,
                length_y=50.0,
                via_stack_width=10.0,
                pin_on_left=True
            )
            ring.move((1100, y_base + i * 80))  # Increased spacing for larger rings
            group_rings.append(ring)
        tx_demux_groups.append(group_rings)
    
    # Stage 4: Modulation - 32 PIN modulators
    tx_modulators = []
    for y_base in demux_y_positions:
        for i in range(8):
            mod = c << gf.components.straight_pin(length=100.0)
            mod.move((1400, y_base + i * 80))  # Match ring spacing
            tx_modulators.append(mod)
    
    # Stage 5: TX Recombining - 4 AWGs
    # Use custom AWG with manual routing for consistent path lengths and no crossings
    tx_awgs = []
    for y_pos in demux_y_positions:
        awg = c << awg_manual_route(
            arms=20,
            outputs=8,
            fpr_spacing=50.0,
            delta_length=10.0
        )
        awg.move((1700, y_pos))
        tx_awgs.append(awg)
    
    # Stage 6: TX Edge couplers
    tx_edge_couplers = []
    for y_pos in demux_y_positions:
        edge = c << gf.components.edge_coupler_silicon()
        edge.move((2700, y_pos))
        tx_edge_couplers.append(edge)
    
    # ========================================================================
    # RECEIVE PATH
    # ========================================================================
    
    # Stage 1: RX Edge couplers
    rx_y_positions = [600, 400, 200, 0]
    rx_edge_couplers = []
    for y_pos in rx_y_positions:
        edge = c << gf.components.edge_coupler_silicon()
        edge.move((2700, y_pos))
        rx_edge_couplers.append(edge)
    
    # Stage 2: RX Demultiplexing - 4 AWGs
    # Use custom AWG with manual routing for consistent path lengths and no crossings
    rx_awgs = []
    for y_pos in rx_y_positions:
        awg = c << awg_manual_route(
            arms=20,
            outputs=8,
            fpr_spacing=50.0,
            delta_length=10.0
        )
        awg.move((2400, y_pos))
        rx_awgs.append(awg)
    
    # ========================================================================
    # CHIP BOUNDARY
    # ========================================================================
    
    # Add chip boundary
    chip_width = 3000
    chip_height = 2500
    boundary = c << gf.components.rectangle(
        size=(chip_width, chip_height),
        layer="FLOORPLAN"
    )
    boundary.move((0, 0))
    
    return c


if __name__ == "__main__":
    print("Creating Silicon Photonics Transceiver Chip (MVP)...")
    print("=" * 80)

    c = transceiver_chip_mvp()

    # Write GDS
    gds_file = "transceiver_chip_mvp.gds"
    c.write_gds(gds_file)
    print(f"GDS file written: {gds_file}")

    # Print summary
    bbox = c.bbox()
    print(f"\nChip dimensions: {bbox.width():.1f} × {bbox.height():.1f} µm")
    print(f"Chip area: {bbox.width() * bbox.height() / 1e6:.2f} mm²")

    print("\nComponent count:")
    print(f"  - Laser emitters: 8")
    print(f"  - Monitoring detectors: 8")
    print(f"  - RX detectors: 32 (4 chips × 8)")
    print(f"  - TX mux rings: 8")
    print(f"  - TX demux rings: 32 (4 groups × 8)")
    print(f"  - PIN modulators: 32")
    print(f"  - AWGs: 8 (4 TX + 4 RX)")
    print(f"  - Edge couplers: 8 (4 TX + 4 RX)")
    print(f"  - 1×4 Splitter: 1")

    # Show in viewer
    try:
        c.show()
    except Exception as e:
        print(f"\nViewer error (GDS file is OK): {e}")

    print("\n" + "=" * 80)
    print("MVP LAYOUT COMPLETE - All components placed")
    print("Next step: Add waveguide routing between components")
    print("=" * 80)

