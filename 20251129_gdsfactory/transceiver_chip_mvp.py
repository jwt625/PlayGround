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
    rx_det_positions = [
        (50, 50),    # Chip 1
        (50, 450),   # Chip 2
        (50, 850),   # Chip 3
        (250, 50),   # Chip 4
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
    # Each laser couples to the ring, drops to the bus waveguide
    tx_mux_rings = []
    for i in range(8):
        ring = c << ring_double_pin(
            gap=0.2,
            radius=10.0,
            length_x=20.0,
            length_y=50.0,
            via_stack_width=10.0,
            pin_on_left=True
        )
        ring.move((500, 1850 + i * 80))  # Increased spacing for larger rings
        tx_mux_rings.append(ring)
    
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

