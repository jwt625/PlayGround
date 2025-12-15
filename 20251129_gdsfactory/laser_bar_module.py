"""
Laser Bar Module for Photonic Integrated Circuits

This module creates a realistic 8-channel laser bar array based on typical
commercial laser diode bar specifications.

Typical laser bar specifications (based on research):
- Bar length: 10 mm
- Number of emitters: 8-50 (we use 8 for this example)
- Emitter width: 90-150 µm (we use 100 µm)
- Emitter pitch (center-to-center): 500 µm (standard)
- Emitter length (cavity): ~1000 µm (1 mm)

Since gdsfactory doesn't have native laser components, we mock them using
the existing photodetector component which has similar geometric layout.
"""

import gdsfactory as gf


@gf.cell
def laser_bar(
    num_emitters: int = 8,
    emitter_pitch: float = 50.0,
    emitter_length: float = 300.0,
    add_labels: bool = True,
    add_boundary: bool = True,
    boundary_margin: float = 50.0,
) -> gf.Component:
    """
    Laser bar array with multiple emitters (mocked using photodetectors).

    Each emitter has:
    - 1 optical output port (o1) - the laser output
    - 8 electrical ports (top_e1-4, bot_e1-4) - for current injection

    Module-level ports:
    - Optical: laser_0_o1, laser_1_o1, ..., laser_N_o1 (laser outputs)
    - Electrical: laser_0_top_e1, laser_0_top_e2, ... (current injection contacts)

    Args:
        num_emitters: Number of laser emitters (typically 8-50)
        emitter_pitch: Center-to-center spacing in µm (default: 50 µm)
        emitter_length: Length of each emitter in µm (default: 300 µm)
        add_labels: Whether to add text labels
        add_boundary: Whether to add chip boundary (FLOORPLAN layer)
        boundary_margin: Margin around components for boundary in µm

    Returns:
        Component representing a complete laser bar with exposed ports
    """
    c = gf.Component()

    # Use detector to mock laser emitter (similar geometry)
    emitter = gf.components.ge_detector_straight_si_contacts(length=emitter_length)

    # Array the emitters with specified pitch and expose their ports
    for i in range(num_emitters):
        y_pos = i * emitter_pitch
        ref = c << emitter
        ref.move((0, y_pos))

        # Expose optical output port (o1 is the laser output)
        c.add_port(f"laser_{i}_o1", port=ref.ports["o1"])

        # Expose electrical ports for current injection
        for port_name in ["top_e1", "top_e2", "top_e3", "top_e4",
                          "bot_e1", "bot_e2", "bot_e3", "bot_e4"]:
            c.add_port(f"laser_{i}_{port_name}", port=ref.ports[port_name])

    # Add chip boundary on FLOORPLAN layer
    if add_boundary:
        # Calculate bounding box
        bbox = c.bbox()
        if bbox is not None:
            width = bbox.width() + 2 * boundary_margin
            height = bbox.height() + 2 * boundary_margin

            # Add boundary rectangle on FLOORPLAN layer
            boundary = c << gf.components.rectangle(
                size=(width, height),
                layer="FLOORPLAN"
            )
            boundary.move((bbox.left - boundary_margin, bbox.bottom - boundary_margin))

    # Add label
    if add_labels:
        label = c << gf.components.text(
            text=f"Laser Bar ({num_emitters}ch)",
            size=20,
            layer=(1, 0)
        )
        label.move((0, (num_emitters - 1) * emitter_pitch + 50))

    return c


@gf.cell
def detector(
    length: float = 150.0,
) -> gf.Component:
    """
    Single photodetector component (wrapper around ge_detector_straight_si_contacts).

    This is 2x shorter than the default laser length (300 µm vs 150 µm).

    Ports:
    - o1: Optical input port
    - top_e1, top_e2, top_e3, top_e4: Top electrical contacts for photocurrent readout
    - bot_e1, bot_e2, bot_e3, bot_e4: Bottom electrical contacts for photocurrent readout

    Args:
        length: Length of the detector in µm (default: 150 µm, 2x shorter than laser)

    Returns:
        Component representing a single photodetector
    """
    return gf.components.ge_detector_straight_si_contacts(length=length)


if __name__ == "__main__":
    # Create and display laser bar
    num_emitters = 8
    emitter_pitch = 50.0
    emitter_length = 300.0

    c_laser = laser_bar(
        num_emitters=num_emitters,
        emitter_pitch=emitter_pitch,
        emitter_length=emitter_length,
        add_boundary=True,
        boundary_margin=50.0
    )
    c_laser.show()
    c_laser.write_gds("laser_bar_8ch.gds")

    print("=" * 80)
    print("Laser Bar Module Created (using detector to mock lasers)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of emitters: {num_emitters}")
    print(f"  - Emitter pitch: {emitter_pitch} µm")
    print(f"  - Emitter length: {emitter_length} µm")
    print(f"  - Total bar height: {(num_emitters - 1) * emitter_pitch} µm")
    print(f"  - Chip boundary: FLOORPLAN layer with 50 µm margin")

    print(f"\nModule Outputs:")
    print(f"  - Optical ports: {num_emitters} laser outputs (laser_0_o1 ... laser_{num_emitters-1}_o1)")
    print(f"  - Electrical ports: {num_emitters * 8} current injection contacts")
    print(f"    (laser_0_top_e1-4, laser_0_bot_e1-4, ... for each laser)")

    print(f"\nPort Summary:")
    port_names = [str(p) for p in c_laser.ports]
    print(f"  Total ports: {len(port_names)}")
    print(f"  Optical: {len([p for p in port_names if 'o1' in p])}")
    print(f"  Electrical: {len([p for p in port_names if '_e' in p])}")

    print(f"\nSample ports:")
    for i, port in enumerate(list(c_laser.ports)[:5]):
        print(f"  {port}")
    print(f"  ...")

    print(f"\nGDS file: laser_bar_8ch.gds")
    print("=" * 80)

    # Create and display single detector
    print("\n")
    detector_length = 150.0  # 2x shorter than lasers

    c_detector = detector(length=detector_length)
    c_detector.show()
    c_detector.write_gds("detector_single.gds")

    print("=" * 80)
    print("Detector Component Created")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Detector length: {detector_length} µm (2x shorter than laser)")

    print(f"\nComponent Ports:")
    port_names_det = [str(p) for p in c_detector.ports]
    print(f"  Total ports: {len(port_names_det)}")
    print(f"  Optical: {len([p for p in port_names_det if 'o1' in p])}")
    print(f"  Electrical: {len([p for p in port_names_det if '_e' in p])}")

    print(f"\nAll ports:")
    for port in c_detector.ports:
        print(f"  {port}")

    print(f"\nGDS file: detector_single.gds")
    print("=" * 80)

