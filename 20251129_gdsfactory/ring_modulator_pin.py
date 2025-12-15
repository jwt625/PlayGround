"""
Ring Modulator with Integrated PIN Junction

This module creates a custom ring resonator with an integrated PIN modulator segment.
The ring structure is based on gdsfactory's ring_single, but replaces the top straight
section with a PIN junction for active modulation.

Structure:
- Bottom: Directional coupler (for coupling light in/out)
- Left/Right: Vertical straight waveguides
- Top: PIN modulator segment (replaces regular straight)
- Bends: Euler bends connecting the sections

The PIN segment provides:
- Optical ports: o1, o2 (integrated into ring)
- Electrical ports: top_e1-e4, bot_e1-e4 (for voltage control)
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_single_pin(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 2.0,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    coupler_ring: ComponentSpec = "coupler_ring",
    cross_section: CrossSectionSpec = "strip",
    cross_section_pin: CrossSectionSpec = "pin",
    via_stack: ComponentSpec = "via_stack_slab_m3",
    via_stack_width: float = 3.0,
    via_stack_spacing: float = 3.0,
    length_extension: float | None = None,
) -> Component:
    """
    Returns a ring resonator with integrated PIN modulator segment.
    
    The ring consists of:
    - A directional coupler (cb) at the bottom for optical coupling
    - Two vertical straights (sl, sr) on the left and right sides
    - Two bends (bl, br) connecting the vertical straights
    - A PIN modulator segment (st_pin) at the top for active modulation
    
    Args:
        gap: Gap between ring and bus waveguide in coupler (μm)
        radius: Radius of the ring bends (μm)
        length_x: Length of both the coupler section AND the PIN segment (μm)
        length_y: Length of the vertical straight sections (μm)
        bend: Component spec for 90-degree bends
        straight: Component spec for straight waveguides
        coupler_ring: Component spec for the ring coupler
        cross_section: Cross section for passive waveguides
        cross_section_pin: Cross section for PIN modulator
        via_stack: Via stack component for electrical contacts
        via_stack_width: Width of via stacks (μm)
        via_stack_spacing: Spacing between via stacks (μm)
        length_extension: Straight length extension at coupler ports
        
    Returns:
        Component with:
        - Optical ports: o1 (input), o2 (through/output)
        - Electrical ports: top_e1-e4, bot_e1-e4 (PIN modulator control)
    
    .. code::
    
                 ═══PIN MODULATOR═══
              xxx                   xxx
            xxx                       xxx
           xx                           xxx
           x                             xxx
          xx                              xx▲
          xx                              xx│length_y
          xx                              xx▼
          xx                             xx
           xx          length_x          x
            xx     ◄───────────────►    x
             xx                       xxx
               xx                   xxx
                xxx──────▲─────────xxx
                         │gap
                 o1──────▼─────────o2
    """
    c = Component()
    
    # Create the bottom coupler
    settings = dict(
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        bend=bend,
        straight=straight,
    )
    
    if length_extension is not None:
        settings["length_extension"] = length_extension
    
    cb = c << gf.get_component(coupler_ring, settings=settings)
    
    # Create passive waveguide components
    sy = gf.get_component(straight, length=length_y, cross_section=cross_section)
    b = gf.get_component(bend, cross_section=cross_section, radius=radius)
    
    # Create PIN modulator segment - same length as coupler length_x
    st_pin = c << gf.components.straight_pin(
        length=length_x,
        cross_section=cross_section_pin,
        via_stack=via_stack,
        via_stack_width=via_stack_width,
        via_stack_spacing=via_stack_spacing,
        taper="taper_strip_to_ridge",
    )
    
    # Place passive components
    sl = c << sy  # Left vertical straight
    sr = c << sy  # Right vertical straight
    bl = c << b   # Left bend
    br = c << b   # Right bend
    
    # Connect all components to form the ring
    sl.connect(port="o1", other=cb.ports["o2"])
    bl.connect(port="o2", other=sl.ports["o2"])
    st_pin.connect(port="o2", other=bl.ports["o1"])  # PIN segment at top
    br.connect(port="o2", other=st_pin.ports["o1"])
    sr.connect(port="o1", other=br.ports["o1"])
    sr.connect(port="o2", other=cb.ports["o3"])
    
    # Add optical ports from the coupler
    c.add_port("o1", port=cb.ports["o1"])
    c.add_port("o2", port=cb.ports["o4"])

    # Add electrical ports from the PIN modulator
    for port in st_pin.ports:
        if port.port_type == "electrical":
            c.add_port(f"pin_{port.name}", port=port)
    
    return c


if __name__ == "__main__":
    # Create and display the ring modulator
    try:
        c = ring_single_pin(
            gap=0.2,
            radius=10.0,
            length_x=20.0,  # Needs to be long enough to accommodate tapers
            length_y=2.0,
        )

        c.show()
        c.write_gds("ring_modulator_pin.gds")
    except Exception as e:
        print(f"Error creating ring: {e}")
        print("\nTrying with larger via_stack_width...")
        c = ring_single_pin(
            gap=0.2,
            radius=10.0,
            length_x=50.0,
            length_y=2.0,
            via_stack_width=10.0,
        )
        c.show()
        c.write_gds("ring_modulator_pin.gds")
    
    print("=" * 80)
    print("Ring Modulator with PIN Junction Created")
    print("=" * 80)
    print(f"\nOptical Ports:")
    for port_name in ["o1", "o2"]:
        if port_name in c.ports:
            print(f"  - {port_name}: {c.ports[port_name]}")
    
    print(f"\nElectrical Ports (PIN Modulator):")
    for port in c.ports:
        if "pin_" in port.name:
            print(f"  - {port.name}: {port}")
    
    print(f"\nComponent Info:")
    print(f"  - Total ports: {len(c.ports)}")
    print(f"  - Bounding box: {c.bbox()}")

    print(f"\nGDS file written to: ring_modulator_pin.gds")
    print(f"Use KLayout or gdsfactory viewer to visualize the layout.")

