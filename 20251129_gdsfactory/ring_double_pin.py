"""
Add-Drop Ring Resonator with Integrated PIN Junction

This module creates an add-drop ring resonator with an integrated PIN modulator.
Based on gdsfactory's ring_double, but replaces one of the vertical straight
sections with a PIN junction for active tuning.

Structure:
- Bottom coupler: Input bus waveguide (o1 → o2)
- Top coupler: Drop bus waveguide (o3 → o4)
- Left arm: PIN modulator segment (for wavelength tuning)
- Right arm: Passive straight waveguide

This is the correct configuration for laser multiplexing where:
- Each laser couples into the ring from the side
- The ring drops the resonant wavelength to the bus waveguide
- PIN modulator tunes the ring resonance to match laser wavelength
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_double_pin(
    gap: float = 0.2,
    gap_top: float | None = None,
    gap_bot: float | None = None,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 2.0,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    coupler_ring: ComponentSpec = "coupler_ring",
    coupler_ring_top: ComponentSpec | None = None,
    cross_section: CrossSectionSpec = "strip",
    cross_section_pin: CrossSectionSpec = "pin",
    via_stack: ComponentSpec = "via_stack_slab_m3",
    via_stack_width: float = 10.0,
    via_stack_spacing: float = 3.0,
    length_extension: float | None = None,
    pin_on_left: bool = True,
) -> Component:
    """
    Returns an add-drop ring resonator with integrated PIN modulator.
    
    The ring consists of:
    - Bottom coupler (cb): Input/through bus waveguide
    - Top coupler (ct): Drop bus waveguide
    - Left arm (sl): PIN modulator segment (default) or passive straight
    - Right arm (sr): Passive straight or PIN modulator segment
    
    Args:
        gap: Gap between ring and bus waveguides (μm)
        gap_top: Gap for top coupler (defaults to gap)
        gap_bot: Gap for bottom coupler (defaults to gap)
        radius: Radius of the ring bends (μm)
        length_x: Length of the coupler sections (μm)
        length_y: Length of the vertical straight sections (μm)
        bend: Component spec for 90-degree bends
        straight: Component spec for straight waveguides
        coupler_ring: Component spec for the ring coupler
        coupler_ring_top: Component spec for top coupler (defaults to coupler_ring)
        cross_section: Cross section for passive waveguides
        cross_section_pin: Cross section for PIN modulator
        via_stack: Via stack component for electrical contacts
        via_stack_width: Width of via stacks (μm)
        via_stack_spacing: Spacing between via stacks (μm)
        length_extension: Straight length extension at coupler ports
        pin_on_left: If True, PIN is on left arm; if False, PIN is on right arm
        
    Returns:
        Component with:
        - Optical ports: o1 (input), o2 (through), o3 (drop), o4 (add)
        - Electrical ports: pin_top_e1-e4, pin_bot_e1-e4 (PIN modulator control)
    
    .. code::
    
           o4──────▲─────────o3
                   │gap_top
           xx──────▼─────────xxx
          xxx                   xxx
        xxx                       xxx
       xx                           xxx
       x                             xxx
      xx══PIN══                      xx▲
      xx MODULATOR                   xx│length_y
      xx══════                       xx▼
      xx                             xx
       xx          length_x          x
        xx     ◄───────────────►    x
         xx                       xxx
           xx                   xxx
            xxx──────▲─────────xxx
                     │gap_bot
             o1──────▼─────────o2
    """
    c = Component()
    
    gap_top = gap_top or gap
    gap_bot = gap_bot or gap
    
    # Create bottom coupler
    coupler_component_bot = gf.get_component(
        coupler_ring,
        gap=gap_bot,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        straight=straight,
        bend=bend,
        length_extension=length_extension,
    )
    
    # Create top coupler
    coupler_component_top = gf.get_component(
        coupler_ring_top or coupler_ring,
        gap=gap_top,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        straight=straight,
        bend=bend,
        length_extension=length_extension,
    )
    
    # Create PIN modulator segment
    st_pin = c << gf.components.straight_pin(
        length=length_y,
        cross_section=cross_section_pin,
        via_stack=via_stack,
        via_stack_width=via_stack_width,
        via_stack_spacing=via_stack_spacing,
        taper="taper_strip_to_ridge",
    )
    
    # Create passive straight
    straight_component = gf.get_component(
        straight,
        length=length_y,
        cross_section=cross_section,
    )
    
    # Add couplers
    cb = c << coupler_component_bot
    ct = c << coupler_component_top
    
    # Add left and right arms based on pin_on_left parameter
    if pin_on_left:
        sl = st_pin  # Left arm is PIN
        sr = c << straight_component  # Right arm is passive
    else:
        sl = c << straight_component  # Left arm is passive
        sr = st_pin  # Right arm is PIN
    
    # Connect components
    sl.connect(port="o1", other=cb.ports["o2"])
    sr.connect(port="o2", other=cb.ports["o3"])
    ct.connect(port="o3", other=sl.ports["o2"])
    
    # Add optical ports
    c.add_port("o1", port=cb.ports["o1"])
    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o3", port=ct.ports["o4"])
    c.add_port("o4", port=ct.ports["o1"])
    
    # Add electrical ports from PIN modulator
    for port in st_pin.ports:
        if port.port_type == "electrical":
            c.add_port(f"pin_{port.name}", port=port)
    
    return c


if __name__ == "__main__":
    print("Creating Add-Drop Ring with PIN modulator...")
    
    c = ring_double_pin(
        gap=0.2,
        radius=10.0,
        length_x=20.0,
        length_y=50.0,  # Longer for PIN segment
        via_stack_width=10.0,
        pin_on_left=True,
    )
    
    c.show()
    c.write_gds("ring_double_pin.gds")
    
    print("=" * 80)
    print("Add-Drop Ring with PIN Junction Created")
    print("=" * 80)
    print(f"\nOptical Ports:")
    for port_name in ["o1", "o2", "o3", "o4"]:
        if port_name in c.ports:
            print(f"  - {port_name}: {c.ports[port_name]}")
    
    print(f"\nElectrical Ports (PIN Modulator):")
    for port in c.ports:
        if "pin_" in port.name:
            print(f"  - {port.name}: {port}")
    
    print(f"\nComponent Info:")
    print(f"  - Total ports: {len(c.ports)}")
    bbox = c.bbox()
    print(f"  - Size: {bbox.width():.1f} × {bbox.height():.1f} µm")
    
    print(f"\nGDS file: ring_double_pin.gds")
    print("=" * 80)

