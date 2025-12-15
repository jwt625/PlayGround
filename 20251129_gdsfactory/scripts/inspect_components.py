"""
Inspect component dimensions for transceiver chip layout planning.
This script creates all necessary components and measures their bounding boxes.
"""

import gdsfactory as gf
from laser_bar_module import laser_bar, detector
from ring_modulator_pin import ring_single_pin
from ring_double_pin import ring_double_pin


def inspect_component(component, name):
    """Print component dimensions and port information."""
    bbox = component.bbox()
    if bbox is not None:
        width = bbox.width()
        height = bbox.height()
        print(f"\n{name}:")
        print(f"  Size: {width:.2f} × {height:.2f} µm")
        print(f"  BBox: ({bbox.left:.2f}, {bbox.bottom:.2f}) to ({bbox.right:.2f}, {bbox.top:.2f})")
        print(f"  Ports: {len(component.ports)}")
        
        # Show optical ports
        optical_ports = [p for p in component.ports if 'o' in p.name or 'laser' in p.name]
        if optical_ports:
            print(f"  Optical ports: {len(optical_ports)}")
            for p in optical_ports[:3]:  # Show first 3
                print(f"    - {p.name}: ({p.x:.2f}, {p.y:.2f}), orientation: {p.orientation}")
            if len(optical_ports) > 3:
                print(f"    ... and {len(optical_ports) - 3} more")
    else:
        print(f"\n{name}: No bounding box")


print("=" * 80)
print("COMPONENT DIMENSION INSPECTION")
print("=" * 80)

# 1. Laser bar (8 emitters)
print("\n" + "-" * 80)
print("1. LASER DIE")
print("-" * 80)
c_laser = laser_bar(num_emitters=8, emitter_pitch=50.0, emitter_length=300.0, add_boundary=False)
inspect_component(c_laser, "Laser Bar (8 emitters)")

# 2. Single detector
print("\n" + "-" * 80)
print("2. SINGLE DETECTOR")
print("-" * 80)
c_detector = detector(length=150.0)
inspect_component(c_detector, "Single Detector")

# 3. Detector array (8 detectors for monitoring)
print("\n" + "-" * 80)
print("3. DETECTOR ARRAY (8 detectors)")
print("-" * 80)
c_det_array = gf.Component()
detector_pitch = 50.0
for i in range(8):
    det = c_det_array << detector(length=150.0)
    det.move((0, i * detector_pitch))
inspect_component(c_det_array, "Detector Array (8 detectors, monitoring)")

# 4. Detector chip (8 detectors for receive)
print("\n" + "-" * 80)
print("4. DETECTOR CHIP (8 detectors for RX)")
print("-" * 80)
c_det_chip = gf.Component()
for i in range(8):
    det = c_det_chip << detector(length=150.0)
    det.move((0, i * detector_pitch))
inspect_component(c_det_chip, "Detector Chip (8 detectors, receive)")

# 5. Ring resonator with PIN (single bus - old design)
print("\n" + "-" * 80)
print("5. RING RESONATOR WITH PIN (single bus)")
print("-" * 80)
try:
    c_ring = ring_single_pin(gap=0.2, radius=10.0, length_x=20.0, length_y=2.0)
    inspect_component(c_ring, "Ring with PIN (single)")
except Exception as e:
    print(f"Error creating ring: {e}")
    print("Trying with larger dimensions...")
    c_ring = ring_single_pin(gap=0.2, radius=10.0, length_x=50.0, length_y=2.0, via_stack_width=10.0)
    inspect_component(c_ring, "Ring with PIN (single, larger)")

# 5b. Add-drop ring resonator with PIN (double bus - new design)
print("\n" + "-" * 80)
print("5b. ADD-DROP RING RESONATOR WITH PIN (double bus)")
print("-" * 80)
c_ring_double = ring_double_pin(gap=0.2, radius=10.0, length_x=20.0, length_y=50.0, via_stack_width=10.0)
inspect_component(c_ring_double, "Add-Drop Ring with PIN")

# 6. PIN modulator
print("\n" + "-" * 80)
print("6. PIN MODULATOR")
print("-" * 80)
c_pin_mod = gf.components.straight_pin(length=100.0)
inspect_component(c_pin_mod, "PIN Modulator")

# 7. 1x4 Splitter
print("\n" + "-" * 80)
print("7. POWER SPLITTER (1×4)")
print("-" * 80)
c_splitter = gf.components.splitter_tree(noutputs=4, spacing=(90, 50))
inspect_component(c_splitter, "1×4 Splitter Tree")

# 8. AWG (8 channels)
print("\n" + "-" * 80)
print("8. AWG (8 CHANNELS)")
print("-" * 80)
c_awg = gf.components.awg(arms=20, outputs=8)
inspect_component(c_awg, "AWG (8 channels)")

# 9. Edge coupler
print("\n" + "-" * 80)
print("9. EDGE COUPLER")
print("-" * 80)
c_edge = gf.components.edge_coupler_silicon()
inspect_component(c_edge, "Edge Coupler")

# 10. MMI 1x2 (alternative to splitter tree)
print("\n" + "-" * 80)
print("10. MMI 1×2 (for comparison)")
print("-" * 80)
c_mmi = gf.components.mmi1x2()
inspect_component(c_mmi, "MMI 1×2")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

