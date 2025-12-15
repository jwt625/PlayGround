"""
Test different AWG parameters to fix waveguide overlapping
"""
import gdsfactory as gf
from functools import partial

c = gf.Component("AWG_Parameter_Tests")

y_offset = 0
spacing = 200

# Test 1: Default AWG
print("=" * 80)
print("AWG PARAMETER TESTS")
print("=" * 80)

awg1 = c << gf.components.awg(arms=20, outputs=8)
awg1.move((0, y_offset))
label1 = c << gf.components.text("1: Default", size=8, layer=(1, 0))
label1.move((-50, y_offset - 20))
print(f"\n1. Default AWG (arms=20, outputs=8):")
print(f"   Size: {awg1.bbox().width():.1f} x {awg1.bbox().height():.1f} µm")

y_offset += spacing

# Test 2: Increased arm_spacing
awg2 = c << gf.components.awg(arms=20, outputs=8, arm_spacing=5.0)
awg2.move((0, y_offset))
label2 = c << gf.components.text("2: arm_spacing=5", size=8, layer=(1, 0))
label2.move((-50, y_offset - 20))
print(f"\n2. AWG with arm_spacing=5.0 (default=1.0):")
print(f"   Size: {awg2.bbox().width():.1f} x {awg2.bbox().height():.1f} µm")

y_offset += spacing

# Test 3: Increased fpr_spacing
awg3 = c << gf.components.awg(arms=20, outputs=8, fpr_spacing=150.0)
awg3.move((0, y_offset))
label3 = c << gf.components.text("3: fpr_spacing=150", size=8, layer=(1, 0))
label3.move((-50, y_offset - 20))
print(f"\n3. AWG with fpr_spacing=150.0 (default=50.0):")
print(f"   Size: {awg3.bbox().width():.1f} x {awg3.bbox().height():.1f} µm")

y_offset += spacing

# Test 4: More arms
awg4 = c << gf.components.awg(arms=40, outputs=8)
awg4.move((0, y_offset))
label4 = c << gf.components.text("4: arms=40", size=8, layer=(1, 0))
label4.move((-50, y_offset - 20))
print(f"\n4. AWG with arms=40 (default=20):")
print(f"   Size: {awg4.bbox().width():.1f} x {awg4.bbox().height():.1f} µm")

y_offset += spacing

# Test 5: Combination - increased arm_spacing + fpr_spacing
awg5 = c << gf.components.awg(arms=20, outputs=8, arm_spacing=5.0, fpr_spacing=150.0)
awg5.move((0, y_offset))
label5 = c << gf.components.text("5: arm_sp=5 + fpr_sp=150", size=8, layer=(1, 0))
label5.move((-50, y_offset - 20))
print(f"\n5. AWG with arm_spacing=5.0 + fpr_spacing=150.0:")
print(f"   Size: {awg5.bbox().width():.1f} x {awg5.bbox().height():.1f} µm")

y_offset += spacing

# Test 6: Even larger arm_spacing
awg6 = c << gf.components.awg(arms=20, outputs=8, arm_spacing=10.0)
awg6.move((0, y_offset))
label6 = c << gf.components.text("6: arm_spacing=10", size=8, layer=(1, 0))
label6.move((-50, y_offset - 20))
print(f"\n6. AWG with arm_spacing=10.0:")
print(f"   Size: {awg6.bbox().width():.1f} x {awg6.bbox().height():.1f} µm")

y_offset += spacing

# Test 7: Very large arm_spacing
awg7 = c << gf.components.awg(arms=20, outputs=8, arm_spacing=20.0)
awg7.move((0, y_offset))
label7 = c << gf.components.text("7: arm_spacing=20", size=8, layer=(1, 0))
label7.move((-50, y_offset - 20))
print(f"\n7. AWG with arm_spacing=20.0:")
print(f"   Size: {awg7.bbox().width():.1f} x {awg7.bbox().height():.1f} µm")

c.write_gds("awg_comparison.gds")
print(f"\n" + "=" * 80)
print(f"GDS written to: awg_comparison.gds")
print("Open in viewer to see which configuration has no waveguide overlaps")
print("=" * 80)

c.show()

