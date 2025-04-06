for (i = [0:5]) {
    translate([i * 335, 0, 0]) {
difference() {
    // Positive shape: extruded by 120 units
    linear_extrude(height = 130)
        import("box_pos.dxf", convexity=10);
    
    // Negative shape: extruded by 100 units and moved down by 1 unit
    translate([0, 0, 31])
    linear_extrude(height = 100)
        import("box_neg.dxf", convexity=10);
}
}
}
