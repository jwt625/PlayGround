// Import a 2D DXF file
//import("output.dxf", convexity=10);

// Optionally, extrude the imported shape into a 3D object
linear_extrude(height = 100)
    import("output.dxf", convexity=10);
