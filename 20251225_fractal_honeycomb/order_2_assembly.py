"""
Order 2 Assembly: Create a single large hexagonal cell from Order 1 slabs

This script assembles a higher-order honeycomb cell where each of the 6 walls
is an Order 1 honeycomb slab, rotated 90° from its original orientation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from honeycomb_3d import (
    create_honeycomb_slab,
    plot_cell,
    setup_3d_axis,
    CELL_RADIUS,
    CELL_DEPTH,
    SLAB_ROWS,
    SLAB_COLS,
    FIGURE_SIZE,
    VIEW_ELEVATION,
    VIEW_AZIMUTH
)

# ============================================================================
# ORDER 2 PARAMETERS
# ============================================================================

# The Order 1 slab dimensions (approximate)
# Width (12 cols): ~12 * sqrt(3) * CELL_RADIUS
# Height (18 rows): ~18 * 1.5 * CELL_RADIUS  
# Depth: CELL_DEPTH

SLAB_WIDTH = 1.5*SLAB_COLS * np.sqrt(3) * CELL_RADIUS   # ~20.8
SLAB_HEIGHT = 1.5*SLAB_ROWS * 1.5 * CELL_RADIUS         # ~27.0
SLAB_THICKNESS = CELL_DEPTH                          # 3.0

# Order 2 hexagonal cell dimensions
# The slab height (27) becomes the wall thickness (depth along Z)
# The slab width (20.8) becomes the width of each hexagonal face
# For a pointy-top hexagon, if face width = w, then radius (center to vertex) = w / sqrt(3)
ORDER2_CELL_HEIGHT = SLAB_WIDTH      # Height of the Order 2 cell (along Z)
ORDER2_WALL_THICKNESS = SLAB_HEIGHT  # Thickness of each wall (radial depth)
ORDER2_HEX_RADIUS = SLAB_WIDTH / np.sqrt(3)  # Radius of hexagonal cross-section (center to vertex)

print(f"Order 1 Slab dimensions: {SLAB_WIDTH:.2f} × {SLAB_HEIGHT:.2f} × {SLAB_THICKNESS:.2f}")
print(f"Order 2 Cell - Height: {ORDER2_CELL_HEIGHT:.2f}, Wall thickness: {ORDER2_WALL_THICKNESS:.2f}, Hex radius: {ORDER2_HEX_RADIUS:.2f}")

# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def rotate_vertices(vertices, axis, angle):
    """
    Rotate vertices around a given axis by angle (in radians).
    
    Args:
        vertices: Array of shape (N, 3)
        axis: 'x', 'y', or 'z'
        angle: Rotation angle in radians
    
    Returns:
        Rotated vertices array
    """
    c, s = np.cos(angle), np.sin(angle)
    
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
    elif axis == 'y':
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
    elif axis == 'z':
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    return vertices @ R.T


def translate_vertices(vertices, offset):
    """
    Translate vertices by offset vector.
    
    Args:
        vertices: Array of shape (N, 3)
        offset: (dx, dy, dz) translation vector
    
    Returns:
        Translated vertices array
    """
    return vertices + np.array(offset)


# ============================================================================
# ORDER 2 ASSEMBLY
# ============================================================================

def create_order2_single_cell():
    """
    Create a single Order 2 hexagonal cell with walls made from Order 1 slabs.
    
    The Order 1 slab (originally oriented with tubes along Z) is rotated 90°
    so that the tubes run parallel to the Order 2 cell's axis.
    
    Returns:
        vertices: Combined vertex array
        edges: Combined edge list
        faces: Combined face list
    """
    # Create the base Order 1 slab
    slab_verts, slab_edges, slab_faces = create_honeycomb_slab(
        rows=SLAB_ROWS,
        cols=SLAB_COLS,
        radius=CELL_RADIUS,
        depth=CELL_DEPTH,
        center=(0, 0, 0)
    )
    
    print(f"Base slab: {len(slab_verts)} vertices, {len(slab_edges)} edges, {len(slab_faces)} faces")
    
    # The 6 hexagonal wall positions (pointy-top hexagon in XY plane)
    # Each wall is positioned at one of the 6 faces of the hexagon
    hex_angles = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/2  # Pointy-top orientation
    
    all_vertices = []
    all_edges = []
    all_faces = []
    vertex_offset = 0
    
    for i, angle in enumerate(hex_angles):
        print(f"\nProcessing wall {i+1}/6 at angle {np.degrees(angle):.1f}°")

        # Start with the base slab
        wall_verts = slab_verts.copy()

        # Step 1: Rotate 90° around X axis (tubes now point along Y instead of Z)
        # After this: slab spans XY plane with tubes along Y
        # The long edge (18 rows, ~27 units) is now along Y
        # The short edge (12 cols, ~20.8 units) is along X
        wall_verts = rotate_vertices(wall_verts, 'x', np.pi/2)

        # Step 2: Rotate around Z axis to align with hexagonal face angle
        wall_verts = rotate_vertices(wall_verts, 'z', angle)

        # Step 3: Rotate 90° around Z to orient the slab correctly in the plane
        wall_verts = rotate_vertices(wall_verts, 'z', np.pi/2)

        # Step 4: Translate outward to hexagon face position
        # The wall should be positioned so its inner surface is at the hexagon face
        # Distance from center to face = radius * cos(30°) = radius * sqrt(3)/2
        face_distance = ORDER2_HEX_RADIUS * np.sqrt(3) / 2
        offset_x = face_distance * np.cos(angle)
        offset_y = face_distance * np.sin(angle)
        wall_verts = translate_vertices(wall_verts, (offset_x, offset_y, 0))
        
        # Offset edge and face indices
        wall_edges = [(i + vertex_offset, j + vertex_offset) for i, j in slab_edges]
        wall_faces = [[i + vertex_offset for i in face] for face in slab_faces]
        
        all_vertices.append(wall_verts)
        all_edges.extend(wall_edges)
        all_faces.extend(wall_faces)
        vertex_offset += len(wall_verts)
    
    # Combine all vertices
    vertices = np.vstack(all_vertices)
    
    print(f"\nOrder 2 cell: {len(vertices)} total vertices, {len(all_edges)} edges, {len(all_faces)} faces")
    
    return vertices, all_edges, all_faces


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Creating Order 2 single hexagonal cell...\n")
    
    vertices, edges, faces = create_order2_single_cell()
    
    # Visualize
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    
    plot_cell(vertices, edges, faces, ax=ax)
    setup_3d_axis(ax, vertices)
    
    ax.set_title('Order 2: Single Hexagonal Cell (walls = Order 1 slabs)', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('order_2_single_cell.png', dpi=150, bbox_inches='tight')
    print("\nSaved: order_2_single_cell.png")
    
    plt.show()

