"""
Order 2 Slab Assembly: Create a slab from Order 1 slabs as walls

This script assembles a 2nd order honeycomb slab by positioning Order 1 slabs
as walls in a hexagonal grid pattern, similar to how Order 1 uses Order 0 cells.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from honeycomb_3d import (
    hex_grid_positions,
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
from order_2_assembly import (
    rotate_vertices,
    translate_vertices,
    SLAB_WIDTH,
    SLAB_HEIGHT,
    SLAB_THICKNESS,
    ORDER2_HEX_RADIUS
)

# ============================================================================
# ORDER 2 SLAB PARAMETERS
# ============================================================================

# Grid size for Order 2 slab
ORDER2_SLAB_ROWS = 3
ORDER2_SLAB_COLS = 3

print(f"Order 2 cell hex radius: {ORDER2_HEX_RADIUS:.2f}")

# ============================================================================
# ORDER 2 CELL - SAME AS ORDER 0 BUT LARGER SCALE
# ============================================================================

def create_order2_hexagonal_cell(radius=1.0, depth=3.0, center=(0, 0, 0)):
    """
    Create a single Order 2 hexagonal cell (tube segment).
    EXACT SAME CODE as create_hexagonal_cell from honeycomb_3d.py

    The cell is oriented with:
    - Hexagonal cross-section in XY plane
    - Tube axis along Z direction
    - Bottom face at z=center[2], top face at z=center[2]+depth

    Args:
        radius: Hexagon radius (center to vertex)
        depth: Height of the tube
        center: (x, y, z) center position of bottom face

    Returns:
        vertices: Array of shape (12, 3) - 6 bottom + 6 top vertices
        edges: List of (i, j) tuples defining edges to draw
        faces: List of vertex index lists defining faces
    """
    cx, cy, cz = center

    # Bottom hexagon vertices
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 vertices
    angles += np.pi/2  # Rotate to pointy-top

    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)

    bottom_verts = np.column_stack([x, y, np.full(6, cz)])

    # Top hexagon vertices
    top_verts = np.column_stack([x, y, np.full(6, cz + depth)])

    # Combine all vertices
    vertices = np.vstack([bottom_verts, top_verts])

    # Define edges
    edges = []

    # Bottom hexagon edges
    for i in range(6):
        edges.append((i, (i + 1) % 6))

    # Top hexagon edges
    for i in range(6):
        edges.append((i + 6, ((i + 1) % 6) + 6))

    # Vertical edges connecting bottom to top
    for i in range(6):
        edges.append((i, i + 6))

    # Define faces for solid rendering
    faces = []

    # Only side faces (6 rectangular faces) - no top/bottom for hollow appearance
    for i in range(6):
        next_i = (i + 1) % 6
        # Each side is a quad: bottom[i], bottom[next_i], top[next_i], top[i]
        faces.append([i, next_i, next_i + 6, i + 6])

    return vertices, edges, faces


# ============================================================================
# ORDER 2 SLAB ASSEMBLY
# ============================================================================

def create_order2_slab_wireframe(rows=3, cols=3, radius=1.0, depth=3.0, center=(0, 0, 0)):
    """
    Create Order 2 slab wireframe using EXACT SAME approach as create_honeycomb_slab.

    Args:
        rows: Number of rows in hex grid
        cols: Number of columns in hex grid
        radius: Cell radius
        depth: Cell depth
        center: (x, y, z) center offset for the entire slab

    Returns:
        vertices: Combined vertex array
        edges: Edge list
        faces: List of faces for solid rendering
    """
    # Get grid positions
    positions = hex_grid_positions(rows, cols, radius)

    # Center the grid around origin
    positions = np.array(positions)
    positions[:, 0] -= positions[:, 0].mean()
    positions[:, 1] -= positions[:, 1].mean()

    all_vertices = []
    all_edges = []
    all_faces = []
    vertex_offset = 0

    # Create each cell
    for x, y in positions:
        cell_center = (x + center[0], y + center[1], center[2])
        verts, edges, faces = create_order2_hexagonal_cell(radius, depth, cell_center)

        # Offset edge indices
        edges_offset = [(i + vertex_offset, j + vertex_offset) for i, j in edges]

        # Offset face indices
        faces_offset = [[i + vertex_offset for i in face] for face in faces]

        all_vertices.append(verts)
        all_edges.extend(edges_offset)
        all_faces.extend(faces_offset)
        vertex_offset += len(verts)

    # Combine all vertices
    vertices = np.vstack(all_vertices)

    return vertices, all_edges, all_faces


def add_single_wall_slab(wireframe_verts, wireframe_edges,
                         cell_index=0, wall_index=0,
                         rows=3, cols=3, radius=1.0):
    """
    Add a single Order 1 slab to one wall of one cell in the wireframe.

    Args:
        wireframe_verts: Wireframe vertices
        wireframe_edges: Wireframe edges
        cell_index: Which cell to add the wall to (0 to rows*cols-1)
        wall_index: Which wall of the hexagon (0-5)
        rows, cols, radius: Grid parameters

    Returns:
        vertices: Combined vertices (wireframe + slab)
        edges: Combined edges
        faces: Slab faces
    """
    # Get grid positions to find the cell center
    positions = hex_grid_positions(rows, cols, radius)
    positions = np.array(positions)
    positions[:, 0] -= positions[:, 0].mean()
    positions[:, 1] -= positions[:, 1].mean()

    cell_x, cell_y = positions[cell_index]

    # Create the Order 1 slab
    slab_verts, slab_edges, slab_faces = create_honeycomb_slab(
        rows=SLAB_ROWS,
        cols=SLAB_COLS,
        radius=CELL_RADIUS,
        depth=CELL_DEPTH,
        center=(0, 0, 0)
    )

    # Calculate the angle for this wall's center
    # Vertex angles for pointy-top hexagon
    vertex_angles = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/2
    # Face centers are offset by 30° (pi/6) from vertices
    face_angles = vertex_angles + np.pi/6
    angle = face_angles[wall_index]

    # Transformation sequence:
    # Step 1: Rotate 90° around X - stands slab up, surface normal faces +Y
    slab_verts = rotate_vertices(slab_verts, 'x', np.pi/2)

    # Step 2: Rotate around Z by wall angle - orients normal toward wall direction
    slab_verts = rotate_vertices(slab_verts, 'z', angle)

    # Step 3: Rotate 90° around Z - final orientation adjustment
    slab_verts = rotate_vertices(slab_verts, 'z', np.pi/2)

    # Step 4: Translate to wall position
    face_distance = radius * np.sqrt(3) / 2
    offset_x = cell_x + face_distance * np.cos(angle)
    offset_y = cell_y + face_distance * np.sin(angle)
    slab_verts = translate_vertices(slab_verts, (offset_x, offset_y, 0))

    # Combine with wireframe
    vertex_offset = len(wireframe_verts)
    combined_verts = np.vstack([wireframe_verts, slab_verts])

    # Offset slab edges and add to wireframe edges
    slab_edges_offset = [(i + vertex_offset, j + vertex_offset) for i, j in slab_edges]
    combined_edges = wireframe_edges + slab_edges_offset

    # Offset slab faces
    slab_faces_offset = [[i + vertex_offset for i in face] for face in slab_faces]

    return combined_verts, combined_edges, slab_faces_offset


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Creating Order 2 honeycomb slab wireframe with single wall...\n")

    # First, draw the hexagonal wireframe using SAME approach as honeycomb_3d
    print(f"Drawing {ORDER2_SLAB_ROWS}x{ORDER2_SLAB_COLS} Order 2 hexagonal cells...")
    wireframe_verts, wireframe_edges, wireframe_faces = create_order2_slab_wireframe(
        rows=ORDER2_SLAB_ROWS,
        cols=ORDER2_SLAB_COLS,
        radius=ORDER2_HEX_RADIUS,
        depth=SLAB_WIDTH,  # Height along Z should be the slab width
        center=(0, 0, 0)
    )

    print(f"Wireframe: {len(wireframe_verts)} vertices, {len(wireframe_edges)} edges, {len(wireframe_faces)} faces")

    # Add an UNTRANSFORMED Order 1 slab for comparison (90° Y rotation only)
    print("\nAdding Order 1 slab with 90° Y rotation for reference...")
    untransformed_slab_verts, untransformed_slab_edges, untransformed_slab_faces = create_honeycomb_slab(
        rows=SLAB_ROWS,
        cols=SLAB_COLS,
        radius=CELL_RADIUS,
        depth=CELL_DEPTH,
        center=(0, 0, 0)
    )
    # Apply 90° rotation around Y axis
    untransformed_slab_verts = rotate_vertices(untransformed_slab_verts, 'x', np.pi/2)

    # Add untransformed slab to wireframe
    vertex_offset_1 = len(wireframe_verts)
    temp_verts = np.vstack([wireframe_verts, untransformed_slab_verts])
    untransformed_edges_offset = [(i + vertex_offset_1, j + vertex_offset_1) for i, j in untransformed_slab_edges]
    temp_edges = wireframe_edges + untransformed_edges_offset
    untransformed_faces_offset = [[i + vertex_offset_1 for i in face] for face in untransformed_slab_faces]

    # Add a single TRANSFORMED Order 1 slab to one wall
    print("Adding TRANSFORMED Order 1 slab to cell 0, wall 0...")
    vertices, edges, faces = add_single_wall_slab(
        temp_verts, temp_edges,
        cell_index=0,
        wall_index=0,
        rows=ORDER2_SLAB_ROWS,
        cols=ORDER2_SLAB_COLS,
        radius=ORDER2_HEX_RADIUS
    )

    # Combine all faces
    all_faces = wireframe_faces + untransformed_faces_offset + faces

    print(f"Total: {len(vertices)} vertices, {len(edges)} edges, {len(all_faces)} faces")

    # Visualize
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111, projection='3d')

    plot_cell(vertices, edges, all_faces, ax=ax)
    setup_3d_axis(ax, vertices)

    # Add axis markers for debugging
    axis_length = ORDER2_HEX_RADIUS * 2
    ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', linewidth=3, label='X axis')
    ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=3, label='Y axis')
    ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', linewidth=3, label='Z axis')
    ax.legend()

    # Show debug markers for sidewall centers
    SHOW_DEBUG_MARKERS = True
    if SHOW_DEBUG_MARKERS:
        # Get cell positions
        positions = hex_grid_positions(ORDER2_SLAB_ROWS, ORDER2_SLAB_COLS, ORDER2_HEX_RADIUS)
        positions = np.array(positions)
        positions[:, 0] -= positions[:, 0].mean()
        positions[:, 1] -= positions[:, 1].mean()

        # Plot sidewall centers for all cells
        # Vertex angles (pointy-top hexagon)
        vertex_angles = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/2
        # Face centers are offset by 30° (pi/6) from vertices
        face_angles = vertex_angles + np.pi/6
        face_distance = ORDER2_HEX_RADIUS * np.sqrt(3) / 2

        for cell_idx, (cell_x, cell_y) in enumerate(positions):
            for wall_idx in range(6):
                angle = face_angles[wall_idx]
                wall_x = cell_x + face_distance * np.cos(angle)
                wall_y = cell_y + face_distance * np.sin(angle)
                ax.scatter([wall_x], [wall_y], [0], color='red', s=50, zorder=10)
                ax.text(wall_x, wall_y, 2, f'C{cell_idx}W{wall_idx}',
                       fontsize=8, color='red', weight='bold')

    ax.set_title(f'Order 2 Slab ({ORDER2_SLAB_ROWS}x{ORDER2_SLAB_COLS})\nUntransformed (offset) + Transformed Wall',
                 fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('order_2_slab_wireframe.png', dpi=150, bbox_inches='tight')
    print("\nSaved: order_2_slab_wireframe.png")

    plt.show()

