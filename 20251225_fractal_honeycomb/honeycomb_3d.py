"""
Self-similar 3D honeycomb visualization
Order 0: Basic hexagonal cell module
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# ============================================================================
# PARAMETERS - Tune these
# ============================================================================

# Order 0 cell geometry
CELL_RADIUS = 1.0           # Radius of hexagon (center to vertex)
CELL_DEPTH = 3.0            # Height/depth of the hexagonal tube
WALL_THICKNESS = 0.05       # Thickness of cell walls (for visualization)

# Order 1 slab geometry
SLAB_ROWS = 18               # Number of rows in hexagonal grid
SLAB_COLS = 10               # Number of columns in hexagonal grid

# Visualization
FIGURE_SIZE = (10, 10)      # Figure size in inches
VIEW_ELEVATION = 20         # Camera elevation angle (degrees)
VIEW_AZIMUTH = 45           # Camera azimuth angle (degrees)
RENDER_MODE = 'solid'       # 'wireframe' or 'solid'
EDGE_COLOR = '#2a2a2a'      # Color of edges
EDGE_WIDTH = 0.8            # Width of edge lines
FACE_ALPHA = 0.8            # Transparency of faces (0=invisible, 1=opaque)
FACE_COLOR = '#d0d0d0'      # Color of faces
SHOW_EDGES = True           # Show edges in solid mode

# ============================================================================
# GEOMETRY FUNCTIONS
# ============================================================================

def hexagon_vertices_2d(radius=1.0, center=(0, 0)):
    """
    Generate vertices of a regular hexagon in 2D.
    Pointy-top orientation (vertex at top).
    
    Args:
        radius: Distance from center to vertex
        center: (x, y) center position
    
    Returns:
        Array of shape (6, 2) with vertex coordinates
    """
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 vertices, exclude duplicate
    angles += np.pi/2  # Rotate to pointy-top
    
    cx, cy = center
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    
    return np.column_stack([x, y])


def create_hexagonal_cell(radius=1.0, depth=3.0, center=(0, 0, 0)):
    """
    Create a single hexagonal cell (tube segment) - Order 0 module.

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
    hex_2d = hexagon_vertices_2d(radius, (cx, cy))
    bottom_verts = np.column_stack([hex_2d, np.full(6, cz)])

    # Top hexagon vertices
    top_verts = np.column_stack([hex_2d, np.full(6, cz + depth)])

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


def hex_grid_positions(rows=3, cols=3, radius=1.0):
    """
    Generate center positions for hexagonal grid (pointy-top).

    Args:
        rows: Number of rows
        cols: Number of columns
        radius: Hexagon radius

    Returns:
        List of (x, y) center positions
    """
    positions = []

    # Spacing between hex centers
    dx = np.sqrt(3) * radius  # Horizontal spacing
    dy = 1.5 * radius          # Vertical spacing

    for row in range(rows):
        for col in range(cols):
            # Offset every other row for hex packing
            x_offset = (dx / 2) if row % 2 == 1 else 0
            x = col * dx + x_offset
            y = row * dy
            positions.append((x, y))

    return positions


def deduplicate_edges(edges, vertices, tolerance=1e-9):
    """
    Remove duplicate edges based on vertex positions.
    Two edges are duplicates if they connect the same vertex positions.

    Args:
        edges: List of (i, j) tuples (vertex indices)
        vertices: Array of vertex positions
        tolerance: Distance tolerance for considering vertices identical

    Returns:
        List of deduplicated edges
    """
    unique_edges = set()

    for i, j in edges:
        # Get vertex positions
        v1 = tuple(vertices[i])
        v2 = tuple(vertices[j])

        # Normalize edge direction (smaller index first)
        edge = tuple(sorted([v1, v2]))
        unique_edges.add(edge)

    # Convert back to index pairs
    # Build vertex position to index mapping
    pos_to_idx = {}
    for idx, v in enumerate(vertices):
        pos = tuple(v)
        if pos not in pos_to_idx:
            pos_to_idx[pos] = idx

    deduped = []
    for v1, v2 in unique_edges:
        i = pos_to_idx[v1]
        j = pos_to_idx[v2]
        deduped.append((i, j))

    return deduped


def create_honeycomb_slab(rows=3, cols=3, radius=1.0, depth=3.0, center=(0, 0, 0)):
    """
    Create a honeycomb slab (Order 1) from an array of hexagonal cells.

    Args:
        rows: Number of rows in hex grid
        cols: Number of columns in hex grid
        radius: Cell radius
        depth: Cell depth
        center: (x, y, z) center offset for the entire slab

    Returns:
        vertices: Combined vertex array
        edges: Deduplicated edge list
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
        verts, edges, faces = create_hexagonal_cell(radius, depth, cell_center)

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

    # Deduplicate edges (shared walls between adjacent cells)
    edges = deduplicate_edges(all_edges, vertices)

    return vertices, edges, all_faces


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_cell(vertices, edges, faces=None, ax=None, edge_color=None, edge_width=None,
              face_alpha=None, face_color=None, render_mode=None, show_edges=None):
    """
    Plot a hexagonal cell or slab using matplotlib 3D.

    Args:
        vertices: Array of shape (N, 3)
        edges: List of (i, j) tuples
        faces: List of face vertex indices (for solid rendering)
        ax: Matplotlib 3D axis (creates new if None)
        edge_color, edge_width, face_alpha, face_color: Visual parameters
        render_mode: 'wireframe' or 'solid'
        show_edges: Whether to show edges in solid mode

    Returns:
        ax: The matplotlib 3D axis
    """
    if ax is None:
        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = fig.add_subplot(111, projection='3d')

    # Use global parameters if not specified
    if edge_color is None:
        edge_color = EDGE_COLOR
    if edge_width is None:
        edge_width = EDGE_WIDTH
    if face_alpha is None:
        face_alpha = FACE_ALPHA
    if face_color is None:
        face_color = FACE_COLOR
    if render_mode is None:
        render_mode = RENDER_MODE
    if show_edges is None:
        show_edges = SHOW_EDGES

    if render_mode == 'solid' and faces is not None:
        # Solid rendering with proper occlusion
        face_verts = []
        for face in faces:
            face_verts.append([vertices[i] for i in face])

        poly_collection = Poly3DCollection(face_verts,
                                          facecolors=face_color,
                                          alpha=face_alpha,
                                          edgecolors=edge_color if show_edges else None,
                                          linewidths=edge_width if show_edges else 0)
        ax.add_collection3d(poly_collection)
    else:
        # Wireframe rendering
        edge_segments = []
        for i, j in edges:
            edge_segments.append([vertices[i], vertices[j]])

        edge_collection = Line3DCollection(edge_segments, colors=edge_color,
                                           linewidths=edge_width)
        ax.add_collection3d(edge_collection)

    return ax


def setup_3d_axis(ax, vertices):
    """
    Configure 3D axis with equal aspect ratio and nice viewing angle.
    """
    # Set equal aspect ratio
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax


# ============================================================================
# MAIN - Test modules
# ============================================================================

if __name__ == "__main__":
    import sys

    # Choose what to visualize
    mode = sys.argv[1] if len(sys.argv) > 1 else "slab"

    if mode == "cell":
        # Test Order 0: Single cell
        vertices, edges, faces = create_hexagonal_cell(
            radius=CELL_RADIUS,
            depth=CELL_DEPTH,
            center=(0, 0, 0)
        )

        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = fig.add_subplot(111, projection='3d')

        plot_cell(vertices, edges, faces, ax=ax)
        setup_3d_axis(ax, vertices)

        ax.set_title('Order 0: Single Hexagonal Cell', fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig('order_0_cell.png', dpi=150, bbox_inches='tight')
        print("Saved: order_0_cell.png")

    elif mode == "slab":
        # Test Order 1: Honeycomb slab
        vertices, edges, faces = create_honeycomb_slab(
            rows=SLAB_ROWS,
            cols=SLAB_COLS,
            radius=CELL_RADIUS,
            depth=CELL_DEPTH,
            center=(0, 0, 0)
        )

        print(f"Slab: {len(vertices)} vertices, {len(edges)} edges (after deduplication), {len(faces)} faces")

        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = fig.add_subplot(111, projection='3d')

        plot_cell(vertices, edges, faces, ax=ax)
        setup_3d_axis(ax, vertices)

        ax.set_title(f'Order 1: Honeycomb Slab ({SLAB_ROWS}x{SLAB_COLS})',
                     fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig('order_1_slab.png', dpi=150, bbox_inches='tight')
        print("Saved: order_1_slab.png")

    plt.show()

