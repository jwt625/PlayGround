import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ezdxf

# Load and process image
image_path = "in.png"  # Replace with your filename
img = Image.open(image_path).convert("L")
img_array = np.array(img)

# Determine grid resolution
grid_size = 16
height, width = img_array.shape
x_step = width / grid_size
y_step = height / grid_size

# Sample at cell centers
binary_grid = np.zeros((grid_size, grid_size), dtype=int)
for j in range(grid_size):
    for i in range(grid_size):
        x = int((i + 0.5) * x_step)
        y = int((j + 0.5) * y_step)
        pixel = img_array[y, x]
        binary_grid[j, i] = 1 if pixel > 128 else 0  # 1: bright, 0: dark

# Print the binary grid
print("16x16 Binary Grid (0 = dark, 1 = bright):")
print(binary_grid)

# Save the binary grid as PNG for verification
plt.figure(figsize=(8, 8))
plt.imshow(binary_grid, cmap='gray', interpolation='nearest')
plt.title("Binary Grid Visualization")
plt.axis('off')
plt.savefig("binary_grid_output.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()  # Close the figure to avoid GUI issues
print("Binary grid saved as: binary_grid_output.png")

# Convert to DXF file
def create_dxf_from_binary_grid(binary_grid, total_width_mm=2.0, filename="output.dxf"):
    """
    Convert binary grid to DXF file with specified total width

    Args:
        binary_grid: 2D numpy array with 0s and 1s
        total_width_mm: Total width of the grid in millimeters
        filename: Output DXF filename
    """
    # Create a new DXF document
    doc = ezdxf.new('R2010')  # DXF R2010 = AutoCAD 2010
    msp = doc.modelspace()

    # Calculate cell size
    grid_size = binary_grid.shape[0]  # Assuming square grid
    cell_size = total_width_mm / grid_size

    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Total width: {total_width_mm} mm")
    print(f"Cell size: {cell_size} mm")

    # Create rectangles for cells with value 1 (bright pixels)
    rectangle_count = 0
    for j in range(grid_size):
        for i in range(grid_size):
            if binary_grid[j, i] == 1:  # Only create rectangles for bright pixels
                # Calculate rectangle coordinates
                x1 = i * cell_size
                y1 = -j * cell_size
                x2 = x1 + cell_size
                y2 = y1 - cell_size

                # Create rectangle using LWPOLYLINE (lightweight polyline) - closed
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                msp.add_lwpolyline(points, close=True)

                # Also add a filled SOLID entity for better visibility
                # msp.add_solid([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                rectangle_count += 1

    print(f"Created {rectangle_count} rectangles in DXF")

    # Save the DXF file
    doc.saveas(filename)
    print(f"DXF file saved as: {filename}")

    return filename

# Create DXF file from the binary grid
dxf_filename = create_dxf_from_binary_grid(binary_grid, total_width_mm=2.0, filename="binary_grid_2mm.dxf")
print(f"Binary grid converted to DXF: {dxf_filename}")
