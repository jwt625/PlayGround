#!/bin/bash

# ParaView launcher script for Palace simulation results
# Usage: ./open_paraview.sh [dataset_type]
# dataset_type: transient (default), boundary, or both

DATASET_TYPE=${1:-transient}

echo "Opening ParaView with Palace simulation results..."
echo "Dataset type: $DATASET_TYPE"

case $DATASET_TYPE in
    "transient")
        echo "Opening transient volume data..."
        paraview example_coaxial_postpro_open/paraview/transient/transient.pvd &
        ;;
    "boundary")
        echo "Opening boundary surface data..."
        paraview example_coaxial_postpro_open/paraview/transient_boundary/transient_boundary.pvd &
        ;;
    "both")
        echo "Opening both transient and boundary data..."
        paraview example_coaxial_postpro_open/paraview/transient/transient.pvd example_coaxial_postpro_open/paraview/transient_boundary/transient_boundary.pvd &
        ;;
    *)
        echo "Unknown dataset type: $DATASET_TYPE"
        echo "Available options: transient, boundary, both"
        exit 1
        ;;
esac

echo "ParaView should be opening shortly..."
echo ""
echo "Quick tips for viewing Palace electromagnetic simulation data:"
echo "1. Use the 'Play' button to animate through time steps"
echo "2. Common fields to visualize:"
echo "   - Electric field (E): Vector field showing electric field distribution"
echo "   - Magnetic field (H): Vector field showing magnetic field distribution"
echo "   - Current density (J): Vector field showing current flow"
echo "   - Power density: Scalar field showing power dissipation"
echo "3. Use 'Filters' > 'Common' > 'Slice' to create cross-sections"
echo "4. Use 'Filters' > 'Common' > 'Streamline' to visualize field lines"
echo "5. Change coloring in the 'Properties' panel"
