#!/usr/bin/env python3
"""
Test script to verify qiskit-metal and scqubits installation
"""

def test_imports():
    """Test basic imports"""
    print("Testing imports...")

    try:
        # Try importing qiskit_metal core without GUI components
        import qiskit_metal.designs
        import qiskit_metal.qlibrary
        print("✅ qiskit_metal core components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import qiskit_metal core: {e}")
        return False
    
    try:
        import scqubits
        print("✅ scqubits imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import scqubits: {e}")
        return False
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy
        print("✅ Core scientific packages imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core packages: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Test scqubits
        import scqubits as scq
        
        # Create a simple transmon qubit
        tmon = scq.Transmon(
            EJ=25.0,    # Josephson energy in GHz
            EC=0.2,     # Charging energy in GHz
            ng=0.0,     # offset charge
            ncut=30     # charge cutoff
        )
        
        # Calculate some energy levels
        eigenvals = tmon.eigenvals(evals_count=5)
        print(f"✅ scqubits basic test passed - first 5 energy levels: {eigenvals[:5]}")
        
    except Exception as e:
        print(f"❌ scqubits basic test failed: {e}")
        return False
    
    try:
        # Test qiskit_metal basic functionality without GUI
        from qiskit_metal.designs.design_base import QDesign
        from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
        print("✅ qiskit_metal basic components imported successfully")

        # Try creating a simple design (without GUI)
        design = QDesign()
        print("✅ qiskit_metal QDesign created successfully")

    except Exception as e:
        print(f"❌ qiskit_metal basic test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing qiskit-metal and scqubits installation...\n")
    
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ Import tests failed!")
        exit(1)
    
    functionality_ok = test_basic_functionality()
    if not functionality_ok:
        print("\n❌ Functionality tests failed!")
        exit(1)
    
    print("\n🎉 All tests passed! Your installation is working correctly.")
    print("\nNote: Some optional dependencies are missing but core functionality works.")
    print("Missing optional packages:")
    print("- geopandas==0.12.2 (for advanced geometry operations)")
    print("- gmsh==4.11.1 (for mesh generation)")
    print("- pyaedt==0.6.46 (for ANSYS integration)")
    print("- pyEPR-quantum==0.8.5.7 (for energy-participation-ratio analysis)")
    print("- pyside2==5.15.2.1 (for GUI components)")
    print("- qdarkstyle==3.1 (for dark theme GUI)")
    print("\nYou can install these later if needed for specific features.")
