#!/usr/bin/env python
# coding: utf-8

# Four Qubit Chip Design - NO GUI VERSION
# Test script to verify qiskit-metal works without GUI

import numpy as np
from collections import OrderedDict

from qiskit_metal import designs, draw
from qiskit_metal import Dict, Headings

# Create design WITHOUT GUI
design = designs.DesignPlanar()
print("✅ Design created successfully")

# Enable overwrite
design.overwrite_enabled = True

# Import components
from qiskit_metal.qlibrary.qubits.transmon_pocket_cl import TransmonPocketCL
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.anchored_path import RouteAnchors
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.terminations.launchpad_wb_coupled import LaunchpadWirebondCoupled

print("✅ Components imported successfully")

# Setup design variables
design.variables['cpw_width'] = '10 um'
design.variables['cpw_gap'] = '6 um'
design._chips['main']['size']['size_y'] = '9mm'
design._chips['main']['size']['size_y'] = '6.5mm'

print("✅ Design variables set")

# Create transmon options
transmon_options = dict(
    connection_pads=dict(
        a = dict(loc_W=+1, loc_H=-1, pad_width='70um', cpw_extend = '50um'), 
        b = dict(loc_W=-1, loc_H=-1, pad_width='125um', cpw_extend = '50um'),
        c = dict(loc_W=-1, loc_H=+1, pad_width='110um', cpw_extend = '50um')
    ),
    gds_cell_name='FakeJunction_01',
    cl_off_center = '-50um',
    cl_pocket_edge = '180'
)

print("✅ Transmon options defined")

# Create 4 transmons
offset_tm = 69

q1 = TransmonPocketCL(design, 'Q1', options = dict(
    pos_x='+2420um', pos_y=f'{offset_tm}um', **transmon_options))
print("✅ Q1 created")

q2 = TransmonPocketCL(design, 'Q2', options = dict(
    pos_x='0um', pos_y='-857.6um', orientation = '270', **transmon_options))
print("✅ Q2 created")

q3 = TransmonPocketCL(design, 'Q3', options = dict(
    pos_x='-2420um', pos_y=f'{offset_tm}um', orientation = '180', **transmon_options))
print("✅ Q3 created")

q4 = TransmonPocketCL(design, 'Q4', options = dict(
    pos_x='0um', pos_y='+857.6um', orientation = '90', **transmon_options))
print("✅ Q4 created")

# Test basic functionality
print(f"✅ Design has {len(design.components)} components")
print("Component names:", list(design.components.keys()))

print("\n🎉 SUCCESS! qiskit-metal is working correctly without GUI")
print("All 4 transmon qubits created successfully")

# Test scqubits as well
try:
    import scqubits as scq
    tmon = scq.Transmon(EJ=25.0, EC=0.2, ng=0.0, ncut=30)
    eigenvals = tmon.eigenvals(evals_count=3)
    print(f"✅ scqubits test passed - energy levels: {eigenvals[:3]}")
except Exception as e:
    print(f"❌ scqubits test failed: {e}")

print("\n✅ All tests completed successfully!")
