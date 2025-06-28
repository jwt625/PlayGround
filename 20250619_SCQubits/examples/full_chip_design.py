#!/usr/bin/env python3
"""
Example Full Chip Design - qiskit-metal
========================================

A comprehensive example showing how to create a multi-qubit quantum chip design
with various components including:
- Transmon pocket and crossmon qubits
- Tunable couplers
- Meandered transmission lines
- Capacitive couplers
- Wirebond launchers
- Multiplexed readout lines

Converted from Jupyter notebook to standalone Python script.
"""

import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal import MetalGUI, Dict, open_docs

# Import qubit components
from qiskit_metal.qlibrary.qubits.transmon_pocket_6 import TransmonPocket6
from qiskit_metal.qlibrary.qubits.transmon_cross_fl import TransmonCrossFL

# Import couplers
from qiskit_metal.qlibrary.couplers.tunable_coupler_01 import TunableCoupler01

# Import transmission lines
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder
from qiskit_metal.qlibrary.tlines.anchored_path import RouteAnchors

# Import capacitors and couplers
from qiskit_metal.qlibrary.lumped.cap_n_interdigital import CapNInterdigital
from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee

# Import terminations
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.terminations.launchpad_wb_coupled import LaunchpadWirebondCoupled

# Import analysis tools
from qiskit_metal.analyses.quantization import LOManalysis
from qiskit_metal.analyses.em.cpw_calculations import guided_wavelength
from collections import OrderedDict

def find_resonator_length(frequency, line_width, line_gap, N): 
    """
    Calculate resonator length for given frequency and geometry.
    
    Args:
        frequency: Frequency in GHz
        line_width: Line width in um
        line_gap: Line gap in um
        N: 2 for lambda/2, 4 for lambda/4
    
    Returns:
        String with length in mm
    """
    [lambdaG, etfSqrt, q] = guided_wavelength(frequency*10**9, line_width*10**-6,
                                              line_gap*10**-6, 750*10**-6, 200*10**-9)
    return str(lambdaG/N*10**3)+" mm"

def create_design():
    """Create the basic design and GUI."""
    print("🚀 Creating qiskit-metal design...")
    
    # Create design
    design = metal.designs.DesignPlanar()
    
    # Launch GUI
    gui = metal.MetalGUI(design)
    
    # Enable overwriting for iterative design
    design.overwrite_enabled = True
    
    # Set chip size
    design.chips.main.size.size_x = '11mm'
    design.chips.main.size.size_y = '9mm'
    
    print(f"✅ Design created with chip size: {design.chips.main.size.size_x} x {design.chips.main.size.size_y}")
    
    return design, gui

def create_qubits(design, gui):
    """Create all qubits for the chip."""
    print("🔬 Creating qubits...")
    
    # Main qubit options
    options = dict(
        pad_width = '425 um', 
        pocket_height = '650um',
        connection_pads=dict(
            readout = dict(loc_W=0, loc_H=-1, pad_width = '80um', pad_gap = '50um'),
            bus_01 = dict(loc_W=-1, loc_H=-1, pad_width = '60um', pad_gap = '10um'),
            bus_02 = dict(loc_W=-1, loc_H=+1, pad_width = '60um', pad_gap = '10um'),
            bus_03 = dict(loc_W=0, loc_H=+1, pad_width = '90um', pad_gap = '30um'),
            bus_04 = dict(loc_W=+1, loc_H=+1, pad_width = '60um', pad_gap = '10um'),
            bus_05 = dict(loc_W=+1, loc_H=-1, pad_width = '60um', pad_gap = '10um')        
        ))

    # Main qubit (center)
    q_main = TransmonPocket6(design,'Q_Main', options = dict(
            pos_x='0mm', 
            pos_y='-1mm', 
            gds_cell_name ='FakeJunction_01',
            hfss_inductance ='14nH',
            **options))

    # Crossmon qubits (west side)
    Q1 = TransmonCrossFL(design, 'Q1', options = dict(pos_x = '-2.75mm', pos_y='-1.8mm',
                                                     connection_pads = dict(
                                                         bus_01 = dict(connector_location = '180',claw_length ='95um'),
                                                         readout = dict(connector_location = '0')),
                                                     fl_options = dict()))

    Q2 = TransmonCrossFL(design, 'Q2', options = dict(pos_x = '-2.75mm', pos_y='-1.2mm', orientation = '180',
                                                     connection_pads = dict(
                                                         bus_02 = dict(connector_location = '0',claw_length ='95um'),
                                                         readout = dict(connector_location = '180')),
                                                     fl_options = dict()))

    # Tunable coupler between Q1 and Q2
    tune_c_Q12 = TunableCoupler01(design,'Tune_C_Q12', options = dict(pos_x = '-2.81mm', pos_y = '-1.5mm', 
                                                                      orientation=90, c_width='500um'))

    # Northern qubits (Q3, Q4, Q5)
    Q3 = TransmonPocket6(design,'Q3', options = dict(
            pos_x='-3mm', 
            pos_y='0.5mm', 
            gds_cell_name ='FakeJunction_01',
            hfss_inductance ='14nH',
            connection_pads = dict(
                bus_03 = dict(loc_W=0, loc_H=-1, pad_width = '80um', pad_gap = '15um'),
                bus_q3_q4 = dict(loc_W=1, loc_H=-1, pad_width = '80um', pad_gap = '15um'),
                readout = dict(loc_W=0, loc_H=1, pad_width = '80um', pad_gap = '50um'))))

    Q4 = TransmonPocket6(design,'Q4', options = dict(
            pos_x='0mm', 
            pos_y='1mm', 
            gds_cell_name ='FakeJunction_01',
            hfss_inductance ='14nH',
            connection_pads = dict(
                bus_04 = dict(loc_W=0, loc_H=-1, pad_width = '80um', pad_gap = '15um'),
                bus_q3_q4 = dict(loc_W=-1, loc_H=-1, pad_width = '80um', pad_gap = '15um'),
                bus_q4_q5 = dict(loc_W=1, loc_H=-1, pad_width = '80um', pad_gap = '15um'),
                readout = dict(loc_W=0, loc_H=1, pad_width = '80um', pad_gap = '50um'))))

    Q5 = TransmonPocket6(design,'Q5', options = dict(
            pos_x='3mm', 
            pos_y='0.5mm', 
            gds_cell_name ='FakeJunction_01',
            hfss_inductance ='14nH',
            connection_pads = dict(
                bus_05 = dict(loc_W=0, loc_H=-1, pad_width = '80um', pad_gap = '15um'),
                bus_q4_q5 = dict(loc_W=-1, loc_H=-1, pad_width = '80um', pad_gap = '15um'),
                readout = dict(loc_W=0, loc_H=1, pad_width = '80um', pad_gap = '50um'))))
    
    gui.rebuild()
    gui.autoscale()
    
    print("✅ Created 6 qubits: Q_Main (center), Q1&Q2 (crossmons), Q3&Q4&Q5 (northern)")
    
    return q_main, Q1, Q2, Q3, Q4, Q5, tune_c_Q12

def create_buses(design, gui):
    """Create coupling buses between qubits."""
    print("🔗 Creating coupling buses...")
    
    # Bus connecting Q_Main to Q1
    bus_01 = RouteMeander(design,'Bus_01', options = dict(hfss_wire_bonds = True, 
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q_Main',
                                                        pin='bus_01'),
                                                    end_pin=Dict(
                                                        component='Q1',
                                                        pin='bus_01')
                                                ),
                                                lead=Dict(
                                                    start_straight='125um',
                                                    end_straight = '225um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '1305um'),
                                                fillet = "99um",
                                                total_length = '6mm'))

    # Bus connecting Q_Main to Q2
    bus_02 = RouteMeander(design,'Bus_02', options = dict(hfss_wire_bonds = True, 
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q_Main',
                                                        pin='bus_02'),
                                                    end_pin=Dict(
                                                        component='Q2',
                                                        pin='bus_02')
                                                ),
                                                lead=Dict(
                                                    start_straight='325um',
                                                    end_straight = '125um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '450um'),
                                                fillet = "99um",
                                                total_length = '6.4mm'))
    
    gui.rebuild()
    print("✅ Created buses connecting Q_Main to crossmons Q1 and Q2")
    
    return bus_01, bus_02

def create_northern_buses(design, gui):
    """Create buses connecting northern qubits."""
    print("🔗 Creating northern qubit buses...")
    
    # Bus connecting Q_Main to Q3
    bus_03 = RouteMeander(design,'Bus_03', options = dict(hfss_wire_bonds = True, 
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q_Main',
                                                        pin='bus_03'),
                                                    end_pin=Dict(
                                                        component='Q3',
                                                        pin='bus_03')
                                                ),
                                                lead=Dict(
                                                    start_straight='225um',
                                                    end_straight = '25um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '50um'),
                                                fillet = "99um",
                                                total_length = '6.8mm'))

    # Jogs for controlled routing
    jogs_start = OrderedDict()
    jogs_start[0] = ["L", '250um']
    jogs_start[1] = ["R", '200um']

    # Bus connecting Q_Main to Q4
    bus_04 = RouteMeander(design,'Bus_04', options = dict(hfss_wire_bonds = True, 
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q_Main',
                                                        pin='bus_04'),
                                                    end_pin=Dict(
                                                        component='Q4',
                                                        pin='bus_04')
                                                ),
                                                lead=Dict(
                                                    start_straight='225um',
                                                    start_jogged_extension=jogs_start,
                                                ),
                                                meander=Dict(
                                                    asymmetry = '150um'),
                                                fillet = "99um",
                                                total_length = '7.2mm'))

    # Bus connecting Q_Main to Q5
    bus_05 = RouteMeander(design,'Bus_05', options = dict(hfss_wire_bonds = True, 
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q_Main',
                                                        pin='bus_05'),
                                                    end_pin=Dict(
                                                        component='Q5',
                                                        pin='bus_05')
                                                ),
                                                lead=Dict(
                                                    start_straight='225um',
                                                    end_straight = '25um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '50um'),
                                                fillet = "99um",
                                                total_length = '7.6mm'))

    # Inter-qubit buses (Q3-Q4-Q5 chain)
    bus_q3_q4 = RouteMeander(design,'Bus_Q3_Q4', options = dict(hfss_wire_bonds = True, 
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q3',
                                                        pin='bus_q3_q4'),
                                                    end_pin=Dict(
                                                        component='Q4',
                                                        pin='bus_q3_q4')
                                                ),
                                                lead=Dict(
                                                    start_straight='125um',
                                                    end_straight = '125um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '50um'),
                                                fillet = "99um",
                                                total_length = '6.4mm'))

    bus_q4_q5 = RouteMeander(design,'Bus_Q4_Q5', options = dict(hfss_wire_bonds = True, 
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q4',
                                                        pin='bus_q4_q5'),
                                                    end_pin=Dict(
                                                        component='Q5',
                                                        pin='bus_q4_q5')
                                                ),
                                                lead=Dict(
                                                    start_straight='125um',
                                                    end_straight = '25um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '50um'),
                                                fillet = "99um",
                                                total_length = '6.8mm'))
    
    gui.rebuild()
    print("✅ Created buses connecting Q_Main to northern qubits and inter-qubit connections")
    
    return bus_03, bus_04, bus_05, bus_q3_q4, bus_q4_q5

def create_launchers(design, gui):
    """Create wirebond launchers at chip edges."""
    print("📡 Creating wirebond launchers...")

    # Main readout launcher
    launch_qmain_read = LaunchpadWirebond(design, 'Launch_QMain_Read',
                                         options = dict(pos_x = '2mm', pos_y ='-4mm', orientation = '90'))

    # Q1 launchers (flux line and readout)
    launch_q1_fl = LaunchpadWirebond(design, 'Launch_Q1_FL',
                                    options = dict(pos_x = '0mm', pos_y ='-4mm', orientation = '90',
                                                  trace_width = '5um', trace_gap = '3um'))
    launch_q1_read = LaunchpadWirebondCoupled(design, 'Launch_Q1_Read',
                                             options = dict(pos_x = '-2mm', pos_y ='-4mm', orientation = '90'))

    # Tunable coupler launchers
    launch_tcoup_fl = LaunchpadWirebond(design, 'Launch_TuneC_FL',
                                       options = dict(pos_x = '-4mm', pos_y ='-4mm', orientation = '90',
                                                     trace_width = '5um', trace_gap = '3um'))
    launch_tcoup_read = LaunchpadWirebondCoupled(design, 'Launch_TuneC_Read',
                                                options = dict(pos_x = '-5mm', pos_y ='-3mm', orientation = '0'))

    # Q2 launchers
    launch_q2_read = LaunchpadWirebondCoupled(design, 'Launch_Q2_Read',
                                             options = dict(pos_x = '-5mm', pos_y ='-1mm', orientation = '0'))
    launch_q2_fl = LaunchpadWirebond(design, 'Launch_Q2_FL',
                                    options = dict(pos_x = '-5mm', pos_y ='1mm', orientation = '0',
                                                  trace_width = '5um', trace_gap = '3um'))

    # Northern launchers for multiplexed readout
    launch_nw = LaunchpadWirebond(design, 'Launch_NW',
                                 options = dict(pos_x = '-5mm', pos_y='3mm', orientation=0))
    launch_ne = LaunchpadWirebond(design, 'Launch_NE',
                                 options = dict(pos_x = '5mm', pos_y='3mm', orientation=180))

    gui.rebuild()
    print("✅ Created 8 wirebond launchers around chip perimeter")

    return (launch_qmain_read, launch_q1_fl, launch_q1_read, launch_tcoup_fl,
            launch_tcoup_read, launch_q2_read, launch_q2_fl, launch_nw, launch_ne)

def create_readout_lines(design, gui):
    """Create readout resonators and connections."""
    print("📊 Creating readout lines...")

    # Main qubit readout with finger capacitor
    read_q_main_cap = CapNInterdigital(design,'Read_Q_Main_Cap',
                                      options = dict(pos_x = '2mm', pos_y ='-3.5mm', orientation = '0'))

    jogs_end = OrderedDict()
    jogs_end[0] = ["L", '600um']
    jogs_start = OrderedDict()
    jogs_start[0] = ["L", '250um']

    read_q_main = RouteMeander(design,'Read_Q_Main', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q_Main',
                                                        pin='readout'),
                                                    end_pin=Dict(
                                                        component='Read_Q_Main_Cap',
                                                        pin='north_end')
                                                ),
                                                lead=Dict(
                                                    start_straight='725um',
                                                    end_straight = '625um',
                                                    start_jogged_extension = jogs_start,
                                                    end_jogged_extension = jogs_end
                                                ),
                                                meander=Dict(
                                                    asymmetry = '50um'),
                                                fillet = "99um",
                                                total_length = '5.6mm'))

    read_q_main_cap_launch = RoutePathfinder(design, 'Read_Q_Main_Cap_Launch',
                                            options = dict(hfss_wire_bonds = True,
                                                          pin_inputs = dict(
                                                              start_pin=Dict(
                                                                  component='Read_Q_Main_Cap',
                                                                  pin='south_end'),
                                                              end_pin=Dict(
                                                                  component='Launch_QMain_Read',
                                                                  pin='tie')),
                                                          lead=Dict(
                                                              start_straight='0um',
                                                              end_straight = '0um')))

    # Q1 readout
    read_q1 = RouteMeander(design,'Read_Q1', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q1',
                                                        pin='readout'),
                                                    end_pin=Dict(
                                                        component='Launch_Q1_Read',
                                                        pin='tie')
                                                ),
                                                lead=Dict(
                                                    start_straight='250um',
                                                    end_straight = '25um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '50um'),
                                                fillet = "99um",
                                                total_length = '6.8mm'))

    # Tunable coupler readout
    read_tunec = RouteMeander(design,'Read_TuneC', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Tune_C_Q12',
                                                        pin='Control'),
                                                    end_pin=Dict(
                                                        component='Launch_TuneC_Read',
                                                        pin='tie')
                                                ),
                                                lead=Dict(
                                                    start_straight='1525um',
                                                    end_straight = '125um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '50um'),
                                                fillet = "99um",
                                                total_length = '5.8mm'))

    # Q2 readout
    read_q2 = RouteMeander(design,'Read_Q2', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q2',
                                                        pin='readout'),
                                                    end_pin=Dict(
                                                        component='Launch_Q2_Read',
                                                        pin='tie')
                                                ),
                                                lead=Dict(
                                                    start_straight='350um',
                                                    end_straight = '0um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '-450um'),
                                                fillet = "99um",
                                                total_length = '5.4mm'))

    gui.rebuild()
    print("✅ Created readout resonators for Q_Main, Q1, Q2, and tunable coupler")

    return read_q_main, read_q1, read_tunec, read_q2

def create_flux_lines(design, gui):
    """Create flux control lines for crossmons and tunable coupler."""
    print("⚡ Creating flux control lines...")

    # Q1 flux line
    flux_line_Q1 = RoutePathfinder(design,'Flux_Line_Q1', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q1',
                                                        pin='flux_line'),
                                                    end_pin=Dict(
                                                        component='Launch_Q1_FL',
                                                        pin='tie')),
                                                fillet = '99um',
                                                trace_width = '5um',
                                                trace_gap = '3um'))

    # Tunable coupler flux line
    jogs_start = OrderedDict()
    jogs_start[0] = ["L", '750um']

    flux_line_tunec = RoutePathfinder(design,'Flux_Line_TuneC', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Tune_C_Q12',
                                                        pin='Flux'),
                                                    end_pin=Dict(
                                                        component='Launch_TuneC_FL',
                                                        pin='tie')),
                                                lead=Dict(
                                                    start_straight='875um',
                                                    end_straight = '350um',
                                                    start_jogged_extension = jogs_start
                                                ),
                                                fillet = '99um',
                                                trace_width = '5um',
                                                trace_gap = '3um'))

    # Q2 flux line
    jogs_start = OrderedDict()
    jogs_start[0] = ["L", '525um']
    jogs_start[1] = ["R", '625um']

    flux_line_Q2 = RoutePathfinder(design,'Flux_Line_Q2', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q2',
                                                        pin='flux_line'),
                                                    end_pin=Dict(
                                                        component='Launch_Q2_FL',
                                                        pin='tie')),
                                                lead=Dict(
                                                    start_straight='175um',
                                                    end_straight = '150um',
                                                    start_jogged_extension = jogs_start
                                                ),
                                                fillet = '99um',
                                                trace_width = '5um',
                                                trace_gap = '3um'))

    gui.rebuild()
    print("✅ Created flux control lines for Q1, Q2, and tunable coupler")

    return flux_line_Q1, flux_line_tunec, flux_line_Q2

def create_multiplexed_readout(design, gui):
    """Create multiplexed readout system for northern qubits (Q3, Q4, Q5)."""
    print("🔀 Creating multiplexed readout system...")

    # Create T-junctions for multiplexed readout
    q3_read_T = CoupledLineTee(design,'Q3_Read_T', options=dict(pos_x = '-3mm', pos_y = '3mm',
                                                            orientation = '0',
                                                            coupling_length = '200um',
                                                            open_termination = False))

    # Use finger count to set the width of the gap capacitance
    q4_read_T = CapNInterdigitalTee(design,'Q4_Read_T', options=dict(pos_x = '0mm', pos_y = '3mm',
                                                               orientation = '0',
                                                               finger_length = '0um',
                                                               finger_count = '8'))

    q5_read_T = CapNInterdigitalTee(design,'Q5_Read_T', options=dict(pos_x = '3mm', pos_y = '3mm',
                                                               orientation = '0',
                                                               finger_length = '50um',
                                                               finger_count = '11'))

    # Create readout resonators for each northern qubit
    read_q3 = RouteMeander(design,'Read_Q3', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q3',
                                                        pin='readout'),
                                                    end_pin=Dict(
                                                        component='Q3_Read_T',
                                                        pin='second_end')
                                                ),
                                                lead=Dict(
                                                    start_straight='150um',
                                                    end_straight = '150um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '0um'),
                                                fillet = "99um",
                                                total_length = '5mm'))

    read_q4 = RouteMeander(design,'Read_Q4', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q4',
                                                        pin='readout'),
                                                    end_pin=Dict(
                                                        component='Q4_Read_T',
                                                        pin='second_end')
                                                ),
                                                lead=Dict(
                                                    start_straight='125um',
                                                    end_straight = '125um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '0um'),
                                                fillet = "99um",
                                                total_length = '5.8mm'))

    read_q5 = RouteMeander(design,'Read_Q5', options = dict(hfss_wire_bonds = True,
                                                pin_inputs=Dict(
                                                    start_pin=Dict(
                                                        component='Q5',
                                                        pin='readout'),
                                                    end_pin=Dict(
                                                        component='Q5_Read_T',
                                                        pin='second_end')
                                                ),
                                                lead=Dict(
                                                    start_straight='125um',
                                                    end_straight = '125um'
                                                ),
                                                meander=Dict(
                                                    asymmetry = '0um'),
                                                fillet = "99um",
                                                total_length = '5.4mm'))

    # Create multiplexed transmission line connecting all T-junctions
    mp_tl_01 = RoutePathfinder(design, 'ML_TL_01', options = dict(hfss_wire_bonds = True,
                                                        pin_inputs = dict(
                                                            start_pin=Dict(
                                                                component='Launch_NW',
                                                                pin='tie'),
                                                            end_pin=Dict(
                                                                component='Q3_Read_T',
                                                                pin='prime_start'))))

    mp_tl_02 = RoutePathfinder(design, 'ML_TL_02', options = dict(hfss_wire_bonds = True,
                                                        pin_inputs = dict(
                                                            start_pin=Dict(
                                                                component='Q3_Read_T',
                                                                pin='prime_end'),
                                                            end_pin=Dict(
                                                                component='Q4_Read_T',
                                                                pin='prime_start'))))

    mp_tl_03 = RoutePathfinder(design, 'ML_TL_03', options = dict(hfss_wire_bonds = True,
                                                        pin_inputs = dict(
                                                            start_pin=Dict(
                                                                component='Q4_Read_T',
                                                                pin='prime_end'),
                                                            end_pin=Dict(
                                                                component='Q5_Read_T',
                                                                pin='prime_start'))))

    mp_tl_04 = RoutePathfinder(design, 'ML_TL_04', options = dict(hfss_wire_bonds = True,
                                                        pin_inputs = dict(
                                                            start_pin=Dict(
                                                                component='Q5_Read_T',
                                                                pin='prime_end'),
                                                            end_pin=Dict(
                                                                component='Launch_NE',
                                                                pin='tie'))))

    gui.rebuild()
    print("✅ Created multiplexed readout system for Q3, Q4, Q5")

    return (q3_read_T, q4_read_T, q5_read_T, read_q3, read_q4, read_q5,
            mp_tl_01, mp_tl_02, mp_tl_03, mp_tl_04)

def run_analysis(design):
    """Run basic capacitance extraction and LOM analysis."""
    print("🔬 Running LOM analysis...")

    try:
        # Create LOM analysis instance
        c1 = LOManalysis(design, "q3d")

        # Configure simulation settings
        c1.sim.setup.name = 'Tune_Q_Main'
        c1.sim.setup.max_passes = 16
        c1.sim.setup.min_converged_passes = 2
        c1.sim.setup.percent_error = 0.05

        print("✅ LOM analysis configured (simulation not run in this demo)")
        print("💡 To run simulation: c1.sim.run(components=['Q_Main'], open_terminations=[...])")

        return c1

    except Exception as e:
        print(f"⚠️  Analysis setup failed: {e}")
        print("💡 This is normal if HFSS/Q3D is not available")
        return None

def main():
    """Main function to create the complete chip design."""
    print("🎯 Starting Full Chip Design Example")
    print("=" * 50)

    # Create design and GUI
    design, gui = create_design()

    # Create all components
    qubits = create_qubits(design, gui)
    buses = create_buses(design, gui)
    northern_buses = create_northern_buses(design, gui)
    launchers = create_launchers(design, gui)
    readout_lines = create_readout_lines(design, gui)
    flux_lines = create_flux_lines(design, gui)
    multiplexed_readout = create_multiplexed_readout(design, gui)

    # Final rebuild and display
    gui.rebuild()
    gui.autoscale()

    print("\n🎉 Chip design complete!")
    print("=" * 50)
    print("📊 Design Summary:")
    print(f"   • Qubits: 6 (1 main + 2 crossmons + 3 northern)")
    print(f"   • Couplers: 1 tunable coupler")
    print(f"   • Buses: 7 coupling buses")
    print(f"   • Launchers: 9 wirebond pads")
    print(f"   • Readout lines: 4 individual + 1 multiplexed")
    print(f"   • Flux lines: 3 control lines")
    print(f"   • Chip size: {design.chips.main.size.size_x} x {design.chips.main.size.size_y}")

    # Optional analysis
    analysis = run_analysis(design)

    print("\n💡 Next steps:")
    print("   • Tune component parameters")
    print("   • Run electromagnetic simulations")
    print("   • Export to GDS for fabrication")
    print("   • Analyze crosstalk and coupling strengths")

    print("\n🖥️  GUI is now open - explore your design!")

    return design, gui

if __name__ == "__main__":
    design, gui = main()

    # Keep the script running so GUI stays open
    print("\n⌨️  Press Ctrl+C to exit...")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
