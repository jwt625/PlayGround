#!/usr/bin/env python3
"""
QASM to Circuit Format Converter
Converts OpenQASM 2.0 files to psiquantum.circuit JSON format
Ignores measurement and conditional operations
"""

import json
import re
import sys
from typing import List, Dict, Any
from datetime import datetime


class QASMToCircuitConverter:
    def __init__(self):
        self.registers = []
        self.operations = []
        self.qreg_map = {}  # Maps qasm register names to circuit register IDs
        self.gate_counter = 0
        self.column_counter = 0
        
    def parse_qasm(self, qasm_content: str):
        """Parse QASM content and extract registers and gates"""
        lines = qasm_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//'):
                continue
            
            # Skip header lines
            if line.startswith('OPENQASM') or line.startswith('include'):
                continue
            
            # Stop at measurement operations
            if line.startswith('creg') or line.startswith('measure') or line.startswith('if'):
                break
            
            # Parse quantum register declarations
            if line.startswith('qreg'):
                self._parse_qreg(line)
            
            # Parse gate operations
            elif any(line.startswith(gate) for gate in ['x ', 'h ', 'cx ', 'ccx ']):
                self._parse_gate(line)
    
    def _parse_qreg(self, line: str):
        """Parse quantum register declaration: qreg name[size];"""
        match = re.match(r'qreg\s+(\w+)\[(\d+)\];', line)
        if match:
            name = match.group(1)
            size = int(match.group(2))
            
            # For now, assume size 1 registers
            reg_id = f"r{len(self.registers) + 1}"
            self.qreg_map[name] = reg_id
            
            self.registers.append({
                "type": "register",
                "id": reg_id,
                "label": name
            })
    
    def _parse_gate(self, line: str):
        """Parse gate operations"""
        # Remove semicolon
        line = line.rstrip(';')
        
        # Parse single-qubit gates: x qubit, h qubit
        match = re.match(r'(x|h)\s+(\w+)', line)
        if match:
            gate_type = match.group(1)
            qubit = match.group(2)
            self._add_single_qubit_gate(gate_type, qubit)
            return
        
        # Parse CNOT: cx control, target
        match = re.match(r'cx\s+(\w+),\s*(\w+)', line)
        if match:
            control = match.group(1)
            target = match.group(2)
            self._add_cnot_gate(control, target)
            return
        
        # Parse Toffoli: ccx control1, control2, target
        match = re.match(r'ccx\s+(\w+),\s*(\w+),\s*(\w+)', line)
        if match:
            control1 = match.group(1)
            control2 = match.group(2)
            target = match.group(3)
            self._add_toffoli_gate(control1, control2, target)
            return
    
    def _add_single_qubit_gate(self, gate_type: str, qubit: str):
        """Add single-qubit gate to operations"""
        self.gate_counter += 1

        # In circuit format: X gate is represented as cnot without controlled_by
        # H gate is represented as hadamard
        gate_type_map = {
            'x': 'cnot',
            'h': 'hadamard'
        }

        gate = {
            "input_registers": self.qreg_map[qubit],
            "gate_type": gate_type_map[gate_type],
            "type": "gate",
            "id": f"gate{self.gate_counter}"
        }

        self.operations.append(gate)
    
    def _add_cnot_gate(self, control: str, target: str):
        """Add CNOT gate to operations"""
        self.gate_counter += 1
        
        gate = {
            "input_registers": self.qreg_map[target],
            "gate_type": "cnot",
            "type": "gate",
            "id": f"gate{self.gate_counter}",
            "controlled_by": self.qreg_map[control]
        }
        
        self.operations.append(gate)
    
    def _add_toffoli_gate(self, control1: str, control2: str, target: str):
        """Add Toffoli (CCX) gate to operations - uses cnot with multiple controls"""
        self.gate_counter += 1

        gate = {
            "input_registers": self.qreg_map[target],
            "gate_type": "cnot",
            "type": "gate",
            "id": f"gate{self.gate_counter}",
            "controlled_by": f"{self.qreg_map[control1]},{self.qreg_map[control2]}"
        }

        self.operations.append(gate)
    
    def generate_circuit_json(self, title: str = "Converted Circuit") -> Dict[str, Any]:
        """Generate circuit format JSON"""
        circuit = {
            "record_type": "psiquantum.circuit",
            "title": title,
            "copyright": f"Copyright (c) {datetime.now().year}. All rights reserved.",
            "record_format": 2,
            "modified": datetime.now().isoformat() + "Z",
            "style": {
                "copyright_el": {
                    "visible": False
                }
            },
            "operations": self.registers + self.operations
        }
        
        return circuit


def convert_qasm_to_circuit(qasm_file: str, output_file: str = None, title: str = None):
    """Convert QASM file to circuit JSON format"""
    # Read QASM file
    with open(qasm_file, 'r') as f:
        qasm_content = f.read()
    
    # Create converter and parse
    converter = QASMToCircuitConverter()
    converter.parse_qasm(qasm_content)
    
    # Generate circuit JSON
    if title is None:
        title = f"Converted from {qasm_file}"
    circuit_json = converter.generate_circuit_json(title)
    
    # Write output
    if output_file is None:
        output_file = qasm_file.replace('.qasm', '.circuit')
    
    with open(output_file, 'w') as f:
        json.dump(circuit_json, f, indent=2)
    
    print(f"Converted {qasm_file} to {output_file}")
    print(f"  Registers: {len(converter.registers)}")
    print(f"  Gates: {len(converter.operations)}")
    
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qasm_to_circuit.py <input.qasm> [output.circuit] [title]")
        sys.exit(1)
    
    qasm_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    title = sys.argv[3] if len(sys.argv) > 3 else None
    
    convert_qasm_to_circuit(qasm_file, output_file, title)

