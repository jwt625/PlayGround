"""
Custom AWG with debug prints and waveguide length information
Based on gdsfactory's built-in AWG component
"""
import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec
from functools import partial


@gf.cell
def awg_debug(
    arms: int = 10,
    outputs: int = 3,
    free_propagation_region_input_function: ComponentSpec = None,
    free_propagation_region_output_function: ComponentSpec = None,
    fpr_spacing: float = 50.0,
    arm_spacing: float = 1.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """
    Custom AWG with debug information.
    
    Returns a basic Arrayed Waveguide grating with debug prints.
    
    Args:
        arms: number of arms.
        outputs: number of outputs.
        free_propagation_region_input_function: for input.
        free_propagation_region_output_function: for output.
        fpr_spacing: x separation between input/output free propagation region.
        arm_spacing: y separation between arms.
        cross_section: cross_section function.
    """
    print("\n" + "=" * 80)
    print("AWG DEBUG - Creating AWG")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  arms: {arms}")
    print(f"  outputs: {outputs}")
    print(f"  fpr_spacing: {fpr_spacing}")
    print(f"  arm_spacing: {arm_spacing}")
    print(f"  cross_section: {cross_section}")
    
    c = Component()
    
    # Set default FPR functions if not provided
    if free_propagation_region_input_function is None:
        free_propagation_region_input_function = partial(
            gf.components.free_propagation_region, inputs=1
        )
    if free_propagation_region_output_function is None:
        free_propagation_region_output_function = partial(
            gf.components.free_propagation_region, inputs=10, width1=10, width2=20.0
        )
    
    # Create FPRs
    fpr_in = gf.get_component(
        free_propagation_region_input_function,
        inputs=1,
        outputs=arms,
        cross_section=cross_section,
    )
    fpr_out = gf.get_component(
        free_propagation_region_output_function,
        inputs=outputs,
        outputs=arms,
        cross_section=cross_section,
    )
    
    print(f"\nFPR Input:")
    print(f"  Size: {fpr_in.bbox().width():.1f} x {fpr_in.bbox().height():.1f} µm")
    print(f"  Ports: {len(fpr_in.ports)}")
    
    print(f"\nFPR Output:")
    print(f"  Size: {fpr_out.bbox().width():.1f} x {fpr_out.bbox().height():.1f} µm")
    print(f"  Ports: {len(fpr_out.ports)}")
    
    # Add FPRs to component
    fpr_in_ref = c.add_ref(fpr_in)
    fpr_out_ref = c.add_ref(fpr_out)
    
    # Rotate FPRs
    fpr_in_ref.rotate(90)
    fpr_out_ref.rotate(90)
    
    # Position FPRs
    fpr_out_ref.x += fpr_spacing
    
    print(f"\nFPR positions after rotation and spacing:")
    print(f"  FPR Input center: {fpr_in_ref.center}")
    print(f"  FPR Output center: {fpr_out_ref.center}")
    
    # Get ports for routing
    ports_out = gf.port.get_ports_list(fpr_out_ref, prefix="E")
    ports_in = gf.port.get_ports_list(fpr_in_ref, prefix="E")
    
    print(f"\nPorts to connect:")
    print(f"  Output FPR ports (E): {len(ports_out)}")
    print(f"  Input FPR ports (E): {len(ports_in)}")
    
    print(f"\nOutput FPR port positions:")
    for i, port in enumerate(ports_out[:5]):  # Show first 5
        print(f"  E{i}: {port.center}, orientation: {port.orientation}°")
    if len(ports_out) > 5:
        print(f"  ... ({len(ports_out) - 5} more ports)")
    
    print(f"\nInput FPR port positions:")
    for i, port in enumerate(ports_in[:5]):  # Show first 5
        print(f"  E{i}: {port.center}, orientation: {port.orientation}°")
    if len(ports_in) > 5:
        print(f"  ... ({len(ports_in) - 5} more ports)")
    
    # Route the bundle
    print(f"\nRouting bundle with separation={arm_spacing}...")
    routes = gf.routing.route_bundle(
        c,
        ports_out,
        ports_in,
        sort_ports=True,
        separation=arm_spacing,
        cross_section=cross_section,
    )
    
    print(f"\nRoute bundle results:")
    print(f"  Number of routes: {len(routes)}")
    print(f"\nDetailed waveguide analysis:")
    print(f"{'WG':<4} {'Length(µm)':<12} {'Backbone(µm)':<14} {'#Bends':<8} {'Diff(µm)':<10} {'Start Port':<25} {'End Port':<25}")
    print("-" * 120)

    lengths = []
    for i, route in enumerate(routes):
        length_um = route.length * 0.001  # Convert dbu to µm
        backbone_um = route.length_backbone * 0.001
        diff = length_um - lengths[-1] if i > 0 else 0

        # Get port info
        start_port_info = f"({route.start_port.center[0]:.1f}, {route.start_port.center[1]:.1f})"
        end_port_info = f"({route.end_port.center[0]:.1f}, {route.end_port.center[1]:.1f})"

        print(f"{i:<4} {length_um:<12.2f} {backbone_um:<14.2f} {route.n_bend90:<8} {diff:<10.2f} {start_port_info:<25} {end_port_info:<25}")
        lengths.append(length_um)

        # Print backbone points for routes with big jumps
        if abs(diff) > 20 and i > 0:
            print(f"     ⚠️  BIG JUMP! Backbone points:")
            for j, pt in enumerate(route.backbone):
                pt_um = (pt.x * 0.001, pt.y * 0.001)
                print(f"        Point {j}: ({pt_um[0]:.2f}, {pt_um[1]:.2f})")

    print(f"\nLength statistics:")
    print(f"  Min length: {min(lengths):.2f} µm")
    print(f"  Max length: {max(lengths):.2f} µm")
    print(f"  Average length: {sum(lengths)/len(lengths):.2f} µm")
    print(f"  Length difference (max-min): {max(lengths) - min(lengths):.2f} µm")
    
    # Add ports
    c.add_port("o1", port=fpr_in_ref.ports["o1"])
    
    for i, port in enumerate(gf.port.get_ports_list(fpr_out_ref, prefix="W")):
        c.add_port(f"E{i}", port=port)
    
    print(f"\nFinal AWG size: {c.bbox().width():.1f} x {c.bbox().height():.1f} µm")
    print("=" * 80)
    
    return c


@gf.cell
def awg_manual_route(
    arms: int = 10,
    outputs: int = 3,
    fpr_spacing: float = 50.0,
    delta_length: float = 10.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """
    AWG with manually routed waveguides for consistent path length differences.

    Args:
        arms: number of arms.
        outputs: number of outputs.
        fpr_spacing: x separation between input/output FPR.
        delta_length: path length difference between adjacent waveguides (µm).
        cross_section: cross_section function.
    """
    print("\n" + "=" * 80)
    print("AWG MANUAL ROUTE - Creating AWG with consistent path lengths")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  arms: {arms}")
    print(f"  outputs: {outputs}")
    print(f"  fpr_spacing: {fpr_spacing}")
    print(f"  delta_length: {delta_length} µm")

    c = Component()

    # Create FPRs
    fpr_in = gf.get_component(
        gf.components.free_propagation_region,
        inputs=1,
        outputs=arms,
        cross_section=cross_section,
    )
    fpr_out = gf.get_component(
        gf.components.free_propagation_region,
        inputs=outputs,
        outputs=arms,
        width1=10,
        width2=20.0,
        cross_section=cross_section,
    )

    # Add FPRs
    fpr_in_ref = c.add_ref(fpr_in)
    fpr_out_ref = c.add_ref(fpr_out)

    # Rotate and position
    fpr_in_ref.rotate(90)
    fpr_out_ref.rotate(90)
    fpr_out_ref.x += fpr_spacing

    # Get ports
    ports_out = gf.port.get_ports_list(fpr_out_ref, prefix="E")
    ports_in = gf.port.get_ports_list(fpr_in_ref, prefix="E")

    print(f"\nPort ordering check:")
    print(f"  Output FPR E ports (first 3):")
    for i in range(min(3, len(ports_out))):
        print(f"    E{i}: x={ports_out[i].center[0]:.2f}")
    print(f"  Input FPR E ports (first 3):")
    for i in range(min(3, len(ports_in))):
        print(f"    E{i}: x={ports_in[i].center[0]:.2f}")

    # Both FPRs have ports in same order (both decreasing X)
    # To avoid crossings, we need to reverse one of them
    # So E0_out connects to E19_in, E1_out to E18_in, etc.
    print(f"\n⚠️  Reversing input port order to avoid crossings...")
    ports_in = list(reversed(ports_in))

    print(f"  After reversal, connections will be:")
    for i in range(min(3, len(ports_out))):
        print(f"    E{i}_out (x={ports_out[i].center[0]:.2f}) → E{len(ports_in)-1-i}_in (x={ports_in[i].center[0]:.2f})")

    print(f"\nManually routing {len(ports_out)} waveguides...")

    # Calculate horizontal distances for all waveguides first
    horizontal_distances = []
    for i in range(len(ports_out)):
        p_out = ports_out[i]
        p_in = ports_in[i]
        h_dist = abs(p_out.center[0] - p_in.center[0])
        horizontal_distances.append(h_dist)

    # Minimum detour height for bend radius
    min_detour = 15.0  # µm, minimum for bend radius

    print(f"\nHorizontal distances range: {min(horizontal_distances):.2f} to {max(horizontal_distances):.2f} µm")

    # Sort waveguides by horizontal distance to determine routing order
    # Shortest horizontal distance should have lowest detour (closest to FPR)
    waveguide_info = []
    for i in range(len(ports_out)):
        waveguide_info.append({
            'index': i,
            'h_dist': horizontal_distances[i],
            'p_out': ports_out[i],
            'p_in': ports_in[i]
        })

    # Sort by horizontal distance (ascending: shortest first)
    waveguide_info_sorted = sorted(waveguide_info, key=lambda x: x['h_dist'])

    # Assign detour heights: shortest h_dist gets min_detour, then increment by small amount
    detour_increment = 1.0  # µm - small increment to avoid crossings

    # Route each waveguide individually with controlled path length
    lengths = []
    for rank, wg_info in enumerate(waveguide_info_sorted):
        i = wg_info['index']
        p_out = wg_info['p_out']
        p_in = wg_info['p_in']
        h_dist = wg_info['h_dist']

        # Shortest horizontal distance gets minimum detour
        # Each subsequent waveguide (by h_dist) gets slightly higher detour
        detour_height = min_detour + rank * detour_increment

        # Waypoints: start -> up -> across -> down -> end
        # Both ports are at same Y, pointing up (90°)
        y_detour = p_out.center[1] + detour_height

        waypoints = [
            (p_out.center[0], y_detour),  # Go straight up from start
            (p_in.center[0], y_detour),   # Go across
        ]

        route = gf.routing.route_single(
            c,
            p_out,
            p_in,
            waypoints=waypoints,
            cross_section=cross_section,
        )

        actual_length = route.length * 0.001  # Convert to µm
        lengths.append((i, actual_length))  # Store with original index

        print(f"  WG {i}: h_dist={h_dist:.2f} µm, rank={rank}, detour={detour_height:.2f} µm, length={actual_length:.2f} µm")

    # Sort lengths back by original index and print differences
    lengths_sorted = sorted(lengths, key=lambda x: x[0])
    print(f"\nLength differences (in waveguide index order):")
    for idx in range(len(lengths_sorted)):
        i, length = lengths_sorted[idx]
        if idx > 0:
            prev_length = lengths_sorted[idx-1][1]
            diff = length - prev_length
            print(f"  WG {i}: length={length:.2f} µm, diff from WG{idx-1}={diff:.2f} µm")
        else:
            print(f"  WG {i}: length={length:.2f} µm")

    # Add ports
    c.add_port("o1", port=fpr_in_ref.ports["o1"])
    for i, port in enumerate(gf.port.get_ports_list(fpr_out_ref, prefix="W")):
        c.add_port(f"E{i}", port=port)

    print(f"\nLength statistics:")
    all_lengths = [l[1] for l in lengths]
    print(f"  Min: {min(all_lengths):.2f} µm")
    print(f"  Max: {max(all_lengths):.2f} µm")
    print(f"  Average: {sum(all_lengths)/len(all_lengths):.2f} µm")
    print(f"  Target delta: {delta_length:.2f} µm")
    print(f"\nFinal size: {c.bbox().width():.1f} x {c.bbox().height():.1f} µm")
    print("=" * 80)

    return c


if __name__ == "__main__":
    # Test manual routing
    print("\n\nTEST: Manual AWG with consistent path lengths")
    awg_manual = awg_manual_route(arms=20, outputs=8, delta_length=10.0)
    awg_manual.write_gds("awg_comparison.gds")
    print("\n\nWritten to: awg_comparison.gds")

