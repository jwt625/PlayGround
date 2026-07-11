from __future__ import annotations

import pytest

from eo_fem.backends.mesh_sequence import parse_mesh_sequence


def test_mesh_sequence_preserves_comsol_style_operation_order() -> None:
    config = {
        "Mesh": {
            "name": "mesh1",
            "dimension": 2,
            "sequence_type": "user_controlled",
            "operations": {
                "global_size": {"type": "size", "h_max": 1.0, "h_min": 0.1},
                "electrode_proximity": {
                    "type": "boundary_proximity",
                    "selection": "signal_boundary",
                    "h_min": 0.05,
                    "h_max": 1.0,
                    "distance_min": 0.1,
                    "distance_max": 0.5,
                },
                "free_triangular": {"type": "free_triangular"},
            },
        }
    }
    sequence = parse_mesh_sequence(config, default_size=2.0)
    assert [operation.name for operation in sequence.operations] == [
        "global_size",
        "electrode_proximity",
        "free_triangular",
    ]
    assert sequence.operations[1].selection == "signal_boundary"


def test_mesh_sequence_rejects_missing_dimension_generator() -> None:
    config = {
        "Mesh": {
            "dimension": 3,
            "operations": {
                "global_size": {"type": "size", "h_max": 1.0},
                "free_triangular": {"type": "free_triangular"},
            },
        }
    }
    with pytest.raises(ValueError, match="free_tetrahedral"):
        parse_mesh_sequence(config, default_size=2.0)


def test_mesh_sequence_rejects_inverted_proximity_transition() -> None:
    config = {
        "Mesh": {
            "operations": {
                "bad": {
                    "type": "boundary_proximity",
                    "selection": "wall",
                    "h_min": 0.1,
                    "h_max": 1.0,
                    "distance_min": 2.0,
                    "distance_max": 1.0,
                },
                "free_triangular": {"type": "free_triangular"},
            }
        }
    }
    with pytest.raises(ValueError, match="distance_min <= distance_max"):
        parse_mesh_sequence(config, default_size=2.0)
