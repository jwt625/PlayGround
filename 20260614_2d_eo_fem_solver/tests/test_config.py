from eo_fem.config import parse_simple_yaml


def test_parse_simple_yaml_nested_scalars():
    cfg = parse_simple_yaml(
        """
Simulation:
  name: demo
  mesh_nx: 11
Materials:
  background:
    eps_r: 3.9
"""
    )
    assert cfg["Simulation"]["name"] == "demo"
    assert cfg["Simulation"]["mesh_nx"] == 11
    assert cfg["Materials"]["background"]["eps_r"] == 3.9
