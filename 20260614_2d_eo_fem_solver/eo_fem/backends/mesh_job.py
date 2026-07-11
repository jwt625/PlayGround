from __future__ import annotations

import argparse
import json

from eo_fem.backends.gmsh_mesh import generate_legacy_gmsh_mesh
from eo_fem.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated Gmsh mesh generation job.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    artifact = generate_legacy_gmsh_mesh(load_config(args.config), args.out, model_name=args.name)
    print(
        json.dumps(
            {
                "mesh_path": str(artifact.mesh_path.resolve()),
                "manifest_path": str(artifact.tag_map_path.resolve()),
                "node_count": artifact.node_count,
                "element_count": artifact.element_count,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
