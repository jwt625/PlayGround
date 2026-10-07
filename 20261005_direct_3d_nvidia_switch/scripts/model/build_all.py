"""Build the full model (run inside Blender, or exec'd by render_views.py --build / fit_params.py).

Groups come from config/model/groups.toml (order = build order); env MODEL_GROUPS="case,crt" limits the set.
Each group module scripts/model/<group>/build.py must define build(P: dict, coll) -> None.
A group that fails to build is logged and replaced by its last-known-good copy (outputs/lkg/<group>/, refreshed
after every successful build), so one agent's half-saved edit does not remove it from the others' renders.
Note: the group's own builder sees its LKG build.py, but helper modules it imports from its folder resolve to
the LKG copies only if imported relative to that folder.
"""

import importlib.util
import os
import sys
import tomllib
import traceback
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "model"))
import lib  # noqa: E402


LKG = ROOT / "outputs" / "lkg"  # last-known-good copy per group: build.py, params TOML, extra files


def load_group_module(group: str, path: Path | None = None):
    path = path or ROOT / "scripts" / "model" / group / "build.py"
    spec = importlib.util.spec_from_file_location(f"group_{group}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save_lkg(group: str) -> None:
    import shutil
    d = LKG / group
    d.mkdir(parents=True, exist_ok=True)
    src = ROOT / "scripts" / "model" / group
    for f in src.glob("*"):
        if f.is_file() and f.suffix in {".py", ".json", ".toml"}:
            shutil.copy2(f, d / f.name)
    toml = ROOT / "config" / "model" / f"{group}.toml"
    if toml.exists():
        shutil.copy2(toml, d / "_params.toml")


def build_group(group: str, overrides: dict | None = None, use_lkg: bool = True) -> bool:
    """Build one group. On failure, fall back to its last-known-good copy (outputs/lkg/<group>/) so a
    half-saved edit by one agent does not remove that group from everyone else's renders."""
    coll = lib.group_collection(group)
    lib.clear_collection(coll)
    try:
        P = lib.load_params(group)  # inside try: a half-saved TOML also falls back to the last-known-good copy
        for k, v in (overrides or {}).items():
            d = P
            keys = k.split(".")
            for kk in keys[:-1]:
                d = d.setdefault(kk, {})
            d[keys[-1]] = v
        load_group_module(group).build(P, coll)
        if not overrides:
            _save_lkg(group)
        return True
    except Exception:
        print(f"BUILD_FAIL group={group}\n{traceback.format_exc()}")
        lib.clear_collection(coll)
    lkg = LKG / group / "build.py"
    if use_lkg and lkg.exists() and not overrides:
        try:
            sys.path.insert(0, str(lkg.parent))
            P2 = tomllib.loads((LKG / group / "_params.toml").read_text()) if (LKG / group / "_params.toml").exists() else {}
            load_group_module(group, lkg).build(P2, coll)
            print(f"BUILD_LKG group={group} (using last-known-good copy)")
            return True
        except Exception:
            print(f"BUILD_LKG_FAIL group={group}\n{traceback.format_exc()}")
            lib.clear_collection(coll)
        finally:
            sys.path.remove(str(lkg.parent))
    return False


def groups() -> list[str]:
    env = os.environ.get("MODEL_GROUPS")
    if env:
        return [g for g in env.split(",") if g]
    return tomllib.loads((ROOT / "config" / "model" / "groups.toml").read_text())["groups"]


def build_all() -> dict:
    for o in list(bpy.data.objects):
        if not o.name.startswith("_eval"):
            bpy.data.objects.remove(o, do_unlink=True)
    return {g: build_group(g) for g in groups()}


if __name__ in ("__main__", "__build__"):
    res = build_all()
    print("BUILD_RESULT", res)
