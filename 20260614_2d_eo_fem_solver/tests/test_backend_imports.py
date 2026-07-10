from __future__ import annotations

import pytest


def test_gmsh_python_api_imports_when_backend_extra_is_installed() -> None:
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        assert hasattr(gmsh, "model")
    finally:
        gmsh.finalize()


def test_mfem_python_api_imports_when_backend_extra_is_installed() -> None:
    mfem = pytest.importorskip("mfem.ser")
    assert hasattr(mfem, "Mesh")
    assert hasattr(mfem, "H1_FECollection")
    assert hasattr(mfem, "ND_FECollection")
