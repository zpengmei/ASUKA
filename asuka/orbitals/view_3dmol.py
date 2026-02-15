from __future__ import annotations

"""Optional Jupyter visualization helpers.

This module is intentionally lightweight and keeps optional dependencies behind
function scope so that importing ASUKA remains lightweight.
"""


def view_cube(
    cube_path: str,
    *,
    iso: float = 0.03,
    opacity: float = 0.75,
    color_pos: str = "blue",
    color_neg: str = "red",
    width: int = 600,
    height: int = 450,
):
    """Return a py3Dmol view with ±isosurfaces for a cube file."""

    try:
        import py3Dmol  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("py3Dmol is required for view_cube (pip install py3Dmol)") from exc

    with open(cube_path, "r", encoding="utf-8") as f:
        cube_data = f.read()

    v = py3Dmol.view(width=int(width), height=int(height))
    v.addVolumetricData(cube_data, "cube", {"isoval": float(iso), "color": str(color_pos), "opacity": float(opacity)})
    v.addVolumetricData(cube_data, "cube", {"isoval": -float(iso), "color": str(color_neg), "opacity": float(opacity)})
    v.zoomTo()
    return v


__all__ = ["view_cube"]

