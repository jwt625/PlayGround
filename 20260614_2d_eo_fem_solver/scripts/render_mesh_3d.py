from __future__ import annotations

import argparse

from paraview.simple import (  # type: ignore[import-not-found]
    Clip,
    GetActiveViewOrCreate,
    OpenDataFile,
    SaveScreenshot,
    Show,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a clipped 3D mesh screenshot with ParaView.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--origin", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--normal", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    args = parser.parse_args()

    source = OpenDataFile(args.input)
    clipped = Clip(Input=source)
    clipped.ClipType = "Plane"
    clipped.ClipType.Origin = args.origin
    clipped.ClipType.Normal = args.normal

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [1600, 1000]
    view.Background = [0.035, 0.047, 0.063]
    display = Show(clipped, view)
    display.Representation = "Surface With Edges"
    display.DiffuseColor = [0.18, 0.62, 0.68]
    display.EdgeColor = [0.82, 0.88, 0.92]
    display.LineWidth = 0.65
    view.ResetCamera()
    camera = view.GetActiveCamera()
    camera.Azimuth(35)
    camera.Elevation(22)
    SaveScreenshot(args.output, view, ImageResolution=[1600, 1000])


if __name__ == "__main__":
    main()
