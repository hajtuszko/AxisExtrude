# Axis Extrude

Click-to-extrude modeling for Blender, made for architectural plans, elevations and sections.
Work in a top, front or side view: hover an edge, click, and it extrudes to the cursor along the world axis.
No hotkey juggling, no switching between extrude, grab and loop cut.

## Features

- Works in top (XY), front (XZ) and side (YZ) views
- Hover to highlight the nearest edge, click to extrude it to the cursor
- Click inside a face to add a loop cut
- Drag a vertex to move it along its axis in 10 cm steps, or click a vertex to pick it up and click again to drop it
- Drag the square handle at an edge midpoint to move the whole edge
- Snaps to nearby vertices and aligns with vertex lines (dashed guides)
- `X` locks or unlocks the selection
- Right-mouse drag up, down, left or right switches the view; middle-mouse double-click on a wall faces it
- Overlapping elements: act on the nearest one only, or on all of them
- Optional minimum extrude of 10 cm (snapping to a vertex ignores it)
- On disable, vertices closer than 0.01 cm are merged

## Requirements

Blender 4.2 or newer. Developed and tested on Blender 5.0.

## Installation

1. Clone or download this repository.
2. Build the extension package:
   ```
   blender --command extension build --source-dir axis_extrude
   ```
   This creates `axis_extrude-0.3.0.zip`.
3. In Blender: **Edit > Preferences > Get Extensions > Install from Disk…** and pick the zip.

## Usage

1. Select a mesh object in the 3D Viewport.
2. Open the sidebar (`N`) and go to the **Axis Extrude** tab.
3. Click **Enable** and switch to a top, front or side view.
4. Hover and click. Right-click disables the tool.

## License

GPL-2.0-or-later. See [LICENSE](LICENSE).

## Author

Patryk Malarczyk, Technical Artist and co-founder of Orion Belt Games.
