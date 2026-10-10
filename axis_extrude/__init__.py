bl_info = {
    "name": "Axis Extrude",
    "author": "Patryk M.",
    "version": (0, 2, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Axis Extrude",
    "description": "Click-to-extrude edges along world X/Y axes (2D top-view workflow).",
    "category": "Mesh",
}

import bpy
import bmesh
import blf
import time
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from bpy_extras import view3d_utils


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _get_state(context):
    return context.window_manager.axis_extrude_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mouse_to_world_xy(context, event):
    """Project mouse coords onto the world Z=0 plane and return a Vector."""
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None

    coord = (event.mouse_region_x, event.mouse_region_y)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    ray_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

    # Intersect with Z=0 plane
    if abs(ray_dir.z) < 1e-8:
        return None
    t = -ray_origin.z / ray_dir.z
    if t < 0:
        return None
    hit = ray_origin + ray_dir * t
    return Vector((hit.x, hit.y, 0.0))


def world_to_region_2d(context, world_point):
    """Project world_point to 2D region coords. Returns None if behind camera."""
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None
    return view3d_utils.location_3d_to_region_2d(region, rv3d, world_point)


def pixel_distance(context, world_a, world_b):
    """Return screen-space pixel distance between two world points, or inf."""
    a = world_to_region_2d(context, world_a)
    b = world_to_region_2d(context, world_b)
    if a is None or b is None:
        return float('inf')
    return (a - b).length


def cm_to_blender_units(context, cm):
    """Convert a distance in centimeters to Blender internal units,
    respecting scene.unit_settings.scale_length."""
    scene = context.scene
    scale = 1.0
    try:
        scale = scene.unit_settings.scale_length
        if scale <= 0:
            scale = 1.0
    except Exception:
        scale = 1.0
    # Blender internal units are meters (scaled by scale_length to get real meters).
    # So: real_meters = cm * 0.01; blender_units = real_meters / scale_length
    return (cm * 0.01) / scale


def blender_units_to_cm(context, units):
    """Inverse of cm_to_blender_units."""
    scene = context.scene
    scale = 1.0
    try:
        scale = scene.unit_settings.scale_length
        if scale <= 0:
            scale = 1.0
    except Exception:
        scale = 1.0
    return (units * scale) / 0.01


def snap_distance_to_step(delta_units, step_cm, context, min_cm=10.0):
    """
    Snap a signed distance (in Blender units) to the nearest multiple of
    step_cm, preserving sign. Enforces minimum of min_cm in the direction
    of motion. Returns (snapped_units, snapped_cm_magnitude).
    """
    cm_val = blender_units_to_cm(context, abs(delta_units))
    snapped_cm = round(cm_val / step_cm) * step_cm
    if snapped_cm < min_cm:
        snapped_cm = min_cm
    sign = 1.0 if delta_units >= 0 else -1.0
    if delta_units == 0:
        sign = 1.0
    return sign * cm_to_blender_units(context, snapped_cm), snapped_cm


def find_edge_candidates(obj, world_point):
    """
    Return a list of edge candidates sorted by XY distance to world_point.
    Each entry: dict(index, edge_axis, extrude_axis, dist).
      - edge_axis:    axis the edge runs along ('X' or 'Y')
      - extrude_axis: the OTHER one — where the new face will stretch
    """
    if obj is None or obj.type != 'MESH':
        return []

    me = obj.data
    mw = obj.matrix_world

    candidates = []

    for e in me.edges:
        v0 = mw @ me.vertices[e.vertices[0]].co
        v1 = mw @ me.vertices[e.vertices[1]].co

        # Work in XY only
        p0 = Vector((v0.x, v0.y))
        p1 = Vector((v1.x, v1.y))
        p  = Vector((world_point.x, world_point.y))

        seg = p1 - p0
        seg_len_sq = seg.length_squared
        if seg_len_sq < 1e-12:
            continue

        dx = abs(seg.x)
        dy = abs(seg.y)
        if dx >= dy:
            edge_axis = 'X'
        else:
            edge_axis = 'Y'

        t = (p - p0).dot(seg) / seg_len_sq
        t = max(0.0, min(1.0, t))
        closest = p0 + seg * t
        dist = (p - closest).length

        candidates.append({
            "index": e.index,
            "edge_axis": edge_axis,
            "extrude_axis": 'Y' if edge_axis == 'X' else 'X',
            "dist": dist,
        })

    candidates.sort(key=lambda c: c["dist"])
    return candidates


def find_nearest_vertex(obj, world_point, exclude_edge_index=None, max_dist=None,
                        exclude_vert_index=None):
    """
    Return (world_co, vert_index) for the vertex closest to world_point in XY.
    If `exclude_edge_index` is given, verts belonging to that edge are skipped
    (we don't want to snap to the edge we're extruding FROM).
    If `exclude_vert_index` is given, that vertex is skipped (the one being dragged).
    If `max_dist` is given, returns None when no vertex is within that radius.
    """
    if obj is None or obj.type != 'MESH':
        return None

    me = obj.data
    mw = obj.matrix_world

    excluded_verts = set()
    if exclude_edge_index is not None and 0 <= exclude_edge_index < len(me.edges):
        e = me.edges[exclude_edge_index]
        excluded_verts.update(e.vertices)
    if exclude_vert_index is not None:
        excluded_verts.add(exclude_vert_index)

    p = Vector((world_point.x, world_point.y))
    best = None
    for v in me.vertices:
        if v.index in excluded_verts:
            continue
        w = mw @ v.co
        d = (Vector((w.x, w.y)) - p).length
        if best is None or d < best[0]:
            best = (d, v.index, w)

    if best is None:
        return None
    if max_dist is not None and best[0] > max_dist:
        return None
    return best[2], best[1]


def determine_vertex_drag_axis(obj, vert_index, world_point):
    """
    Decide which axis the vertex should move along when being dragged.
    Axis is determined by the dominant delta between cursor and vertex:
      cursor further along X → drag along X
      cursor further along Y → drag along Y
    Returns 'X' or 'Y', or None if vertex not found.
    """
    me = obj.data
    mw = obj.matrix_world
    if vert_index >= len(me.vertices):
        return None
    v_co = mw @ me.vertices[vert_index].co
    dx = abs(world_point.x - v_co.x)
    dy = abs(world_point.y - v_co.y)
    if dx == 0 and dy == 0:
        return 'X'  # arbitrary fallback, user hasn't moved yet
    return 'X' if dx >= dy else 'Y'


def move_vertex_along_axis(obj, vert_index, axis, new_coord):
    """Set vertex[vert_index]'s world coordinate on `axis` to new_coord,
    leaving the other axes untouched."""
    me = obj.data
    mw = obj.matrix_world
    mw_inv = mw.inverted()
    if vert_index >= len(me.vertices):
        return False
    v = me.vertices[vert_index]
    w = mw @ v.co
    if axis == 'X':
        w.x = new_coord
    else:
        w.y = new_coord
    v.co = mw_inv @ w
    me.update()
    return True


def move_vertex_xy(obj, vert_index, new_x, new_y):
    """Set the vertex's world X and Y coordinates simultaneously."""
    me = obj.data
    mw = obj.matrix_world
    mw_inv = mw.inverted()
    if vert_index >= len(me.vertices):
        return False
    v = me.vertices[vert_index]
    w = mw @ v.co
    w.x = new_x
    w.y = new_y
    v.co = mw_inv @ w
    me.update()
    return True


def move_verts_world(obj, moves):
    """Set world positions for several vertices at once.
    `moves` is a list of (vert_index, world_co)."""
    me = obj.data
    mw_inv = obj.matrix_world.inverted()
    n = len(me.vertices)
    for vi, world_co in moves:
        if vi < n:
            me.vertices[vi].co = mw_inv @ world_co
    me.update()


def edge_midpoint_world(obj, edge_index):
    me = obj.data
    if edge_index >= len(me.edges):
        return None
    mw = obj.matrix_world
    e = me.edges[edge_index]
    return (mw @ me.vertices[e.vertices[0]].co + mw @ me.vertices[e.vertices[1]].co) * 0.5


def pixels_per_unit(context, world_point):
    """Screen pixels per 1 world unit (in XY) around world_point, or None."""
    a = world_to_region_2d(context, world_point)
    b = world_to_region_2d(context, world_point + Vector((1.0, 0.0, 0.0)))
    if a is None or b is None:
        return None
    ppu = (b - a).length
    return ppu if ppu > 1e-9 else None


POINT_SNAP_PIXELS = 20.0   # cursor this close to a vertex → snap to it
LINE_SNAP_PIXELS = 10.0    # cursor this close to the X/Y line through a vertex → align


def snap_axis_coord(context, obj, target, axis, exclude_verts=()):
    """
    Snap the `axis` ('X' or 'Y') coordinate of `target`:
      1. point snap — a vertex within POINT_SNAP_PIXELS of the cursor,
      2. line snap  — the cursor within LINE_SNAP_PIXELS of the axis-aligned
                      line through any vertex (alignment with distant geometry).
    Returns (coord, ref_vertex_world, kind) with kind 'point' | 'line', or None.
    """
    if obj is None or obj.type != 'MESH':
        return None
    ppu = pixels_per_unit(context, target)
    if ppu is None:
        return None
    cursor_px = world_to_region_2d(context, target)
    if cursor_px is None:
        return None

    me = obj.data
    mw = obj.matrix_world
    excluded = set(exclude_verts)
    cur_c = target.x if axis == 'X' else target.y

    best_point = None  # (px, coord, world)
    best_line = None   # (px, along_dist, coord, world)
    for v in me.vertices:
        if v.index in excluded:
            continue
        w = mw @ v.co
        c = w.x if axis == 'X' else w.y
        line_px = abs(c - cur_c) * ppu
        # Distance to the point is never smaller than to the line through it
        if line_px > POINT_SNAP_PIXELS:
            continue
        v_px = world_to_region_2d(context, w)
        if v_px is not None:
            d = (v_px - cursor_px).length
            if d <= POINT_SNAP_PIXELS and (best_point is None or d < best_point[0]):
                best_point = (d, c, w)
        if line_px <= LINE_SNAP_PIXELS:
            # Among vertices on (nearly) the same line, reference the closest one
            along = abs((w.y - target.y) if axis == 'X' else (w.x - target.x))
            key = (round(line_px, 2), along)
            if best_line is None or key < (round(best_line[0], 2), best_line[1]):
                best_line = (line_px, along, c, w)

    if best_point is not None:
        return best_point[1], best_point[2], 'point'
    if best_line is not None:
        return best_line[2], best_line[3], 'line'
    return None


def extrude_edge_to_point(obj, edge_index, extrude_axis, target_world,
                          min_distance=0.0):
    """
    Extrude the given edge and snap the new verts so the extruded quad extends
    to `target_world` along `extrude_axis` (world axis). Other axis is locked
    to the original edge's value (snap-to-world).

    If the signed distance from the original edge to target_world along
    extrude_axis is smaller than min_distance, the target is clamped so the
    extrusion equals min_distance in the same direction the user was heading.
    """
    me = obj.data
    mw = obj.matrix_world
    mw_inv = mw.inverted()

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    if edge_index >= len(bm.edges):
        bm.free()
        return False

    edge = bm.edges[edge_index]
    if not edge.is_valid:
        bm.free()
        return False

    # Original edge endpoints in world space
    v0_world = mw @ edge.verts[0].co
    v1_world = mw @ edge.verts[1].co

    # Enforce minimum distance along extrude axis
    orig_coord = v0_world.x if extrude_axis == 'X' else v0_world.y
    target_coord = target_world.x if extrude_axis == 'X' else target_world.y
    delta = target_coord - orig_coord
    if min_distance > 0.0 and abs(delta) < min_distance:
        sign = 1.0 if delta >= 0 else -1.0
        # If user hasn't moved at all (delta == 0), pick +direction by default
        if delta == 0:
            sign = 1.0
        target_coord = orig_coord + sign * min_distance
        clamped_target = Vector(target_world)
        if extrude_axis == 'X':
            clamped_target.x = target_coord
        else:
            clamped_target.y = target_coord
        target_world = clamped_target

    # Extrude the edge
    ret = bmesh.ops.extrude_edge_only(bm, edges=[edge])
    new_geom = ret['geom']
    new_verts = [g for g in new_geom if isinstance(g, bmesh.types.BMVert)]

    if len(new_verts) != 2:
        bm.free()
        return False

    # Match new verts to original verts by proximity (extrude_edge_only puts
    # them in the same position as the originals initially)
    # Build pairing: new_vert -> original world co
    pairs = []
    for nv in new_verts:
        nv_world = mw @ nv.co
        if (nv_world - v0_world).length < (nv_world - v1_world).length:
            pairs.append((nv, v0_world))
        else:
            pairs.append((nv, v1_world))

    # Move new verts: keep perpendicular axis locked, set extrude axis to target
    for nv, orig_world in pairs:
        new_world = Vector(orig_world)
        if extrude_axis == 'X':
            new_world.x = target_world.x
        elif extrude_axis == 'Y':
            new_world.y = target_world.y
        # Z stays at orig_world.z (which is 0 for 2D plans)
        nv.co = mw_inv @ new_world

    bm.to_mesh(me)
    bm.free()
    me.update()
    return True


def find_face_at_point(obj, world_point):
    """
    Return (face_index, face_axis) for a face containing world_point in XY,
    or None if no face contains the point.
      - face_axis: 'X' or 'Y' — the axis along which the face is longer
                   (so loop-cut will be perpendicular to this axis)
    """
    if obj is None or obj.type != 'MESH':
        return None

    me = obj.data
    mw = obj.matrix_world
    p = Vector((world_point.x, world_point.y))

    best = None  # (area, face_index, face_axis)
    for f in me.polygons:
        if len(f.vertices) < 3:
            continue

        # Gather face vertices in XY
        pts = [mw @ me.vertices[vi].co for vi in f.vertices]
        pts_xy = [Vector((v.x, v.y)) for v in pts]

        # Point-in-polygon (ray casting)
        inside = False
        n = len(pts_xy)
        j = n - 1
        for i in range(n):
            xi, yi = pts_xy[i].x, pts_xy[i].y
            xj, yj = pts_xy[j].x, pts_xy[j].y
            if ((yi > p.y) != (yj > p.y)):
                x_int = (xj - xi) * (p.y - yi) / (yj - yi + 1e-30) + xi
                if p.x < x_int:
                    inside = not inside
            j = i
        if not inside:
            continue

        # Determine face axis from bbox: longer side = face_axis
        xs = [v.x for v in pts_xy]
        ys = [v.y for v in pts_xy]
        size_x = max(xs) - min(xs)
        size_y = max(ys) - min(ys)
        face_axis = 'X' if size_x >= size_y else 'Y'

        # Prefer smallest face if overlapping (approximate via bbox area)
        area = size_x * size_y
        if best is None or area < best[0]:
            best = (area, f.index, face_axis)

    if best is None:
        return None
    return best[1], best[2]


def _determine_cut_axis_from_face(obj, face_index, world_point):
    """
    Pick the cut axis by looking at the face edge closest to world_point.
    Ctrl+R-style: the cut runs PARALLEL to that edge, so the cut axis (where
    cut_coord is measured) is the axis PERPENDICULAR to that edge.

    Returns (cut_axis, cut_coord) or None.
      - cut_axis: 'X' or 'Y' (axis on which cut_coord is measured;
                  cut line itself runs along the other axis)
    """
    me = obj.data
    mw = obj.matrix_world
    if face_index >= len(me.polygons):
        return None
    face = me.polygons[face_index]

    p = Vector((world_point.x, world_point.y))
    best = None  # (dist, edge_axis)
    for ek in face.edge_keys:
        v0 = mw @ me.vertices[ek[0]].co
        v1 = mw @ me.vertices[ek[1]].co
        a = Vector((v0.x, v0.y))
        b = Vector((v1.x, v1.y))
        seg = b - a
        L2 = seg.length_squared
        if L2 < 1e-12:
            continue
        t = max(0.0, min(1.0, (p - a).dot(seg) / L2))
        closest = a + seg * t
        d = (p - closest).length
        # Classify the edge's dominant direction
        if abs(seg.x) >= abs(seg.y):
            edge_axis = 'X'  # edge runs along X, so cut is perpendicular → cut_axis = X
        else:
            edge_axis = 'Y'
        if best is None or d < best[0]:
            best = (d, edge_axis)

    if best is None:
        return None
    cut_axis = best[1]
    cut_coord = world_point.x if cut_axis == 'X' else world_point.y
    return cut_axis, cut_coord


def _collect_loop_cut_ring(bm, start_face, cut_axis, cut_coord, mw):
    """
    BFS from start_face across faces whose boundary straddles cut_coord on
    cut_axis. A face is included if it has exactly 2 straddling edges
    (standard loop-cut condition); we hop to neighbors through those edges.
    Returns (list_of_affected_faces, set_of_straddling_edges).
    """
    def edge_straddles(e):
        v0 = mw @ e.verts[0].co
        v1 = mw @ e.verts[1].co
        a = v0.x if cut_axis == 'X' else v0.y
        b = v1.x if cut_axis == 'X' else v1.y
        return (a - cut_coord) * (b - cut_coord) < -1e-9

    if not start_face.is_valid:
        return [], set()
    start_straddling = [e for e in start_face.edges if edge_straddles(e)]
    if len(start_straddling) < 2:
        return [], set()

    visited = {start_face.index}
    affected = [start_face]
    edges_set = set(start_straddling)
    queue = list(start_straddling)

    while queue:
        e = queue.pop(0)
        for nf in e.link_faces:
            if nf.index in visited:
                continue
            nf_straddling = [ee for ee in nf.edges if edge_straddles(ee)]
            if len(nf_straddling) < 2:
                # Non-quad-like face in the ring — skip this neighbor but
                # continue from other neighbors
                continue
            visited.add(nf.index)
            affected.append(nf)
            for se in nf_straddling:
                if se not in edges_set:
                    edges_set.add(se)
                    queue.append(se)

    return affected, edges_set


def loop_cut_face_at_point(obj, face_index, world_point, forced_cut_axis=None):
    """
    Ctrl+R-style loop cut. Axis is determined by the face edge closest to
    world_point; cut propagates through the ring of faces.

    If forced_cut_axis ('X' or 'Y') is given, use that instead of the
    auto-detected axis (used by locked loopcut mode).
    """
    me = obj.data
    mw = obj.matrix_world
    mw_inv = mw.inverted()

    if forced_cut_axis in ('X', 'Y'):
        cut_axis = forced_cut_axis
        cut_coord = world_point.x if cut_axis == 'X' else world_point.y
    else:
        axis_info = _determine_cut_axis_from_face(obj, face_index, world_point)
        if axis_info is None:
            return False
        cut_axis, cut_coord = axis_info

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    if face_index >= len(bm.faces) or not bm.faces[face_index].is_valid:
        bm.free()
        return False
    start = bm.faces[face_index]

    affected, edges_to_cut = _collect_loop_cut_ring(bm, start, cut_axis, cut_coord, mw)
    if not affected or not edges_to_cut:
        bm.free()
        return False

    new_verts = []
    for e in list(edges_to_cut):
        v0 = mw @ e.verts[0].co
        v1 = mw @ e.verts[1].co
        a = v0.x if cut_axis == 'X' else v0.y
        b = v1.x if cut_axis == 'X' else v1.y
        if abs(b - a) < 1e-12:
            continue
        t = (cut_coord - a) / (b - a)
        t = max(0.0, min(1.0, t))
        try:
            _ne, nv = bmesh.utils.edge_split(e, e.verts[0], t)
        except Exception:
            continue
        nw = mw @ nv.co
        if cut_axis == 'X':
            nw.x = cut_coord
        else:
            nw.y = cut_coord
        nv.co = mw_inv @ nw
        new_verts.append(nv)

    valid_new = [v for v in new_verts if v.is_valid]
    if len(valid_new) >= 2:
        try:
            bmesh.ops.connect_verts(bm, verts=valid_new)
        except Exception:
            pass

    bm.to_mesh(me)
    bm.free()
    me.update()
    return True




# ---------------------------------------------------------------------------
# Drawing (preview line)
# ---------------------------------------------------------------------------

_draw_handle = {"handle": None, "text_handle": None}


def _draw_callback(op, context):
    # Preview lines (edge extrude preview)
    if op._preview_points:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": op._preview_points})
        gpu.state.line_width_set(2.0)
        gpu.state.blend_set('ALPHA')
        shader.bind()
        shader.uniform_float("color", (0.1, 0.8, 1.0, 0.9))
        batch.draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')

    # Snap indicator — small cross at the snapped vertex
    snap_pt = getattr(op, "_preview_snap_point", None)
    if snap_pt is not None:
        region = context.region
        rv3d = context.region_data
        if region and rv3d:
            scr = view3d_utils.location_3d_to_region_2d(region, rv3d, snap_pt)
            if scr is not None:
                # Build a small world-space cross around the snap point.
                # Use screen-aligned axes by projecting pixel offsets back.
                SIZE_PX = 8
                def unproj(px, py):
                    origin = view3d_utils.region_2d_to_origin_3d(
                        region, rv3d, (scr.x + px, scr.y + py))
                    direction = view3d_utils.region_2d_to_vector_3d(
                        region, rv3d, (scr.x + px, scr.y + py))
                    if abs(direction.z) < 1e-8:
                        return Vector((snap_pt.x, snap_pt.y, 0))
                    t = -origin.z / direction.z
                    hit = origin + direction * t
                    return Vector((hit.x, hit.y, 0))

                a = unproj(-SIZE_PX, 0)
                b = unproj(SIZE_PX, 0)
                c = unproj(0, -SIZE_PX)
                d = unproj(0, SIZE_PX)

                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                batch = batch_for_shader(shader, 'LINES', {"pos": [a, b, c, d]})
                gpu.state.line_width_set(2.0)
                gpu.state.blend_set('ALPHA')
                shader.bind()
                shader.uniform_float("color", (1.0, 0.6, 0.1, 1.0))
                batch.draw(shader)
                gpu.state.line_width_set(1.0)
                gpu.state.blend_set('NONE')

    # Hover vertex highlight (cyan circle) — only when not dragging/locked
    hover = getattr(op, "_hover_vertex", None)
    dragging = getattr(op, "_drag", None)
    if hover is not None and dragging is None:
        region = context.region
        rv3d = context.region_data
        if region and rv3d:
            vw, _vi = hover
            scr = view3d_utils.location_3d_to_region_2d(region, rv3d, vw)
            if scr is not None:
                # Ring of ~16 points around the hover vertex
                import math
                R_PX = 9
                verts_ring = []
                def _unp(px, py, anchor):
                    origin = view3d_utils.region_2d_to_origin_3d(
                        region, rv3d, (scr.x + px, scr.y + py))
                    direction = view3d_utils.region_2d_to_vector_3d(
                        region, rv3d, (scr.x + px, scr.y + py))
                    if abs(direction.z) < 1e-8:
                        return Vector((anchor.x, anchor.y, 0))
                    t = -origin.z / direction.z
                    hit = origin + direction * t
                    return Vector((hit.x, hit.y, 0))

                segments = 24
                pts = []
                for i in range(segments):
                    a1 = (i / segments) * 2.0 * math.pi
                    a2 = ((i + 1) / segments) * 2.0 * math.pi
                    pts.append(_unp(R_PX * math.cos(a1), R_PX * math.sin(a1), vw))
                    pts.append(_unp(R_PX * math.cos(a2), R_PX * math.sin(a2), vw))

                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                batch = batch_for_shader(shader, 'LINES', {"pos": pts})
                gpu.state.line_width_set(2.0)
                gpu.state.blend_set('ALPHA')
                shader.bind()
                shader.uniform_float("color", (0.2, 1.0, 0.6, 1.0))
                batch.draw(shader)
                gpu.state.line_width_set(1.0)
                gpu.state.blend_set('NONE')

    # Hover edge highlight — the whole edge, when its midpoint can be grabbed
    hover_edge = getattr(op, "_hover_edge", None)
    if hover_edge is not None and dragging is None:
        obj = context.active_object
        if obj is not None and obj.type == 'MESH' and hover_edge[0] < len(obj.data.edges):
            mw = obj.matrix_world
            e = obj.data.edges[hover_edge[0]]
            a = mw @ obj.data.vertices[e.vertices[0]].co
            b = mw @ obj.data.vertices[e.vertices[1]].co
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            batch = batch_for_shader(shader, 'LINES', {"pos": [a, b]})
            gpu.state.line_width_set(4.0)
            gpu.state.blend_set('ALPHA')
            shader.bind()
            shader.uniform_float("color", (1.0, 0.85, 0.2, 0.9))
            batch.draw(shader)
            gpu.state.line_width_set(1.0)
            gpu.state.blend_set('NONE')

    # Dragging indicator — axis guide line through the moved vertex / edge midpoint
    if dragging is not None and dragging.get("axis") is not None:
        vw = dragging["current_anchor"]
        axis = dragging["axis"]
        LEN = 1000.0
        if axis == 'X':
            a = Vector((vw.x - LEN, vw.y, 0))
            b = Vector((vw.x + LEN, vw.y, 0))
        else:
            a = Vector((vw.x, vw.y - LEN, 0))
            b = Vector((vw.x, vw.y + LEN, 0))
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": [a, b]})
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('ALPHA')
        shader.bind()
        # Red for X axis, green for Y axis (Blender convention)
        if axis == 'X':
            shader.uniform_float("color", (1.0, 0.2, 0.2, 0.5))
        else:
            shader.uniform_float("color", (0.2, 1.0, 0.2, 0.5))
        batch.draw(shader)
        gpu.state.blend_set('NONE')


def _draw_2d_lines(points, color, width=1.0):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": [(p.x, p.y, 0.0) for p in points]})
    gpu.state.line_width_set(width)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_text_callback(op, context):
    """Pixel-space overlay: alignment line, edge midpoint handle, dimension label."""
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return

    # Alignment (line snap) guide — dashed line from the reference vertex.
    # Only shown together with the snap cross, so clearing the snap point hides it.
    snap_line = getattr(op, "_preview_snap_line", None)
    if snap_line is not None and getattr(op, "_preview_snap_point", None) is not None:
        a = view3d_utils.location_3d_to_region_2d(region, rv3d, snap_line[0])
        b = view3d_utils.location_3d_to_region_2d(region, rv3d, snap_line[1])
        if a is not None and b is not None:
            seg = b - a
            length = seg.length
            if length > 1.0:
                DASH, GAP = 6.0, 4.0
                direction = seg / length
                pts = []
                s = 0.0
                while s < length:
                    pts.append(a + direction * s)
                    pts.append(a + direction * min(s + DASH, length))
                    s += DASH + GAP
                _draw_2d_lines(pts, (1.0, 0.3, 0.9, 0.9), 1.5)

    # Edge midpoint handle — faint hint near an edge, bright when grabbable
    handle = getattr(op, "_edge_handle", None)
    if handle is not None and getattr(op, "_drag", None) is None:
        mid_world, active = handle
        m = view3d_utils.location_3d_to_region_2d(region, rv3d, mid_world)
        if m is not None:
            R = 6.0 if active else 4.0
            c0 = Vector((m.x - R, m.y - R))
            c1 = Vector((m.x + R, m.y - R))
            c2 = Vector((m.x + R, m.y + R))
            c3 = Vector((m.x - R, m.y + R))
            if active:
                color = (1.0, 0.85, 0.2, 1.0)
            else:
                color = (1.0, 1.0, 1.0, 0.45)
            _draw_2d_lines([c0, c1, c1, c2, c2, c3, c3, c0], color, 2.0)

    dim = getattr(op, "_dim_info", None)
    if not dim:
        return
    orig_mid, new_mid, cm_value, snapped = dim[:4]
    # Optional 5th flag: draw a single label at the segment midpoint (vertex move)
    single_label = len(dim) > 4 and dim[4]
    a = view3d_utils.location_3d_to_region_2d(region, rv3d, orig_mid)
    b = view3d_utils.location_3d_to_region_2d(region, rv3d, new_mid)
    if a is None or b is None:
        return

    # Format value: show whole cm if close to integer, else 1 decimal
    if abs(cm_value - round(cm_value)) < 0.05:
        text = f"{int(round(cm_value))} cm"
    else:
        text = f"{cm_value:.1f} cm"
    if snapped:
        text = "[" + text + "]"

    font_id = 0
    blf.size(font_id, 14)
    # Draw at both midpoints with a small offset so text doesn't sit on the line
    OFFSET_Y = 10
    positions = ((a + b) * 0.5,) if single_label else (a, b)
    for pos in positions:
        label = text
        x = pos.x + 6
        y = pos.y + OFFSET_Y
        # Shadow for readability
        blf.color(font_id, 0.0, 0.0, 0.0, 0.85)
        blf.position(font_id, x + 1, y - 1, 0)
        blf.draw(font_id, label)
        # Foreground (orange when snapped, cyan otherwise)
        if snapped:
            blf.color(font_id, 1.0, 0.75, 0.15, 1.0)
        else:
            blf.color(font_id, 0.4, 1.0, 1.0, 1.0)
        blf.position(font_id, x, y, 0)
        blf.draw(font_id, label)


# ---------------------------------------------------------------------------
# Modal operator
# ---------------------------------------------------------------------------

class AXISEXTRUDE_OT_modal(bpy.types.Operator):
    bl_idname = "axis_extrude.modal"
    bl_label = "Axis Extrude (Modal)"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return (context.area and context.area.type == 'VIEW_3D'
                and context.active_object is not None
                and context.active_object.type == 'MESH')

    def invoke(self, context, event):
        state = _get_state(context)
        if state.running:
            self.report({'WARNING'}, "Axis Extrude already running")
            return {'CANCELLED'}

        # Ensure object mode so bmesh ops on obj.data don't fight with edit-mode bmesh
        if context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        self._preview_points = []
        self._preview_snap_point = None  # world pos of snapped vertex (for visual)
        self._preview_snap_line = None   # (ref_vertex_world, aligned_point_world) for line snap
        self._cycle_offset = 0           # which candidate from sorted list
        self._last_mouse_world = None    # last XY target under cursor
        self._last_candidates = []       # cached candidate list
        self._edge_locked = False        # X toggles this — freezes edge selection
        self._locked_candidate = None    # candidate dict that is locked
        self._locked_mode = None         # None | 'edge' | 'loopcut'
        self._locked_face_index = None   # for loopcut lock
        self._locked_long_axis = None    # 'X' or 'Y' — long axis of locked face
        # Vertex / edge move
        self._hover_vertex = None        # (world_pos, vert_index) if a click would grab a vertex
        self._hover_edge = None          # (edge_index, mid_world, move_axis) if a click would grab an edge
        self._edge_handle = None         # (mid_world, active) — midpoint handle to draw
        self._drag = None                # dict describing the vertex/edge being moved, see _start_drag
        # Hold-snap mode — while LMB is held, extrusion snaps to multiples of 10 cm
        self._lmb_press_time = 0.0       # time.time() when PRESS fired
        self._lmb_press_pos = None       # world Vector when PRESS fired
        self._lmb_active_extrude = False # True while an extrude PRESS is in progress
        self._hold_candidate = None      # frozen candidate at press time
        self._hold_snapped_cm = 0.0      # currently shown snap distance in cm
        # Dimension label & hold-snap
        self._dim_info = None            # (orig_mid_world, new_mid_world, cm_value, snapped)
        self._lmb_down_time = None       # time when LMB pressed (for hold detection)
        self._lmb_hold_active = False    # True if current LMB is being held (snap mode)

        args = (self, context)
        _draw_handle["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, args, 'WINDOW', 'POST_VIEW'
        )
        _draw_handle["text_handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_text_callback, args, 'WINDOW', 'POST_PIXEL'
        )

        state.running = True
        self._timer = context.window_manager.event_timer_add(
            0.05, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _cancel(self, context):
        state = _get_state(context)
        state.running = False

        # Remove timer
        if getattr(self, "_timer", None) is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

        # Merge vertices by distance (0.01 cm) on the active mesh
        obj = context.active_object
        if obj is not None and obj.type == 'MESH':
            merge_dist = cm_to_blender_units(context, 0.01)
            try:
                bm = bmesh.new()
                bm.from_mesh(obj.data)
                bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
                bm.to_mesh(obj.data)
                bm.free()
                obj.data.update()
                bpy.ops.ed.undo_push(message="Axis Extrude: Merge By Distance")
            except Exception:
                pass

        if _draw_handle["handle"] is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(_draw_handle["handle"], 'WINDOW')
            except Exception:
                pass
            _draw_handle["handle"] = None
        if _draw_handle.get("text_handle") is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(_draw_handle["text_handle"], 'WINDOW')
            except Exception:
                pass
            _draw_handle["text_handle"] = None
        if context.area:
            context.area.tag_redraw()

    # ---- vertex / edge move -----------------------------------------------

    VERTEX_GRAB_PIXELS = 12.0   # cursor this close to a vertex → click grabs it
    EDGE_GRAB_PIXELS = 10.0     # cursor this close to an edge midpoint → click grabs the edge
    EDGE_HINT_PIXELS = 40.0     # cursor this close to an edge → show its midpoint handle
    DRAG_START_PIXELS = 5.0     # dead zone before a press turns into a drag

    def _find_grab_target(self, context, obj, target, candidates=None):
        """
        What a click at `target` would grab. Vertex and edge midpoint compete;
        the one nearer to the cursor on screen wins.
        Returns ('vertex', (world, index)) | ('edge', (edge_index, mid_world, move_axis)) | None.
        Also refreshes the midpoint handle hint (self._edge_handle).
        """
        self._edge_handle = None
        best = None  # (px, kind, payload)

        vert_hit = find_nearest_vertex(obj, target)
        if vert_hit is not None:
            d = pixel_distance(context, vert_hit[0], target)
            if d <= self.VERTEX_GRAB_PIXELS:
                best = (d, 'vertex', vert_hit)

        if candidates is None:
            candidates = find_edge_candidates(obj, target)
        if candidates:
            c = candidates[0]
            mid = edge_midpoint_world(obj, c["index"])
            ppu = pixels_per_unit(context, target)
            if mid is not None and ppu is not None and c["dist"] * ppu <= self.EDGE_HINT_PIXELS:
                d = pixel_distance(context, mid, target)
                grab = d <= self.EDGE_GRAB_PIXELS and (best is None or d < best[0])
                if grab:
                    best = (d, 'edge', (c["index"], mid, c["extrude_axis"]))
                self._edge_handle = (mid, grab)

        if best is None:
            return None
        return best[1], best[2]

    def _update_hover(self, context, obj, target):
        self._hover_vertex = None
        self._hover_edge = None
        self._edge_handle = None
        if self._edge_locked or self._lmb_active_extrude:
            return
        hit = self._find_grab_target(context, obj, target, self._last_candidates)
        if hit is None:
            return
        kind, payload = hit
        if kind == 'vertex':
            self._hover_vertex = payload
        else:
            self._hover_edge = payload

    def _start_drag(self, context, event, kind, payload):
        """
        Begin moving a vertex or an edge.
          kind 'vertex': payload (world, vert_index) — axis follows cursor dominance
          kind 'edge':   payload (edge_index, mid_world, move_axis) — axis fixed,
                         perpendicular to the edge (parallel move)
        """
        obj = context.active_object
        mw = obj.matrix_world
        me = obj.data
        if kind == 'vertex':
            vw, vi = payload
            vert_ids = [vi]
            anchor = Vector(vw)
            fixed_axis = None
        else:
            ei, mid, move_axis = payload
            vert_ids = list(me.edges[ei].vertices)
            anchor = Vector(mid)
            fixed_axis = move_axis
        self._drag = {
            "kind": kind,
            "verts": [(vi, mw @ me.vertices[vi].co) for vi in vert_ids],
            "anchor": anchor,                 # original vertex pos / edge midpoint
            "current_anchor": Vector(anchor),
            "fixed_axis": fixed_axis,
            "axis": fixed_axis,
            "mode": 'press',                  # 'press' (LMB held) | 'pickup' (follows cursor)
            "press_px": Vector((event.mouse_region_x, event.mouse_region_y)),
            "started": False,                 # True once cursor left the dead zone
        }
        self._hover_vertex = None
        self._hover_edge = None
        self._edge_handle = None
        self._preview_points = []
        self._preview_snap_point = None
        self._dim_info = None

    def _set_snap_preview(self, snap, moved_points):
        """Show the snap cross (and dashed alignment line for line snaps)."""
        if snap is None:
            self._preview_snap_point = None
            self._preview_snap_line = None
            return
        _coord, ref, kind = snap
        self._preview_snap_point = ref
        if kind == 'line' and moved_points:
            end = min(moved_points, key=lambda p: (p - ref).length)
            self._preview_snap_line = (ref, end)
        else:
            self._preview_snap_line = None

    def _update_drag(self, context, obj, event, target):
        """
        Move the dragged vertex/edge along its axis — same rules as edge extrude:
          - snap to a vertex / alignment with a vertex line wins,
          - LMB held (drag) → 10 cm steps,
          - picked up by a quick click → free movement.
        Vertex axis is re-evaluated from the ORIGINAL position every frame, so
        the user switches axes just by moving the mouse the other way.
        """
        d = self._drag
        anchor = d["anchor"]

        if d["mode"] == 'press' and not d["started"]:
            mouse_px = Vector((event.mouse_region_x, event.mouse_region_y))
            if (mouse_px - d["press_px"]).length < self.DRAG_START_PIXELS:
                return
            d["started"] = True

        axis = d["fixed_axis"]
        if axis is None:
            axis = 'X' if abs(target.x - anchor.x) >= abs(target.y - anchor.y) else 'Y'
        d["axis"] = axis
        orig_c = anchor.x if axis == 'X' else anchor.y
        raw_c = target.x if axis == 'X' else target.y

        step_snapped = False
        snap = snap_axis_coord(context, obj, target, axis,
                               exclude_verts=[vi for vi, _w in d["verts"]])
        if snap is not None:
            new_c = snap[0]
        elif d["mode"] == 'press':
            snapped_delta, _cm = snap_distance_to_step(
                raw_c - orig_c, 10.0, context, min_cm=0.0
            )
            new_c = orig_c + snapped_delta
            step_snapped = True
        else:
            new_c = raw_c

        if axis == 'X':
            offset = Vector((new_c - orig_c, 0.0, 0.0))
        else:
            offset = Vector((0.0, new_c - orig_c, 0.0))
        moves = [(vi, w + offset) for vi, w in d["verts"]]
        move_verts_world(obj, moves)

        new_anchor = anchor + offset
        d["current_anchor"] = new_anchor
        self._preview_points = [anchor, new_anchor]
        self._set_snap_preview(snap, [w for _vi, w in moves])
        cm_value = blender_units_to_cm(context, abs(new_c - orig_c))
        self._dim_info = (anchor, new_anchor, cm_value, step_snapped, True)

    def _end_drag(self, context, restore):
        if self._drag is not None and restore:
            move_verts_world(context.active_object, self._drag["verts"])
        self._drag = None
        self._hover_vertex = None
        self._hover_edge = None
        self._edge_handle = None
        self._preview_points = []
        self._preview_snap_point = None
        self._dim_info = None
        if context.area:
            context.area.tag_redraw()

    # ---- helpers used by modal ---------------------------------------------

    def _pick_candidate(self):
        """Return the currently selected candidate.
        If locked, return the frozen one regardless of cursor position.
        Otherwise return the offset-th candidate from the sorted list."""
        if self._edge_locked and self._locked_candidate is not None:
            return self._locked_candidate
        if not self._last_candidates:
            return None
        idx = self._cycle_offset % len(self._last_candidates)
        return self._last_candidates[idx]

    def _resolve_target(self, context, obj, target_world, candidate):
        """
        Given a candidate edge and raw cursor target, return the final target
        world position — applying vertex / alignment snap on the extrude axis only.

        Returns (final_target, snap) where snap is the snap_axis_coord() result or None.
        """
        if candidate is None:
            return target_world, None

        # The edge's own verts sit at the extrude origin — snapping to them
        # would collapse the extrusion, so they are excluded.
        me = obj.data
        exclude = ()
        if candidate["index"] < len(me.edges):
            exclude = tuple(me.edges[candidate["index"]].vertices)
        snap = snap_axis_coord(context, obj, target_world,
                               candidate["extrude_axis"], exclude_verts=exclude)
        if snap is None:
            return target_world, None

        # Snap only on the extrude axis, keep other axis from raw cursor
        final = Vector(target_world)
        if candidate["extrude_axis"] == 'X':
            final.x = snap[0]
        else:
            final.y = snap[0]
        return final, snap

    def _loopcut_snap_target(self, context, obj, target_world, cut_axis):
        """Locked loop cut: vertex / alignment snap on the cut axis.
        Returns (adjusted_target, snap_or_None)."""
        snap = snap_axis_coord(context, obj, target_world, cut_axis)
        if snap is None:
            return Vector(target_world), None
        adjusted = Vector(target_world)
        if cut_axis == 'X':
            adjusted.x = snap[0]
        else:
            adjusted.y = snap[0]
        return adjusted, snap

    def _build_preview(self, context, obj, target_world):
        """Recompute preview lines and snap indicator based on current state."""
        # Clear dimension label by default; extrude branch will set it
        self._dim_info = None

        # Hover on a grabbable vertex / edge midpoint takes priority — no extrude preview
        if (self._hover_vertex is not None or self._hover_edge is not None) and self._drag is None:
            self._preview_points = []
            self._preview_snap_point = None
            return

        # Loopcut lock — use frozen face and long axis, apply vertex snap on long axis
        if self._edge_locked and self._locked_mode == 'loopcut':
            face_idx = self._locked_face_index
            long_axis = self._locked_long_axis
            if face_idx is not None and long_axis is not None:
                adjusted_target, snap = self._loopcut_snap_target(
                    context, obj, target_world, long_axis
                )
                pts = self._compute_loop_cut_preview(
                    obj, face_idx, adjusted_target, forced_cut_axis=long_axis
                )
                if not pts and snap is not None:
                    # Snapped onto the face boundary (no cut possible) — use raw cursor
                    snap = None
                    pts = self._compute_loop_cut_preview(
                        obj, face_idx, target_world, forced_cut_axis=long_axis
                    )
                if pts:
                    self._preview_points = pts
                    if snap is not None:
                        # Cut line lies on the snapped coordinate; point the guide at its nearest end
                        self._set_snap_preview(snap, pts)
                    else:
                        self._set_snap_preview(None, None)
                    return

        # Loop-cut preview (no lock) when cursor is inside a face
        if not self._edge_locked:
            face_hit = find_face_at_point(obj, target_world)
            if face_hit is not None:
                face_idx, _fa = face_hit
                pts = self._compute_loop_cut_preview(obj, face_idx, target_world)
                if pts:
                    self._preview_points = pts
                    self._preview_snap_point = None
                    return

        candidate = self._pick_candidate()
        if candidate is None:
            self._preview_points = []
            self._preview_snap_point = None
            return

        final_target, snap = self._resolve_target(
            context, obj, target_world, candidate
        )

        me = obj.data
        mw = obj.matrix_world
        e = me.edges[candidate["index"]]
        v0 = mw @ me.vertices[e.vertices[0]].co
        v1 = mw @ me.vertices[e.vertices[1]].co

        state = _get_state(context)

        # Decide if hold-snap mode is active (LMB held past threshold & no vertex snap)
        hold_snap_active = False
        if self._lmb_active_extrude and snap is None and self._hold_candidate is not None:
            held_duration = time.time() - self._lmb_press_time
            moved = 0.0
            if self._lmb_press_pos is not None:
                moved = (target_world - self._lmb_press_pos).length
            movement_threshold = cm_to_blender_units(context, 1.0)
            if held_duration >= 0.15 or moved >= movement_threshold:
                hold_snap_active = True

        if hold_snap_active:
            # Snap target to nearest 10 cm step
            orig_coord = v0.x if candidate["extrude_axis"] == 'X' else v0.y
            tgt_coord = final_target.x if candidate["extrude_axis"] == 'X' else final_target.y
            delta = tgt_coord - orig_coord
            min_cm = 10.0 if state.enforce_min_distance else 0.0
            snapped_delta, snapped_cm = snap_distance_to_step(
                delta, 10.0, context, min_cm=min_cm
            )
            self._hold_snapped_cm = snapped_cm
            snapped_coord = orig_coord + snapped_delta
            if candidate["extrude_axis"] == 'X':
                final_target = Vector((snapped_coord, final_target.y, final_target.z))
            else:
                final_target = Vector((final_target.x, snapped_coord, final_target.z))
        elif state.enforce_min_distance and snap is None:
            # Regular minimum-10cm enforcement (non-hold mode)
            min_dist = cm_to_blender_units(context, 10.0)
            orig_coord = v0.x if candidate["extrude_axis"] == 'X' else v0.y
            tgt_coord = final_target.x if candidate["extrude_axis"] == 'X' else final_target.y
            delta = tgt_coord - orig_coord
            if abs(delta) < min_dist:
                sign = 1.0 if delta >= 0 else -1.0
                if delta == 0:
                    sign = 1.0
                tgt_coord = orig_coord + sign * min_dist
                if candidate["extrude_axis"] == 'X':
                    final_target = Vector((tgt_coord, final_target.y, final_target.z))
                else:
                    final_target = Vector((final_target.x, tgt_coord, final_target.z))

        new_v0 = Vector(v0)
        new_v1 = Vector(v1)
        if candidate["extrude_axis"] == 'X':
            new_v0.x = final_target.x
            new_v1.x = final_target.x
        else:
            new_v0.y = final_target.y
            new_v1.y = final_target.y

        self._preview_points = [
            v0, new_v0,
            v1, new_v1,
            new_v0, new_v1,
        ]
        self._set_snap_preview(snap, [new_v0, new_v1])

        # Compute dimension label (distance in cm)
        orig_coord = v0.x if candidate["extrude_axis"] == 'X' else v0.y
        tgt_coord = final_target.x if candidate["extrude_axis"] == 'X' else final_target.y
        delta_units = abs(tgt_coord - orig_coord)
        cm_value = blender_units_to_cm(context, delta_units)
        orig_mid = (v0 + v1) * 0.5
        new_mid = (new_v0 + new_v1) * 0.5
        self._dim_info = (orig_mid, new_mid, cm_value, hold_snap_active)

    def _compute_loop_cut_preview(self, obj, face_index, world_point,
                                  forced_cut_axis=None):
        """
        Compute preview line segments for the full loop cut across the ring
        of faces. Returns a flat list of Vectors forming LINES pairs
        (segment1_start, segment1_end, segment2_start, segment2_end, ...).
        """
        if forced_cut_axis in ('X', 'Y'):
            cut_axis = forced_cut_axis
            cut_coord = world_point.x if cut_axis == 'X' else world_point.y
        else:
            axis_info = _determine_cut_axis_from_face(obj, face_index, world_point)
            if axis_info is None:
                return None
            cut_axis, cut_coord = axis_info

        me = obj.data
        mw = obj.matrix_world

        # Build temporary bmesh for ring collection (read-only)
        bm = bmesh.new()
        try:
            bm.from_mesh(me)
            bm.faces.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            if face_index >= len(bm.faces):
                bm.free()
                return None
            start = bm.faces[face_index]
            affected, _edges = _collect_loop_cut_ring(bm, start, cut_axis, cut_coord, mw)
            if not affected:
                bm.free()
                return None

            segments = []
            for f in affected:
                # Find intersection points of cut line with the face boundary
                hits = []
                verts = list(f.verts)
                n = len(verts)
                for i in range(n):
                    a = mw @ verts[i].co
                    b = mw @ verts[(i + 1) % n].co
                    a_c = a.x if cut_axis == 'X' else a.y
                    b_c = b.x if cut_axis == 'X' else b.y
                    if (a_c - cut_coord) * (b_c - cut_coord) < -1e-9:
                        t = (cut_coord - a_c) / (b_c - a_c)
                        hit = a.lerp(b, t)
                        if cut_axis == 'X':
                            hit.x = cut_coord
                        else:
                            hit.y = cut_coord
                        hits.append(hit)

                if len(hits) >= 2:
                    # Sort along the perpendicular axis and pair extremes.
                    # For convex quads this gives a single segment; for ngons
                    # with multiple crossings we still take min/max.
                    if cut_axis == 'X':
                        hits.sort(key=lambda p: p.y)
                    else:
                        hits.sort(key=lambda p: p.x)
                    segments.append(hits[0])
                    segments.append(hits[-1])

            bm.free()
            return segments if segments else None
        except Exception:
            bm.free()
            return None

    # ---- modal loop --------------------------------------------------------

    def modal(self, context, event):
        state = _get_state(context)

        # User turned it off from the panel
        if not state.running:
            self._cancel(context)
            return {'CANCELLED'}

        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            # If moving a vertex/edge, cancel the move (restore original) instead of the addon
            if self._drag is not None:
                self._end_drag(context, restore=True)
                return {'RUNNING_MODAL'}
            self._cancel(context)
            return {'CANCELLED'}

        # Toggle lock with X — behavior depends on cursor context
        if event.type == 'X' and event.value == 'PRESS':
            if self._drag is not None:
                return {'RUNNING_MODAL'}
            if self._edge_locked:
                # Unlock whatever is locked
                self._edge_locked = False
                self._locked_candidate = None
                self._locked_mode = None
                self._locked_face_index = None
                self._locked_long_axis = None
            else:
                obj = context.active_object
                target = self._last_mouse_world
                locked = False

                if target is not None and obj is not None:
                    # Prefer locking the face (loop cut mode) if cursor is inside one
                    face_hit = find_face_at_point(obj, target)
                    if face_hit is not None:
                        face_idx, _fa = face_hit
                        # Freeze cut axis using Ctrl+R-style: the edge of this
                        # face closest to cursor determines it.
                        axis_info = _determine_cut_axis_from_face(obj, face_idx, target)
                        if axis_info is not None:
                            cut_axis, _cc = axis_info
                            self._edge_locked = True
                            self._locked_mode = 'loopcut'
                            self._locked_face_index = face_idx
                            self._locked_long_axis = cut_axis
                            locked = True

                if not locked:
                    # Fall back to edge lock on current candidate
                    current = self._pick_candidate()
                    if current is not None:
                        self._edge_locked = True
                        self._locked_mode = 'edge'
                        self._locked_candidate = dict(current)

            if self._last_mouse_world is not None:
                self._build_preview(context, context.active_object,
                                    self._last_mouse_world)
                if context.area:
                    context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # TIMER — refresh preview during LMB hold so snap activates after delay
        if event.type == 'TIMER':
            if self._lmb_active_extrude and self._last_mouse_world is not None:
                obj = context.active_object
                if obj is not None:
                    self._build_preview(context, obj, self._last_mouse_world)
                    if context.area:
                        context.area.tag_redraw()
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            target = mouse_to_world_xy(context, event)
            obj = context.active_object
            if target is not None and obj is not None:
                self._last_mouse_world = target

                # Move mode — move the captured vertex / edge live
                if self._drag is not None:
                    self._update_drag(context, obj, event, target)
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                # When locked, keep the candidate frozen — only refresh preview
                if not self._edge_locked:
                    self._last_candidates = find_edge_candidates(obj, target)
                    self._cycle_offset = 0

                # Hover detection — cursor on a vertex / edge midpoint means a
                # click grabs it. Skipped when locked (click near a vertex =
                # snap target) and during an edge extrude hold-drag.
                self._update_hover(context, obj, target)
                self._build_preview(context, obj, target)
                if context.area:
                    context.area.tag_redraw()

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            target = mouse_to_world_xy(context, event)
            obj = context.active_object
            if target is None or obj is None:
                return {'RUNNING_MODAL'}

            # Picked-up vertex/edge — this click drops it (commits the move)
            if self._drag is not None:
                self._update_drag(context, obj, event, target)
                kind = self._drag["kind"]
                bpy.ops.ed.undo_push(
                    message="Axis Extrude: Move Edge" if kind == 'edge'
                    else "Axis Extrude: Move Vertex")
                self._end_drag(context, restore=False)
                return {'RUNNING_MODAL'}

            # Press on a vertex / edge midpoint — start moving it
            # (drag, or quick click = pick up)
            if not self._edge_locked:
                hit = self._find_grab_target(context, obj, target)
                if hit is not None:
                    self._start_drag(context, event, *hit)
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            # Locked loopcut — use frozen face + axis, apply vertex snap
            if self._edge_locked and self._locked_mode == 'loopcut':
                face_idx = self._locked_face_index
                long_axis = self._locked_long_axis
                if face_idx is not None and long_axis is not None:
                    adjusted_target, snap = self._loopcut_snap_target(
                        context, obj, target, long_axis
                    )
                    ok = loop_cut_face_at_point(
                        obj, face_idx, adjusted_target,
                        forced_cut_axis=long_axis,
                    )
                    if not ok and snap is not None:
                        # Snapped onto the face boundary — cut at the raw cursor instead
                        ok = loop_cut_face_at_point(
                            obj, face_idx, target,
                            forced_cut_axis=long_axis,
                        )
                    if ok:
                        bpy.ops.ed.undo_push(message="Axis Extrude: Loop Cut")
                    else:
                        self.report({'WARNING'}, "Loop cut failed")

                self._preview_points = []
                self._preview_snap_point = None
                self._last_candidates = []
                self._cycle_offset = 0
                self._edge_locked = False
                self._locked_candidate = None
                self._locked_mode = None
                self._locked_face_index = None
                self._locked_long_axis = None
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # If cursor is inside a face and edge is NOT locked → loop cut
            if not self._edge_locked:
                face_hit = find_face_at_point(obj, target)
                if face_hit is not None:
                    face_idx, _face_axis = face_hit
                    ok = loop_cut_face_at_point(obj, face_idx, target)
                    if ok:
                        bpy.ops.ed.undo_push(message="Axis Extrude: Loop Cut")
                    else:
                        self.report({'WARNING'}, "Loop cut failed")
                    self._preview_points = []
                    self._preview_snap_point = None
                    self._last_candidates = []
                    self._cycle_offset = 0
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            # Ensure candidate list is fresh when not locked
            if not self._edge_locked:
                if not self._last_candidates or self._last_mouse_world != target:
                    self._last_candidates = find_edge_candidates(obj, target)
                    if self._cycle_offset >= len(self._last_candidates):
                        self._cycle_offset = 0
                    self._last_mouse_world = target

            candidate = self._pick_candidate()
            if candidate is None:
                self.report({'WARNING'}, "No edge found")
                return {'RUNNING_MODAL'}

            # Enter "extrude press" state — RELEASE will commit.
            # If the user holds / drags, preview becomes snapped to 10 cm steps.
            self._lmb_active_extrude = True
            self._lmb_press_time = time.time()
            self._lmb_press_pos = Vector(target)
            self._hold_candidate = dict(candidate)
            # Rebuild preview immediately (it will honor hold state)
            self._build_preview(context, obj, target)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # LMB release — commit the extrude that was started on PRESS
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            # Vertex/edge press: dragged → commit; released in place → pick it up
            if self._drag is not None and self._drag["mode"] == 'press':
                if self._drag["started"]:
                    kind = self._drag["kind"]
                    bpy.ops.ed.undo_push(
                        message="Axis Extrude: Move Edge" if kind == 'edge'
                        else "Axis Extrude: Move Vertex")
                    self._end_drag(context, restore=False)
                else:
                    self._drag["mode"] = 'pickup'
                return {'RUNNING_MODAL'}

            if not self._lmb_active_extrude:
                return {'PASS_THROUGH'}
            target = mouse_to_world_xy(context, event)
            obj = context.active_object
            if target is None or obj is None or self._hold_candidate is None:
                self._lmb_active_extrude = False
                self._hold_candidate = None
                return {'RUNNING_MODAL'}

            candidate = self._hold_candidate
            state = _get_state(context)

            # Decide whether this was a "quick click" or a "hold"
            held_duration = time.time() - self._lmb_press_time
            moved = (target - self._lmb_press_pos).length if self._lmb_press_pos else 0
            # Quick click: release within 150ms and no meaningful movement
            movement_threshold = cm_to_blender_units(context, 1.0)
            is_quick_click = (held_duration < 0.15 and moved < movement_threshold)

            # Base target resolution (with vertex snap)
            final_target, snap = self._resolve_target(
                context, obj, target, candidate
            )

            if is_quick_click or snap is not None:
                # Normal commit — snap to vertex wins over the 10cm grid too
                if state.enforce_min_distance and snap is None:
                    min_dist = cm_to_blender_units(context, 10.0)
                else:
                    min_dist = 0.0
                ok = extrude_edge_to_point(
                    obj, candidate["index"], candidate["extrude_axis"], final_target,
                    min_distance=min_dist,
                )
            else:
                # Hold commit — snap to multiples of 10 cm
                me = obj.data
                mw = obj.matrix_world
                e = me.edges[candidate["index"]]
                v0 = mw @ me.vertices[e.vertices[0]].co
                orig_coord = v0.x if candidate["extrude_axis"] == 'X' else v0.y
                tgt_coord = final_target.x if candidate["extrude_axis"] == 'X' else final_target.y
                delta = tgt_coord - orig_coord
                min_cm = 10.0 if state.enforce_min_distance else 0.0
                snapped_delta, _snap_cm = snap_distance_to_step(
                    delta, 10.0, context, min_cm=min_cm
                )
                snapped_coord = orig_coord + snapped_delta
                snapped_target = Vector(final_target)
                if candidate["extrude_axis"] == 'X':
                    snapped_target.x = snapped_coord
                else:
                    snapped_target.y = snapped_coord
                ok = extrude_edge_to_point(
                    obj, candidate["index"], candidate["extrude_axis"], snapped_target,
                    min_distance=0.0,  # already enforced via snap
                )

            if ok:
                bpy.ops.ed.undo_push(message="Axis Extrude: Extrude")
            else:
                self.report({'WARNING'}, "Extrude failed")

            # Reset state — auto-unlock after click
            self._preview_points = []
            self._preview_snap_point = None
            self._last_candidates = []
            self._cycle_offset = 0
            self._edge_locked = False
            self._locked_candidate = None
            self._locked_mode = None
            self._locked_face_index = None
            self._locked_long_axis = None
            self._lmb_active_extrude = False
            self._hold_candidate = None
            self._hold_snapped_cm = 0.0
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Let view navigation pass through
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                          'N', 'T'}:
            return {'PASS_THROUGH'}

        return {'RUNNING_MODAL'}


# ---------------------------------------------------------------------------
# Toggle operator (button target)
# ---------------------------------------------------------------------------

class AXISEXTRUDE_OT_toggle(bpy.types.Operator):
    bl_idname = "axis_extrude.toggle"
    bl_label = "Toggle Axis Extrude"
    bl_options = {'REGISTER'}

    def execute(self, context):
        state = _get_state(context)
        if state.running:
            # Flag-based stop: modal will see state.running=False and clean up
            state.running = False
            self.report({'INFO'}, "Axis Extrude: OFF")
        else:
            # Start modal
            bpy.ops.axis_extrude.modal('INVOKE_DEFAULT')
            self.report({'INFO'}, "Axis Extrude: ON")
        return {'FINISHED'}


class AXISEXTRUDE_OT_uninstall(bpy.types.Operator):
    """Uninstall this addon. Blender may show a confirmation step."""
    bl_idname = "axis_extrude.uninstall"
    bl_label = "Uninstall Axis Extrude"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # Stop modal first if running
        state = _get_state(context)
        if state.running:
            state.running = False

        pkg = __package__ or "axis_extrude"

        # Try the Extensions API first (Blender 4.2+ / 5.0)
        try:
            bpy.ops.extensions.package_uninstall(
                'INVOKE_DEFAULT',
                repo_index=-1,
                pkg_id=pkg.split(".")[-1],
            )
            self.report({'INFO'}, "Axis Extrude uninstalled")
            return {'FINISHED'}
        except Exception:
            pass

        # Fallback: legacy addon system
        try:
            bpy.ops.preferences.addon_disable(module=pkg)
        except Exception:
            pass
        try:
            bpy.ops.preferences.addon_remove(module=pkg)
            self.report({'INFO'}, "Axis Extrude removed")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Automatic uninstall failed: {e}. "
                                   f"Please remove manually from Preferences → Add-ons.")
            return {'CANCELLED'}


class AXISEXTRUDE_AddonPreferences(bpy.types.AddonPreferences):
    # For extensions, bl_idname should equal __package__
    bl_idname = __package__ or "axis_extrude"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Axis Extrude — click-to-extrude tool for 2D floor plans.")
        layout.separator()
        row = layout.row()
        row.operator("axis_extrude.uninstall", text="Uninstall addon", icon='TRASH')


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class AXISEXTRUDE_PT_panel(bpy.types.Panel):
    bl_label = "Axis Extrude"
    bl_idname = "AXISEXTRUDE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Axis Extrude"

    def draw(self, context):
        layout = self.layout
        state = _get_state(context)

        col = layout.column(align=True)
        if state.running:
            col.operator("axis_extrude.toggle", text="Disable", icon='PAUSE', depress=True)
            col.label(text="Active — click to extrude", icon='INFO')
            col.label(text="Right-click to cancel")
        else:
            col.operator("axis_extrude.toggle", text="Enable", icon='PLAY')

        layout.separator()
        layout.prop(state, "enforce_min_distance", text="Minimum 10 cm extrude")

        layout.separator()
        box = layout.box()
        box.label(text="How it works:")
        box.label(text="• Top view (XY) 2D workflow")
        box.label(text="• Move mouse — nearest edge highlights")
        box.label(text="• Click — extrude to cursor")
        box.label(text="• Press X — lock/unlock selection")
        box.label(text="• Click inside a face — loop cut")
        box.label(text="• Drag a vertex — move on X/Y (10 cm steps)")
        box.label(text="• Click a vertex — pick up, click to drop")
        box.label(text="• Drag edge midpoint (square) — move edge")
        box.label(text="• Snaps to nearby vertex")
        box.label(text="• Aligns with vertex X/Y lines (dashed)")
        box.label(text="• On disable: merge 0.01 cm")
        box.label(text="• Right-click — disable")


# ---------------------------------------------------------------------------
# State PropertyGroup
# ---------------------------------------------------------------------------

class AXISEXTRUDE_state(bpy.types.PropertyGroup):
    running: bpy.props.BoolProperty(
        name="Running",
        default=False,
    )
    enforce_min_distance: bpy.props.BoolProperty(
        name="Minimalna ekstruzja 10 cm",
        description="Wymusza minimalną ekstruzję 10 cm. Snap do vertexa zawsze ignoruje to ograniczenie.",
        default=True,
    )


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

classes = (
    AXISEXTRUDE_state,
    AXISEXTRUDE_OT_modal,
    AXISEXTRUDE_OT_toggle,
    AXISEXTRUDE_OT_uninstall,
    AXISEXTRUDE_AddonPreferences,
    AXISEXTRUDE_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.axis_extrude_state = bpy.props.PointerProperty(
        type=AXISEXTRUDE_state
    )


def unregister():
    # Clean up any lingering draw handlers
    if _draw_handle["handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle["handle"], 'WINDOW')
        except Exception:
            pass
        _draw_handle["handle"] = None
    if _draw_handle.get("text_handle") is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle["text_handle"], 'WINDOW')
        except Exception:
            pass
        _draw_handle["text_handle"] = None

    del bpy.types.WindowManager.axis_extrude_state
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
