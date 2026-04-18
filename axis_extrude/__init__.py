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


def find_nearest_vertex(obj, world_point, exclude_edge_index=None, max_dist=None):
    """
    Return (world_co, vert_index) for the vertex closest to world_point in XY.
    If `exclude_edge_index` is given, verts belonging to that edge are skipped
    (we don't want to snap to the edge we're extruding FROM).
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
    Picks the incident edge of the vertex closest to world_point in XY,
    then moves perpendicular to that edge's direction.

    Returns 'X' or 'Y' (the drag axis), or None if no incident edge found.
    """
    me = obj.data
    mw = obj.matrix_world
    if vert_index >= len(me.vertices):
        return None

    # Find edges incident to this vertex
    p = Vector((world_point.x, world_point.y))
    v_co = mw @ me.vertices[vert_index].co

    best = None  # (dist, edge_axis)
    for e in me.edges:
        if vert_index not in e.vertices:
            continue
        # Other endpoint
        other_idx = e.vertices[0] if e.vertices[1] == vert_index else e.vertices[1]
        v1 = mw @ me.vertices[other_idx].co

        a = Vector((v_co.x, v_co.y))
        b = Vector((v1.x, v1.y))
        seg = b - a
        L2 = seg.length_squared
        if L2 < 1e-12:
            continue
        t = max(0.0, min(1.0, (p - a).dot(seg) / L2))
        closest = a + seg * t
        d = (p - closest).length

        if abs(seg.x) >= abs(seg.y):
            edge_axis = 'X'  # edge runs along X → perpendicular drag = Y
        else:
            edge_axis = 'Y'
        if best is None or d < best[0]:
            best = (d, edge_axis)

    if best is None:
        return None
    edge_axis = best[1]
    # Drag perpendicular to the edge direction
    return 'Y' if edge_axis == 'X' else 'X'


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

_draw_handle = {"handle": None}


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
    dragging = getattr(op, "_dragging_vertex", None)
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

    # Dragging indicator — draw axis guide line through the vertex
    if dragging is not None:
        vi, axis, _orig = dragging
        obj = context.active_object
        if obj is not None and obj.type == 'MESH' and vi < len(obj.data.vertices):
            mw = obj.matrix_world
            vw = mw @ obj.data.vertices[vi].co
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
        self._cycle_offset = 0           # which candidate from sorted list
        self._last_mouse_world = None    # last XY target under cursor
        self._last_candidates = []       # cached candidate list
        self._edge_locked = False        # X toggles this — freezes edge selection
        self._locked_candidate = None    # candidate dict that is locked
        self._locked_mode = None         # None | 'edge' | 'loopcut'
        self._locked_face_index = None   # for loopcut lock
        self._locked_long_axis = None    # 'X' or 'Y' — long axis of locked face
        # Vertex drag mode
        self._hover_vertex = None        # (world_pos, vert_index) if cursor hovers a vertex
        self._dragging_vertex = None     # (vert_index, axis, orig_world_other_coord)

        args = (self, context)
        _draw_handle["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, args, 'WINDOW', 'POST_VIEW'
        )

        state.running = True
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _cancel(self, context):
        state = _get_state(context)
        state.running = False

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
        world position — applying vertex snap on the extrude axis only.

        Returns (final_target, snap_vert_world_or_None).
        """
        if candidate is None:
            return target_world, None

        VERT_SNAP_PIXELS = 20.0
        vert_hit = find_nearest_vertex(
            obj, target_world,
            exclude_edge_index=candidate["index"],
        )
        if vert_hit is None:
            return target_world, None

        vert_world, _vert_idx = vert_hit
        if pixel_distance(context, vert_world, target_world) > VERT_SNAP_PIXELS:
            return target_world, None

        # Snap only on the extrude axis, keep other axis from raw cursor
        final = Vector(target_world)
        if candidate["extrude_axis"] == 'X':
            final.x = vert_world.x
        else:
            final.y = vert_world.y
        return final, vert_world

    def _build_preview(self, context, obj, target_world):
        """Recompute preview lines and snap indicator based on current state."""
        # Hover on vertex takes priority — no extrude preview
        if self._hover_vertex is not None and self._dragging_vertex is None:
            self._preview_points = []
            self._preview_snap_point = None
            return

        # Loopcut lock — use frozen face and long axis, apply vertex snap on long axis
        if self._edge_locked and self._locked_mode == 'loopcut':
            face_idx = self._locked_face_index
            long_axis = self._locked_long_axis
            if face_idx is not None and long_axis is not None:
                adjusted_target = Vector(target_world)
                snap_vert = None

                # Vertex snap on the long axis (cut axis) — any vertex in mesh
                VERT_SNAP_PIXELS = 20.0
                vert_hit = find_nearest_vertex(obj, target_world)  # no exclusions
                if vert_hit is not None:
                    vw, _vi = vert_hit
                    if pixel_distance(context, vw, target_world) <= VERT_SNAP_PIXELS:
                        if long_axis == 'X':
                            adjusted_target.x = vw.x
                        else:
                            adjusted_target.y = vw.y
                        snap_vert = vw

                pts = self._compute_loop_cut_preview(
                    obj, face_idx, adjusted_target, forced_cut_axis=long_axis
                )
                if pts:
                    self._preview_points = pts
                    self._preview_snap_point = snap_vert
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

        final_target, snap_vert = self._resolve_target(
            context, obj, target_world, candidate
        )

        me = obj.data
        mw = obj.matrix_world
        e = me.edges[candidate["index"]]
        v0 = mw @ me.vertices[e.vertices[0]].co
        v1 = mw @ me.vertices[e.vertices[1]].co

        # Enforce minimum extrude distance (10 cm) in preview too —
        # only when the user hasn't snapped to a vertex (snap always wins)
        state = _get_state(context)
        if state.enforce_min_distance and snap_vert is None:
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
        self._preview_snap_point = snap_vert

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

        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            # If dragging a vertex, cancel the drag (restore original) instead of the addon
            if self._dragging_vertex is not None:
                vi, axis, orig_coord = self._dragging_vertex
                if orig_coord is not None:
                    move_vertex_along_axis(context.active_object, vi, axis, orig_coord)
                self._dragging_vertex = None
                self._hover_vertex = None
                self._preview_points = []
                self._preview_snap_point = None
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            self._cancel(context)
            return {'CANCELLED'}

        if event.type == 'ESC' and event.value == 'PRESS':
            if self._dragging_vertex is not None:
                vi, axis, orig_coord = self._dragging_vertex
                if orig_coord is not None:
                    move_vertex_along_axis(context.active_object, vi, axis, orig_coord)
                self._dragging_vertex = None
                self._hover_vertex = None
                self._preview_points = []
                self._preview_snap_point = None
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            self._cancel(context)
            return {'CANCELLED'}

        # Toggle lock with X — behavior depends on cursor context
        if event.type == 'X' and event.value == 'PRESS':
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

        if event.type == 'MOUSEMOVE':
            target = mouse_to_world_xy(context, event)
            obj = context.active_object
            if target is not None and obj is not None:
                self._last_mouse_world = target

                # Vertex drag mode — move the captured vertex live
                if self._dragging_vertex is not None:
                    vidx, axis, _orig_other = self._dragging_vertex
                    new_coord = target.x if axis == 'X' else target.y
                    move_vertex_along_axis(obj, vidx, axis, new_coord)
                    # Clear preview (dragging doesn't need extrude preview)
                    self._preview_points = []
                    self._preview_snap_point = None
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                # Hover detection — is the cursor close to an existing vertex?
                HOVER_PIXELS = 15.0
                self._hover_vertex = None
                vert_hit = find_nearest_vertex(obj, target)
                if vert_hit is not None:
                    vw, vi = vert_hit
                    if pixel_distance(context, vw, target) <= HOVER_PIXELS:
                        self._hover_vertex = (vw, vi)

                # When locked, keep the candidate frozen — only refresh preview
                if not self._edge_locked:
                    self._last_candidates = find_edge_candidates(obj, target)
                    self._cycle_offset = 0
                self._build_preview(context, obj, target)
                if context.area:
                    context.area.tag_redraw()

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            target = mouse_to_world_xy(context, event)
            obj = context.active_object
            if target is None or obj is None:
                return {'RUNNING_MODAL'}

            # Vertex drag drop — second click commits the move
            if self._dragging_vertex is not None:
                bpy.ops.ed.undo_push(message="Axis Extrude: Move Vertex")
                self._dragging_vertex = None
                self._hover_vertex = None
                self._preview_points = []
                self._preview_snap_point = None
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Vertex drag pickup — if cursor is hovering a vertex, start drag
            if self._hover_vertex is not None and not self._edge_locked:
                vw, vi = self._hover_vertex
                axis = determine_vertex_drag_axis(obj, vi, target)
                if axis is not None:
                    # Remember original coord on drag axis for cancel
                    orig_coord = vw.x if axis == 'X' else vw.y
                    self._dragging_vertex = (vi, axis, orig_coord)
                    self._preview_points = []
                    self._preview_snap_point = None
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            # Locked loopcut — use frozen face + axis, apply vertex snap
            if self._edge_locked and self._locked_mode == 'loopcut':
                face_idx = self._locked_face_index
                long_axis = self._locked_long_axis
                if face_idx is not None and long_axis is not None:
                    adjusted_target = Vector(target)
                    VERT_SNAP_PIXELS = 20.0
                    vert_hit = find_nearest_vertex(obj, target)
                    if vert_hit is not None:
                        vw, _vi = vert_hit
                        if pixel_distance(context, vw, target) <= VERT_SNAP_PIXELS:
                            if long_axis == 'X':
                                adjusted_target.x = vw.x
                            else:
                                adjusted_target.y = vw.y

                    ok = loop_cut_face_at_point(
                        obj, face_idx, adjusted_target,
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

            final_target, snap_vert = self._resolve_target(
                context, obj, target, candidate
            )

            state = _get_state(context)
            if state.enforce_min_distance and snap_vert is None:
                min_dist = cm_to_blender_units(context, 10.0)
            else:
                min_dist = 0.0
            ok = extrude_edge_to_point(
                obj, candidate["index"], candidate["extrude_axis"], final_target,
                min_distance=min_dist,
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
        box.label(text="• Snaps to nearby vertex")
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
    # Clean up any lingering draw handler
    if _draw_handle["handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle["handle"], 'WINDOW')
        except Exception:
            pass
        _draw_handle["handle"] = None

    del bpy.types.WindowManager.axis_extrude_state
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
