bl_info = {
    "name": "Axis Extrude",
    "author": "Patryk M.",
    "version": (0, 3, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Axis Extrude",
    "description": "Click-to-extrude edges along world axes in top, front and side views.",
    "category": "Mesh",
}

import bpy
import bmesh
import blf
import math
import time
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _get_state(context):
    return context.window_manager.axis_extrude_state


# ---------------------------------------------------------------------------
# Work plane — which world axes the tool operates on
# ---------------------------------------------------------------------------

AXES = "XYZ"
ALIGN_DOT = 0.999          # view direction this close to a world axis → axis-aligned view
FACE_PARALLEL_DOT = 0.99   # face normal this close to the depth axis → face looks at the viewer
WALL_ALIGN_DOT = 0.9       # MMB double-click: wall normal must be this close to a world axis
COINCIDENT_TOL = 1e-4      # world units — elements closer than this overlap on screen
PIN_TOL = 1e-3             # world units — element lies on the pinned wall plane

# (depth axis, side the viewer is on) → Blender view type
VIEW_FOR_SIDE = {
    (2, +1): 'TOP', (2, -1): 'BOTTOM',
    (1, -1): 'FRONT', (1, +1): 'BACK',
    (0, +1): 'RIGHT', (0, -1): 'LEFT',
}
VIEW_AXIS = {'TOP': 2, 'BOTTOM': 2, 'FRONT': 1, 'BACK': 1, 'RIGHT': 0, 'LEFT': 0}
OPPOSITE_VIEW = {'TOP': 'BOTTOM', 'BOTTOM': 'TOP', 'FRONT': 'BACK', 'BACK': 'FRONT',
                 'RIGHT': 'LEFT', 'LEFT': 'RIGHT'}
VIEW_LABEL = {'TOP': "Góra", 'BOTTOM': "Dół", 'FRONT': "Przód", 'BACK': "Tył",
              'RIGHT': "Prawy", 'LEFT': "Lewy"}


class WorkPlane:
    """
    The world plane the tool works in, derived from the viewport:
      top/bottom → XY, front/back → XZ, right/left → YZ.
    Views not aligned to a world axis fall back to XY (the original 2D workflow).
      u, v — in-plane axis indices (0=X, 1=Y, 2=Z); n — the depth axis.
    """
    __slots__ = ("n", "u", "v", "view_dir", "aligned")

    def __init__(self, n, view_dir, aligned):
        self.n = n
        self.u, self.v = [i for i in range(3) if i != n]
        self.view_dir = view_dir
        self.aligned = aligned

    @property
    def name(self):
        return AXES[self.u] + AXES[self.v]

    @property
    def view_type(self):
        """Blender view type ('TOP', 'FRONT', ...) or None for a free view."""
        if not self.aligned:
            return None
        side = -1 if self.view_dir[self.n] > 0 else +1
        return VIEW_FOR_SIDE[(self.n, side)]

    def other(self, axis):
        return self.v if axis == self.u else self.u

    def closeness(self, world):
        """Larger = nearer to the viewer."""
        return -world.dot(self.view_dir)

    def uv(self, world):
        return Vector((world[self.u], world[self.v]))


def detect_plane(rv3d):
    if rv3d is None:
        return WorkPlane(2, Vector((0.0, 0.0, -1.0)), False)
    view_dir = (rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))).normalized()
    n = max(range(3), key=lambda i: abs(view_dir[i]))
    if abs(view_dir[n]) >= ALIGN_DOT:
        return WorkPlane(n, view_dir, True)
    return WorkPlane(2, view_dir, False)


class Scope:
    """
    Which elements the tool may act on:
      mode 'NEAREST' — the element nearest to the viewer (or only the pinned wall),
      mode 'ALL'     — every element overlapping on screen.
    """
    __slots__ = ("plane", "mode", "pinned")

    def __init__(self, plane, mode, pinned=None):
        self.plane = plane
        self.mode = mode
        self.pinned = pinned  # depth along plane.n of the pinned wall, or None

    def depth_ok(self, world):
        return self.pinned is None or abs(world[self.plane.n] - self.pinned) <= PIN_TOL


def toggle_view(slot, current):
    """HUD button: clicking the active view flips to the opposite side."""
    return OPPOSITE_VIEW[slot] if current == slot else slot


def gesture_view(delta, current):
    """RMB gesture: up = top, down = front (repeat flips), right / left = sides."""
    if abs(delta.y) >= abs(delta.x):
        slot = 'TOP' if delta.y > 0 else 'FRONT'
        return toggle_view(slot, current)
    return 'RIGHT' if delta.x > 0 else 'LEFT'


def view_for_face_normal(normal, view_dir):
    """
    View that looks straight at a face. The normal is flipped to face the
    current viewer first. Returns (view_type, depth_axis) or None when the
    face isn't parallel to a world plane.
    """
    nw = Vector(normal).normalized()
    if nw.dot(view_dir) > 0:
        nw = -nw
    n = max(range(3), key=lambda i: abs(nw[i]))
    if abs(nw[n]) < WALL_ALIGN_DOT:
        return None
    side = +1 if nw[n] > 0 else -1
    return VIEW_FOR_SIDE[(n, side)], n


# ---------------------------------------------------------------------------
# Viewport projection helpers
# ---------------------------------------------------------------------------

class View:
    """The 3D viewport region the tool works in."""
    __slots__ = ("region", "rv3d")

    def __init__(self, region, rv3d):
        self.region = region
        self.rv3d = rv3d

    def to_px(self, world):
        """World point → region pixel coords, or None if behind the camera."""
        if self.region is None or self.rv3d is None:
            return None
        return view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, world)


def mouse_to_world(view, px, plane, depth=0.0):
    """Project region pixel coords onto the work plane (depth axis = `depth`)."""
    if view.region is None or view.rv3d is None:
        return None
    origin = view3d_utils.region_2d_to_origin_3d(view.region, view.rv3d, px)
    direction = view3d_utils.region_2d_to_vector_3d(view.region, view.rv3d, px)
    n = plane.n
    if abs(direction[n]) < 1e-8:
        return None
    t = (depth - origin[n]) / direction[n]
    if t < 0 and view.rv3d.is_perspective:
        return None
    hit = origin + direction * t
    hit[n] = depth
    return hit


def pixel_distance(view, world_a, world_b):
    """Screen-space pixel distance between two world points, or inf."""
    a = view.to_px(world_a)
    b = view.to_px(world_b)
    if a is None or b is None:
        return float('inf')
    return (a - b).length


def pixels_per_unit(view, world_point, axis=0):
    """Screen pixels per 1 world unit along `axis` around world_point, or None."""
    a = view.to_px(world_point)
    step = Vector(world_point)
    step[axis] += 1.0
    b = view.to_px(step)
    if a is None or b is None:
        return None
    ppu = (b - a).length
    return ppu if ppu > 1e-9 else None


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def _scale_length(context):
    try:
        scale = context.scene.unit_settings.scale_length
        return scale if scale > 0 else 1.0
    except Exception:
        return 1.0


def cm_to_blender_units(context, cm):
    """Centimeters → Blender units, respecting scene.unit_settings.scale_length."""
    return (cm * 0.01) / _scale_length(context)


def blender_units_to_cm(context, units):
    """Inverse of cm_to_blender_units."""
    return (units * _scale_length(context)) / 0.01


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
    sign = -1.0 if delta_units < 0 else 1.0
    return sign * cm_to_blender_units(context, snapped_cm), snapped_cm


# ---------------------------------------------------------------------------
# Mesh queries (all in world space; `coords` = mesh_world_coords(obj))
# ---------------------------------------------------------------------------

def mesh_world_coords(obj):
    mw = obj.matrix_world
    return [mw @ v.co for v in obj.data.vertices]


def edge_face_counts(me):
    counts = [0] * len(me.edges)
    loops = me.loops
    for p in me.polygons:
        for li in p.loop_indices:
            counts[loops[li].edge_index] += 1
    return counts


def edge_midpoint(obj, edge_index, coords):
    me = obj.data
    if edge_index >= len(me.edges):
        return None
    e = me.edges[edge_index]
    return (coords[e.vertices[0]] + coords[e.vertices[1]]) * 0.5


def find_edge_candidates(obj, world_point, scope, coords):
    """
    Edges sorted by on-screen (work plane) distance to world_point.
    Each entry: dict(index, edge_axis, extrude_axis, dist, length).
      - edge_axis:    axis index the edge runs along (one of plane.u / plane.v)
      - extrude_axis: the other in-plane axis — where the new face will stretch
    Edges seen end-on (pointing at the viewer) are skipped.
    Overlapping edges: NEAREST prefers the one closest to the viewer,
    ALL prefers the longest (so its group covers the most).
    """
    if obj is None or obj.type != 'MESH':
        return []
    plane = scope.plane
    p = plane.uv(world_point)
    candidates = []
    for e in obj.data.edges:
        a = coords[e.vertices[0]]
        b = coords[e.vertices[1]]
        if scope.pinned is not None and not (scope.depth_ok(a) and scope.depth_ok(b)):
            continue
        pa = plane.uv(a)
        seg = plane.uv(b) - pa
        seg_len_sq = seg.length_squared
        if seg_len_sq < 1e-12:
            continue
        edge_axis = plane.u if abs(seg.x) >= abs(seg.y) else plane.v
        t = max(0.0, min(1.0, (p - pa).dot(seg) / seg_len_sq))
        dist = (p - (pa + seg * t)).length
        length = math.sqrt(seg_len_sq)
        if scope.mode == 'ALL':
            key = (round(dist, 5), -length)
        else:
            key = (round(dist, 5), -round(plane.closeness((a + b) * 0.5), 5))
        candidates.append((key, {
            "index": e.index,
            "edge_axis": edge_axis,
            "extrude_axis": plane.other(edge_axis),
            "dist": dist,
            "length": length,
        }))
    candidates.sort(key=lambda c: c[0])
    return [c[1] for c in candidates]


def coincident_edge_group(obj, edge_index, plane, coords, boundary_counts=None):
    """
    Everything that looks like one line on screen together with `edge_index`:
    edges lying on the same projected line whose spans chain into the span of
    `edge_index` — including edges seen end-on (e.g. in front view a flat
    floor plan collapses into one line at Z=0).
    If `boundary_counts` (edge_face_counts) is given, interior edges
    (2+ faces) are dropped from the result.
    """
    me = obj.data
    e0 = me.edges[edge_index]
    a0 = coords[e0.vertices[0]]
    b0 = coords[e0.vertices[1]]
    seg = plane.uv(b0) - plane.uv(a0)
    along = plane.u if abs(seg.x) >= abs(seg.y) else plane.v
    across = plane.other(along)
    c0 = a0[across]
    if abs(b0[across] - c0) > COINCIDENT_TOL:
        return [edge_index]  # diagonal on screen — nothing can coincide with it

    spans = []
    for e in me.edges:
        a = coords[e.vertices[0]]
        b = coords[e.vertices[1]]
        if abs(a[across] - c0) > COINCIDENT_TOL or abs(b[across] - c0) > COINCIDENT_TOL:
            continue
        lo, hi = sorted((a[along], b[along]))
        spans.append((lo, hi, e.index))
    spans.sort()

    # Sweep into clusters of overlapping/touching spans; keep the one with edge_index
    cluster, c_hi, found = [], None, False
    for lo, hi, ei in spans:
        if c_hi is not None and lo > c_hi + COINCIDENT_TOL:
            if found:
                break
            cluster, c_hi = [], None
        cluster.append(ei)
        c_hi = hi if c_hi is None else max(c_hi, hi)
        if ei == edge_index:
            found = True
    group = cluster if found else [edge_index]

    if boundary_counts is not None:
        group = [ei for ei in group if boundary_counts[ei] <= 1 or ei == edge_index]
    return group


def coincident_verts(coords, vert_index, plane):
    """Vertices at the same screen position as vert_index (any depth)."""
    p = plane.uv(coords[vert_index])
    return [i for i, w in enumerate(coords) if (plane.uv(w) - p).length <= COINCIDENT_TOL]


def find_nearest_vertex(obj, world_point, scope, coords, exclude=None):
    """
    Return (world_co, vert_index) for the vertex closest to world_point on
    screen (work plane). Ties (overlapping verts) go to the one nearest the
    viewer. Pinned scope only considers verts on the pinned wall.
    """
    if obj is None or obj.type != 'MESH':
        return None
    plane = scope.plane
    p = plane.uv(world_point)
    best = None
    for i, w in enumerate(coords):
        if exclude and i in exclude:
            continue
        if not scope.depth_ok(w):
            continue
        d = (plane.uv(w) - p).length
        key = (round(d, 5), -round(plane.closeness(w), 5))
        if best is None or key < best[0]:
            best = (key, i, w)
    if best is None:
        return None
    return best[2], best[1]


POINT_SNAP_PIXELS = 20.0   # cursor this close to a vertex → snap to it
LINE_SNAP_PIXELS = 10.0    # cursor this close to the axis line through a vertex → align


def snap_axis_coord(view, obj, target, axis, plane, coords, exclude_verts=()):
    """
    Snap the `axis` coordinate of `target`:
      1. point snap — a vertex within POINT_SNAP_PIXELS of the cursor,
      2. line snap  — the cursor within LINE_SNAP_PIXELS of the axis-aligned
                      line through any vertex (alignment with distant geometry).
    Returns (coord, ref_vertex_world, kind) with kind 'point' | 'line', or None.
    """
    if obj is None or obj.type != 'MESH':
        return None
    ppu = pixels_per_unit(view, target, plane.u)
    cursor_px = view.to_px(target)
    if ppu is None or cursor_px is None:
        return None

    excluded = set(exclude_verts)
    other = plane.other(axis)
    cur_c = target[axis]

    best_point = None  # (px, coord, world)
    best_line = None   # (key, coord, world)
    for i, w in enumerate(coords):
        if i in excluded:
            continue
        c = w[axis]
        line_px = abs(c - cur_c) * ppu
        # Distance to the point is never smaller than to the line through it
        if line_px > POINT_SNAP_PIXELS:
            continue
        v_px = view.to_px(w)
        if v_px is not None:
            d = (v_px - cursor_px).length
            if d <= POINT_SNAP_PIXELS and (best_point is None or d < best_point[0]):
                best_point = (d, c, w)
        if line_px <= LINE_SNAP_PIXELS:
            # Among vertices on (nearly) the same line, reference the closest one
            key = (round(line_px, 2), abs(w[other] - target[other]))
            if best_line is None or key < best_line[0]:
                best_line = (key, c, w)

    if best_point is not None:
        return best_point[1], best_point[2], 'point'
    if best_line is not None:
        return best_line[1], best_line[2], 'line'
    return None


def _normal_matrix(obj):
    return obj.matrix_world.inverted_safe().transposed().to_3x3()


def _faces_viewer(nmat, local_normal, plane):
    nw = nmat @ local_normal
    if nw.length < 1e-12:
        return False
    nw.normalize()
    return abs(nw[plane.n]) >= FACE_PARALLEL_DOT


def find_face_at_point(obj, world_point, scope, coords):
    """
    Index of the face under world_point, or None. Only faces looking at the
    viewer (parallel to the work plane) count. Overlaps: nearest to the
    viewer first, then the smallest.
    """
    if obj is None or obj.type != 'MESH':
        return None
    plane = scope.plane
    nmat = _normal_matrix(obj)
    p = plane.uv(world_point)

    best = None  # (key, face_index)
    for f in obj.data.polygons:
        if len(f.vertices) < 3:
            continue
        if not _faces_viewer(nmat, f.normal, plane):
            continue
        pts = [coords[vi] for vi in f.vertices]
        if scope.pinned is not None and not all(scope.depth_ok(w) for w in pts):
            continue
        pts2 = [plane.uv(w) for w in pts]

        # Point-in-polygon (ray casting)
        inside = False
        n = len(pts2)
        j = n - 1
        for i in range(n):
            xi, yi = pts2[i].x, pts2[i].y
            xj, yj = pts2[j].x, pts2[j].y
            if (yi > p.y) != (yj > p.y):
                x_int = (xj - xi) * (p.y - yi) / (yj - yi + 1e-30) + xi
                if p.x < x_int:
                    inside = not inside
            j = i
        if not inside:
            continue

        xs = [q.x for q in pts2]
        ys = [q.y for q in pts2]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        center = sum(pts, Vector()) / len(pts)
        key = (-round(plane.closeness(center), 5), area)
        if best is None or key < best[0]:
            best = (key, f.index)

    return None if best is None else best[1]


def determine_cut_axis_from_face(obj, face_index, world_point, plane, coords):
    """
    Pick the cut axis from the face edge closest to world_point (Ctrl+R-style:
    the cut runs PARALLEL to that edge). Returns (cut_axis, cut_coord) or None;
    cut_coord is measured on cut_axis.
    """
    me = obj.data
    if face_index >= len(me.polygons):
        return None
    p = plane.uv(world_point)
    best = None  # (dist, edge_axis)
    for ek in me.polygons[face_index].edge_keys:
        a = plane.uv(coords[ek[0]])
        seg = plane.uv(coords[ek[1]]) - a
        L2 = seg.length_squared
        if L2 < 1e-12:
            continue
        t = max(0.0, min(1.0, (p - a).dot(seg) / L2))
        d = (p - (a + seg * t)).length
        edge_axis = plane.u if abs(seg.x) >= abs(seg.y) else plane.v
        if best is None or d < best[0]:
            best = (d, edge_axis)
    if best is None:
        return None
    cut_axis = best[1]
    return cut_axis, world_point[cut_axis]


def _collect_loop_cut_ring(bm, start_face, cut_axis, cut_coord, mw, restrict_plane=None):
    """
    BFS from start_face across faces whose boundary straddles cut_coord on
    cut_axis. A face is included if it has at least 2 straddling edges; we hop
    to neighbors through those edges.
    If restrict_plane is given, only faces looking at the viewer at the same
    depth as start_face join the ring (NEAREST mode: one wall only).
    Returns (list_of_affected_faces, set_of_straddling_edges).
    """
    def edge_straddles(e):
        a = (mw @ e.verts[0].co)[cut_axis]
        b = (mw @ e.verts[1].co)[cut_axis]
        return (a - cut_coord) * (b - cut_coord) < -1e-9

    if not start_face.is_valid:
        return [], set()

    if restrict_plane is not None:
        nmat = mw.inverted_safe().transposed().to_3x3()
        start_depth = (mw @ start_face.verts[0].co)[restrict_plane.n]

        def face_allowed(f):
            if not _faces_viewer(nmat, f.normal, restrict_plane):
                return False
            return abs((mw @ f.verts[0].co)[restrict_plane.n] - start_depth) <= PIN_TOL
    else:
        def face_allowed(f):
            return True

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
            if nf.index in visited or not face_allowed(nf):
                continue
            nf_straddling = [ee for ee in nf.edges if edge_straddles(ee)]
            if len(nf_straddling) < 2:
                continue
            visited.add(nf.index)
            affected.append(nf)
            for se in nf_straddling:
                if se not in edges_set:
                    edges_set.add(se)
                    queue.append(se)

    return affected, edges_set


def _resolve_cut(obj, face_index, world_point, plane, coords, forced_cut_axis):
    if forced_cut_axis is not None:
        return forced_cut_axis, world_point[forced_cut_axis]
    return determine_cut_axis_from_face(obj, face_index, world_point, plane, coords)


def loop_cut_face_at_point(obj, face_index, world_point, plane, restrict, coords,
                           forced_cut_axis=None):
    """
    Ctrl+R-style loop cut. Axis comes from the face edge closest to world_point
    unless forced_cut_axis is given; the cut propagates through the ring of
    faces (only the wall at the same depth when `restrict`).
    """
    cut = _resolve_cut(obj, face_index, world_point, plane, coords, forced_cut_axis)
    if cut is None:
        return False
    cut_axis, cut_coord = cut

    me = obj.data
    mw = obj.matrix_world
    mw_inv = mw.inverted()

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    if face_index >= len(bm.faces) or not bm.faces[face_index].is_valid:
        bm.free()
        return False

    affected, edges_to_cut = _collect_loop_cut_ring(
        bm, bm.faces[face_index], cut_axis, cut_coord, mw,
        restrict_plane=plane if restrict else None,
    )
    if not affected or not edges_to_cut:
        bm.free()
        return False

    new_verts = []
    for e in list(edges_to_cut):
        a = (mw @ e.verts[0].co)[cut_axis]
        b = (mw @ e.verts[1].co)[cut_axis]
        if abs(b - a) < 1e-12:
            continue
        t = max(0.0, min(1.0, (cut_coord - a) / (b - a)))
        try:
            _ne, nv = bmesh.utils.edge_split(e, e.verts[0], t)
        except Exception:
            continue
        nw = mw @ nv.co
        nw[cut_axis] = cut_coord
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


def compute_loop_cut_preview(obj, face_index, world_point, plane, restrict, coords,
                             forced_cut_axis=None):
    """
    Preview segments for the loop cut across the ring of faces, as a flat
    list of Vectors forming LINES pairs, or None.
    """
    cut = _resolve_cut(obj, face_index, world_point, plane, coords, forced_cut_axis)
    if cut is None:
        return None
    cut_axis, cut_coord = cut

    mw = obj.matrix_world
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        if face_index >= len(bm.faces):
            return None
        affected, _edges = _collect_loop_cut_ring(
            bm, bm.faces[face_index], cut_axis, cut_coord, mw,
            restrict_plane=plane if restrict else None,
        )
        segments = []
        for f in affected:
            hits = []
            verts = list(f.verts)
            n = len(verts)
            for i in range(n):
                a = mw @ verts[i].co
                b = mw @ verts[(i + 1) % n].co
                a_c = a[cut_axis]
                b_c = b[cut_axis]
                if (a_c - cut_coord) * (b_c - cut_coord) < -1e-9:
                    hit = a.lerp(b, (cut_coord - a_c) / (b_c - a_c))
                    hit[cut_axis] = cut_coord
                    hits.append(hit)
            if len(hits) >= 2:
                # Pair the extremes along whichever remaining axis the hits spread on
                rest = [i for i in range(3) if i != cut_axis]
                spread_axis = max(rest, key=lambda ax: max(h[ax] for h in hits) - min(h[ax] for h in hits))
                hits.sort(key=lambda h: h[spread_axis])
                segments.append(hits[0])
                segments.append(hits[-1])
        return segments or None
    except Exception:
        return None
    finally:
        bm.free()


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


def extrude_edges_to_coord(obj, edge_indices, axis, coord):
    """
    Extrude the given edges together and place the new verts at world
    coordinate `coord` on `axis` (other coordinates copied from the originals).
    """
    me = obj.data
    mw = obj.matrix_world
    mw_inv = mw.inverted()

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    edges = [bm.edges[i] for i in edge_indices if i < len(bm.edges) and bm.edges[i].is_valid]
    if not edges:
        bm.free()
        return False

    ret = bmesh.ops.extrude_edge_only(bm, edges=edges)
    new_verts = [g for g in ret['geom'] if isinstance(g, bmesh.types.BMVert)]
    if not new_verts:
        bm.free()
        return False

    # New verts start on top of the originals — only the extrude axis changes
    for nv in new_verts:
        w = mw @ nv.co
        w[axis] = coord
        nv.co = mw_inv @ w

    bm.to_mesh(me)
    bm.free()
    me.update()
    return True


# ---------------------------------------------------------------------------
# HUD — view buttons and depth mode switch at the bottom of the viewport
# ---------------------------------------------------------------------------

HUD_FONT = 0


def hud_layout(context, region, plane, mode, pin):
    """
    Buttons as dicts: id (None = label only), label, active, rect (x0, y0, x1, y1)
    in region pixels. Shared by drawing and click hit-testing.
    """
    scale = context.preferences.system.ui_scale
    current = plane.view_type

    items = []  # (id, label, active, group)
    if plane.aligned:
        items.append((None, plane.name, False, 0))
    else:
        items.append((None, "Swobodny: XY", False, 0))
    for slot in ('TOP', 'FRONT', 'RIGHT'):
        opposite = OPPOSITE_VIEW[slot]
        if current == opposite:
            items.append(("view:" + slot, VIEW_LABEL[opposite], True, 1))
        else:
            items.append(("view:" + slot, VIEW_LABEL[slot], current == slot, 1))
    items.append(("mode:NEAREST", "Najbliższe", mode == 'NEAREST', 2))
    items.append(("mode:ALL", "Wszystkie", mode == 'ALL', 2))
    if pin is not None:
        cm = blender_units_to_cm(context, pin[1])
        items.append(("unpin", f"Ściana {AXES[pin[0]]} = {cm:.0f} cm  [x]", True, 3))

    blf.size(HUD_FONT, 13 * scale)
    pad_x = 10 * scale
    height = 24 * scale
    gap = 3 * scale
    group_gap = 14 * scale
    widths = [blf.dimensions(HUD_FONT, it[1])[0] + 2 * pad_x for it in items]
    total = sum(widths)
    for prev, cur in zip(items, items[1:]):
        total += group_gap if prev[3] != cur[3] else gap

    x = max(10 * scale, (region.width - total) * 0.5)
    y = 14 * scale
    buttons = []
    prev_group = None
    for (bid, label, active, group), w in zip(items, widths):
        if prev_group is not None:
            x += group_gap if group != prev_group else gap
        buttons.append({"id": bid, "label": label, "active": active,
                        "rect": (x, y, x + w, y + height)})
        x += w
        prev_group = group
    return buttons


def hud_hit(buttons, px):
    for b in buttons:
        x0, y0, x1, y1 = b["rect"]
        if b["id"] is not None and x0 <= px.x <= x1 and y0 <= px.y <= y1:
            return b["id"]
    return None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_draw_handle = {"handle": None, "text_handle": None}


def _is_tool_area(op, context):
    area = context.area
    return area is not None and area.as_pointer() == getattr(op, "_area_ptr", None)


def _draw_world_lines(points, color, width):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": points})
    gpu.state.line_width_set(width)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_2d_lines(points, color, width=1.0):
    _draw_world_lines([(p[0], p[1], 0.0) for p in points], color, width)


def _draw_2d_rect(x0, y0, x1, y1, color):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    pts = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0),
           (x0, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
    batch = batch_for_shader(shader, 'TRIS', {"pos": pts})
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set('NONE')


def _draw_text(x, y, text, color, size, shadow=True):
    blf.size(HUD_FONT, size)
    if shadow:
        blf.color(HUD_FONT, 0.0, 0.0, 0.0, 0.85)
        blf.position(HUD_FONT, x + 1, y - 1, 0)
        blf.draw(HUD_FONT, text)
    blf.color(HUD_FONT, *color)
    blf.position(HUD_FONT, x, y, 0)
    blf.draw(HUD_FONT, text)


def _draw_callback(op, context):
    """World-space overlay: previews, hovered edges, drag axis guide."""
    if not _is_tool_area(op, context):
        return
    obj = context.active_object

    if op._preview_points:
        _draw_world_lines(op._preview_points, (0.1, 0.8, 1.0, 0.9), 2.0)

    # Hovered edge(s) — the whole group that a click on the midpoint would move
    if op._hover_edges and op._drag is None and obj is not None and obj.type == 'MESH':
        me = obj.data
        mw = obj.matrix_world
        pts = []
        for ei in op._hover_edges:
            if ei < len(me.edges):
                e = me.edges[ei]
                pts.append(mw @ me.vertices[e.vertices[0]].co)
                pts.append(mw @ me.vertices[e.vertices[1]].co)
        if pts:
            _draw_world_lines(pts, (1.0, 0.85, 0.2, 0.9), 4.0)

    # Drag axis guide through the moved vertex / edge midpoint
    drag = op._drag
    if drag is not None and drag.get("axis") is not None:
        anchor = drag["current_anchor"]
        axis = drag["axis"]
        a = Vector(anchor)
        b = Vector(anchor)
        a[axis] -= 1000.0
        b[axis] += 1000.0
        colors = {0: (1.0, 0.2, 0.2, 0.5), 1: (0.2, 1.0, 0.2, 0.5), 2: (0.3, 0.5, 1.0, 0.6)}
        _draw_world_lines([a, b], colors[axis], 1.0)


def _draw_pixel_callback(op, context):
    """Pixel-space overlay: snap marks, handles, labels, gesture hint, HUD."""
    if not _is_tool_area(op, context):
        return
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return
    view = View(region, rv3d)
    scale = context.preferences.system.ui_scale

    # Alignment (line snap) guide — dashed line from the reference vertex.
    snap_pt = op._preview_snap_point
    snap_line = op._preview_snap_line
    if snap_line is not None and snap_pt is not None:
        a = view.to_px(snap_line[0])
        b = view.to_px(snap_line[1])
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

    # Snap cross at the snapped vertex
    if snap_pt is not None:
        c = view.to_px(snap_pt)
        if c is not None:
            S = 8
            _draw_2d_lines([(c.x - S, c.y), (c.x + S, c.y), (c.x, c.y - S), (c.x, c.y + S)],
                           (1.0, 0.6, 0.1, 1.0), 2.0)

    # Hovered vertex ring
    if op._hover_vertex is not None and op._drag is None:
        c = view.to_px(op._hover_vertex[0])
        if c is not None:
            R = 9
            seg_count = 24
            pts = []
            for i in range(seg_count):
                a1 = (i / seg_count) * 2.0 * math.pi
                a2 = ((i + 1) / seg_count) * 2.0 * math.pi
                pts.append((c.x + R * math.cos(a1), c.y + R * math.sin(a1)))
                pts.append((c.x + R * math.cos(a2), c.y + R * math.sin(a2)))
            _draw_2d_lines(pts, (0.2, 1.0, 0.6, 1.0), 2.0)

    # Edge midpoint handle — faint hint near an edge, bright when grabbable
    if op._edge_handle is not None and op._drag is None:
        mid_world, active = op._edge_handle
        m = view.to_px(mid_world)
        if m is not None:
            R = 6.0 if active else 4.0
            c0, c1 = (m.x - R, m.y - R), (m.x + R, m.y - R)
            c2, c3 = (m.x + R, m.y + R), (m.x - R, m.y + R)
            color = (1.0, 0.85, 0.2, 1.0) if active else (1.0, 1.0, 1.0, 0.45)
            _draw_2d_lines([c0, c1, c1, c2, c2, c3, c3, c0], color, 2.0)

    # Dimension label
    dim = op._dim_info
    if dim:
        orig_mid, new_mid, cm_value, snapped = dim[:4]
        single_label = len(dim) > 4 and dim[4]
        a = view.to_px(orig_mid)
        b = view.to_px(new_mid)
        if a is not None and b is not None:
            if abs(cm_value - round(cm_value)) < 0.05:
                text = f"{int(round(cm_value))} cm"
            else:
                text = f"{cm_value:.1f} cm"
            if snapped:
                text = "[" + text + "]"
            color = (1.0, 0.75, 0.15, 1.0) if snapped else (0.4, 1.0, 1.0, 1.0)
            positions = ((a + b) * 0.5,) if single_label else (a, b)
            for pos in positions:
                _draw_text(pos.x + 6, pos.y + 10, text, color, 14)

    # "×N" — how many overlapping elements the action affects (ALL mode)
    mouse = op._last_mouse_px
    if op._count_label > 1 and mouse is not None:
        _draw_text(mouse.x + 16, mouse.y - 22, f"×{op._count_label}", (1.0, 0.85, 0.2, 1.0), 14)

    # RMB gesture hint
    g = op._gesture
    if g is not None:
        start, cur = g["start"], g["current"]
        _draw_2d_lines([start, cur], (1.0, 1.0, 1.0, 0.6), 2.0)
        delta = cur - start
        if delta.length >= op.GESTURE_PIXELS:
            label = "→ " + VIEW_LABEL[gesture_view(delta, detect_plane(rv3d).view_type)]
            color = (1.0, 1.0, 1.0, 1.0)
        else:
            label = "puść: wyłącz addon"
            color = (1.0, 0.6, 0.6, 1.0)
        _draw_text(cur.x + 14, cur.y + 8, label, color, 14)

    # HUD
    state = _get_state(context)
    plane = detect_plane(rv3d)
    for b in hud_layout(context, region, plane, state.depth_mode, op._pin):
        x0, y0, x1, y1 = b["rect"]
        if b["id"] is None:
            bg = (0.0, 0.0, 0.0, 0.45)
        elif b["id"] == "unpin":
            bg = (0.85, 0.5, 0.15, 0.95) if b["id"] != op._hud_hover else (0.95, 0.62, 0.25, 1.0)
        elif b["active"]:
            bg = (0.28, 0.45, 0.85, 0.95)
        elif b["id"] == op._hud_hover:
            bg = (0.34, 0.34, 0.34, 0.92)
        else:
            bg = (0.12, 0.12, 0.12, 0.8)
        _draw_2d_rect(x0, y0, x1, y1, bg)
        size = 13 * scale
        blf.size(HUD_FONT, size)
        tw = blf.dimensions(HUD_FONT, b["label"])[0]
        text_color = (0.8, 0.8, 0.8, 1.0) if b["id"] is None else (1.0, 1.0, 1.0, 1.0)
        _draw_text(x0 + (x1 - x0 - tw) * 0.5, y0 + (y1 - y0) * 0.5 - 4.5 * scale,
                   b["label"], text_color, size, shadow=False)


# ---------------------------------------------------------------------------
# Modal operator
# ---------------------------------------------------------------------------

NAV_EVENTS = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'N', 'T', 'HOME', 'ACCENT_GRAVE',
    'MOUSEPAN', 'MOUSEZOOM', 'MOUSEROTATE', 'MOUSESMARTZOOM', 'INBETWEEN_MOUSEMOVE',
    'WINDOW_DEACTIVATE',
}


class AXISEXTRUDE_OT_modal(bpy.types.Operator):
    bl_idname = "axis_extrude.modal"
    bl_label = "Axis Extrude (Modal)"
    bl_options = {'REGISTER'}

    VERTEX_GRAB_PIXELS = 12.0   # cursor this close to a vertex → click grabs it
    EDGE_GRAB_PIXELS = 10.0     # cursor this close to an edge midpoint → click grabs the edge
    EDGE_HINT_PIXELS = 40.0     # cursor this close to an edge → show its midpoint handle
    DRAG_START_PIXELS = 5.0     # dead zone before a press turns into a drag
    GESTURE_PIXELS = 40.0       # RMB drag at least this far → view switch instead of exit

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

        self._init_state()
        self._area_ptr = context.area.as_pointer()

        args = (self, context)
        _draw_handle["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, args, 'WINDOW', 'POST_VIEW'
        )
        _draw_handle["text_handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_pixel_callback, args, 'WINDOW', 'POST_PIXEL'
        )

        state.running = True
        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _init_state(self):
        # Preview
        self._preview_points = []
        self._preview_snap_point = None  # world pos of snapped vertex (for visual)
        self._preview_snap_line = None   # (ref_vertex_world, aligned_point_world) for line snap
        self._dim_info = None            # (orig_mid, new_mid, cm_value, snapped[, single_label])
        self._count_label = 0            # ×N label next to the cursor
        # Cursor / candidates
        self._area_ptr = None
        self._last_mouse_world = None
        self._last_mouse_px = None
        self._last_candidates = []
        self._cycle_offset = 0
        self._last_plane_key = None      # plane change → drop stale candidates / previews
        # X lock
        self._edge_locked = False
        self._locked_candidate = None
        self._locked_mode = None         # None | 'edge' | 'loopcut'
        self._locked_face_index = None
        self._locked_long_axis = None
        self._locked_plane_n = None
        # Vertex / edge move
        self._hover_vertex = None        # (world_pos, vert_index) if a click would grab a vertex
        self._hover_edge = None          # (edge_index, mid_world, move_axis) if a click would grab an edge
        self._hover_edges = []           # edges highlighted for the hovered edge (its group)
        self._edge_handle = None         # (mid_world, active) — midpoint handle to draw
        self._drag = None                # dict describing the element being moved, see _start_drag
        # 3D views
        self._pin = None                 # (depth_axis, depth) — wall picked by MMB double-click
        self._hud_hover = None           # id of HUD button under the cursor
        self._hud_press = False          # LMB press consumed by the HUD (swallow its release)
        self._gesture = None             # RMB gesture: {"start": px, "current": px}
        # Extrude press / hold (10 cm steps while held)
        self._lmb_press_time = 0.0
        self._lmb_press_pos = None
        self._lmb_active_extrude = False
        self._hold_candidate = None

    def _cancel(self, context):
        state = _get_state(context)
        state.running = False

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

        _remove_draw_handlers()
        if context.area:
            context.area.tag_redraw()

    # ---- context helpers ----------------------------------------------------

    def _get_view(self, context):
        area = context.area
        region = context.region
        if region is None or region.type != 'WINDOW':
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        space = area.spaces.active
        rv3d = getattr(space, "region_3d", None)
        return View(region, rv3d)

    def _scope(self, context, view):
        plane = detect_plane(view.rv3d)
        mode = _get_state(context).depth_mode
        pinned = None
        if (mode == 'NEAREST' and self._pin is not None and plane.aligned
                and self._pin[0] == plane.n):
            pinned = self._pin[1]
        return Scope(plane, mode, pinned)

    def _mouse_world(self, view, px, scope):
        depth = scope.pinned if scope.pinned is not None else 0.0
        return mouse_to_world(view, px, scope.plane, depth)

    def _in_viewport(self, context, event, view):
        """True when the cursor is over the 3D view itself — not over its
        header, side panels, navigation gizmo or another editor."""
        area = context.area
        region = view.region
        mx, my = event.mouse_x, event.mouse_y
        if not (area.x <= mx < area.x + area.width and area.y <= my < area.y + area.height):
            return False
        if not (region.x <= mx < region.x + region.width and region.y <= my < region.y + region.height):
            return False
        for r in area.regions:
            if r.type == 'WINDOW' or r.width <= 1 or r.height <= 1:
                continue
            if r.x <= mx < r.x + r.width and r.y <= my < r.y + r.height:
                return False
        return not self._over_nav_gizmo(context, view, mx - region.x, my - region.y)

    def _over_nav_gizmo(self, context, view, rx, ry):
        space = context.area.spaces.active
        if not (getattr(space, "show_gizmo", True) and getattr(space, "show_gizmo_navigate", True)):
            return False
        prefs = context.preferences
        scale = prefs.system.ui_scale
        size = getattr(prefs.view, "gizmo_size_navigate_v3d", 80) * scale
        region = view.region
        right = region.width
        area = context.area
        for r in area.regions:
            # N panel overlapping the viewport pushes the gizmo to the left
            if r.type == 'UI' and r.width > 1 and r.x >= area.x + area.width // 2:
                right = min(right, r.x - region.x)
        x0 = right - size - 30 * scale
        y0 = region.height - size - 160 * scale
        return x0 <= rx <= right and ry >= y0

    def _tag(self, context):
        if context.area:
            context.area.tag_redraw()

    # ---- state resets --------------------------------------------------------

    def _clear_hover(self):
        self._hover_vertex = None
        self._hover_edge = None
        self._hover_edges = []
        self._edge_handle = None
        self._count_label = 0

    def _clear_preview(self):
        self._preview_points = []
        self._preview_snap_point = None
        self._preview_snap_line = None
        self._dim_info = None
        self._count_label = 0

    def _unlock(self):
        self._edge_locked = False
        self._locked_candidate = None
        self._locked_mode = None
        self._locked_face_index = None
        self._locked_long_axis = None
        self._locked_plane_n = None

    def _reset_extrude(self):
        self._lmb_active_extrude = False
        self._hold_candidate = None
        self._clear_preview()

    def _refresh_on_plane_change(self, plane):
        key = (plane.n, plane.aligned, plane.view_type)
        if key == self._last_plane_key:
            return
        self._last_plane_key = key
        if self._edge_locked and self._locked_plane_n != plane.n:
            self._unlock()
        if self._drag is None and not self._lmb_active_extrude:
            self._last_candidates = []
            self._clear_hover()
            self._clear_preview()

    # ---- views ---------------------------------------------------------------

    def _switch_view(self, context, view, view_type):
        try:
            with context.temp_override(area=context.area, region=view.region):
                bpy.ops.view3d.view_axis(type=view_type)
        except Exception as exc:
            self.report({'WARNING'}, f"Nie udało się zmienić widoku: {exc}")
            return
        if self._pin is not None and self._pin[0] != VIEW_AXIS[view_type]:
            self._pin = None
        self._unlock()
        self._last_candidates = []
        self._clear_hover()
        self._clear_preview()

    def _align_to_wall(self, context, view, obj, px):
        """MMB double-click: look straight at the wall under the cursor and pin
        work to it; on empty space go back to top view."""
        coords = mesh_world_coords(obj)
        polys = [tuple(p.vertices) for p in obj.data.polygons]
        loc = normal = face_index = None
        if polys:
            origin = view3d_utils.region_2d_to_origin_3d(view.region, view.rv3d, px)
            direction = view3d_utils.region_2d_to_vector_3d(view.region, view.rv3d, px)
            if not view.rv3d.is_perspective:
                origin = origin - direction * 100000.0
            tree = BVHTree.FromPolygons(coords, polys)
            loc, normal, face_index, _dist = tree.ray_cast(origin, direction)

        if loc is None:
            self._pin = None
            self._switch_view(context, view, 'TOP')
            return

        plane = detect_plane(view.rv3d)
        found = view_for_face_normal(normal, plane.view_dir)
        if found is None:
            self.report({'WARNING'}, "Ta ściana nie jest równoległa do osi X / Y / Z")
            return
        view_type, n = found
        face = polys[face_index]
        depth = sum(coords[vi][n] for vi in face) / len(face)
        _get_state(context).depth_mode = 'NEAREST'
        self._switch_view(context, view, view_type)
        self._pin = (n, depth)

    def _hud_action(self, context, view, plane, hud_id):
        state = _get_state(context)
        if hud_id.startswith("view:"):
            self._switch_view(context, view, toggle_view(hud_id[5:], plane.view_type))
        elif hud_id == "mode:NEAREST":
            state.depth_mode = 'NEAREST'
        elif hud_id == "mode:ALL":
            state.depth_mode = 'ALL'
            self._pin = None
        elif hud_id == "unpin":
            self._pin = None
        self._unlock()
        self._last_candidates = []
        self._clear_hover()
        self._clear_preview()

    # ---- vertex / edge move -----------------------------------------------

    def _edge_group(self, obj, edge_index, scope, coords, for_extrude):
        """Edges an action on `edge_index` affects: itself (NEAREST) or
        everything forming the same line on screen (ALL). Extruding a boundary
        edge only takes boundary edges along — walls, not floor divisions."""
        if scope.mode != 'ALL':
            return [edge_index]
        counts = None
        if for_extrude:
            counts = edge_face_counts(obj.data)
            if counts[edge_index] > 1:
                counts = None
        return coincident_edge_group(obj, edge_index, scope.plane, coords, counts)

    def _find_grab_target(self, view, obj, target, scope, coords, candidates=None):
        """
        What a click at `target` would grab. Vertex and edge midpoint compete;
        the one nearer to the cursor on screen wins.
        Returns ('vertex', (world, index)) | ('edge', (edge_index, mid_world, move_axis)) | None.
        Also refreshes the midpoint handle hint (self._edge_handle).
        """
        self._edge_handle = None
        best = None  # (px, kind, payload)

        vert_hit = find_nearest_vertex(obj, target, scope, coords)
        if vert_hit is not None:
            d = pixel_distance(view, vert_hit[0], target)
            if d <= self.VERTEX_GRAB_PIXELS:
                best = (d, 'vertex', vert_hit)

        if candidates is None:
            candidates = find_edge_candidates(obj, target, scope, coords)
        if candidates:
            c = candidates[0]
            mid = edge_midpoint(obj, c["index"], coords)
            ppu = pixels_per_unit(view, target, scope.plane.u)
            if mid is not None and ppu is not None and c["dist"] * ppu <= self.EDGE_HINT_PIXELS:
                d = pixel_distance(view, mid, target)
                grab = d <= self.EDGE_GRAB_PIXELS and (best is None or d < best[0])
                if grab:
                    best = (d, 'edge', (c["index"], mid, c["extrude_axis"]))
                self._edge_handle = (mid, grab)

        if best is None:
            return None
        return best[1], best[2]

    def _update_hover(self, view, obj, target, scope, coords):
        self._clear_hover()
        if self._edge_locked or self._lmb_active_extrude:
            return
        hit = self._find_grab_target(view, obj, target, scope, coords, self._last_candidates)
        if hit is None:
            return
        kind, payload = hit
        if kind == 'vertex':
            self._hover_vertex = payload
            if scope.mode == 'ALL':
                self._count_label = len(coincident_verts(coords, payload[1], scope.plane))
        else:
            self._hover_edge = payload
            self._hover_edges = self._edge_group(obj, payload[0], scope, coords, for_extrude=False)
            if scope.mode == 'ALL':
                self._count_label = len(self._hover_edges)

    def _start_drag(self, px, obj, scope, coords, kind, payload):
        """
        Begin moving a vertex or an edge (with everything overlapping it in ALL mode).
          kind 'vertex': payload (world, vert_index) — axis follows cursor dominance
          kind 'edge':   payload (edge_index, mid_world, move_axis) — axis fixed,
                         perpendicular to the edge (parallel move)
        """
        me = obj.data
        if kind == 'vertex':
            vw, vi = payload
            if scope.mode == 'ALL':
                vert_ids = coincident_verts(coords, vi, scope.plane)
            else:
                vert_ids = [vi]
            anchor = Vector(vw)
            fixed_axis = None
        else:
            ei, mid, move_axis = payload
            edges = self._edge_group(obj, ei, scope, coords, for_extrude=False)
            vert_ids = sorted({vi for e in edges for vi in me.edges[e].vertices})
            anchor = Vector(mid)
            fixed_axis = move_axis
        self._drag = {
            "kind": kind,
            "verts": [(vi, Vector(coords[vi])) for vi in vert_ids],
            "anchor": anchor,                 # original vertex pos / edge midpoint
            "current_anchor": Vector(anchor),
            "fixed_axis": fixed_axis,
            "axis": fixed_axis,
            "plane": scope.plane,             # locked for the whole move
            "mode": 'press',                  # 'press' (LMB held) | 'pickup' (follows cursor)
            "press_px": Vector(px),
            "started": False,                 # True once cursor left the dead zone
        }
        count = len(vert_ids) if kind == 'vertex' else len(edges)
        self._clear_hover()
        self._clear_preview()
        self._count_label = count

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

    def _update_drag(self, context, view, obj, px, target, coords):
        """
        Move the dragged element along its axis — same rules as edge extrude:
          - snap to a vertex / alignment with a vertex line wins,
          - LMB held (drag) → 10 cm steps,
          - picked up by a quick click → free movement.
        Vertex axis is re-evaluated from the ORIGINAL position every frame, so
        the user switches axes just by moving the mouse the other way.
        """
        d = self._drag
        anchor = d["anchor"]
        plane = d["plane"]

        if d["mode"] == 'press' and not d["started"]:
            if (px - d["press_px"]).length < self.DRAG_START_PIXELS:
                return
            d["started"] = True

        axis = d["fixed_axis"]
        if axis is None:
            axis = plane.u if abs(target[plane.u] - anchor[plane.u]) >= abs(target[plane.v] - anchor[plane.v]) else plane.v
        d["axis"] = axis
        orig_c = anchor[axis]
        raw_c = target[axis]

        step_snapped = False
        snap = snap_axis_coord(view, obj, target, axis, plane, coords,
                               exclude_verts=[vi for vi, _w in d["verts"]])
        if snap is not None:
            new_c = snap[0]
        elif d["mode"] == 'press':
            snapped_delta, _cm = snap_distance_to_step(raw_c - orig_c, 10.0, context, min_cm=0.0)
            new_c = orig_c + snapped_delta
            step_snapped = True
        else:
            new_c = raw_c

        offset = Vector((0.0, 0.0, 0.0))
        offset[axis] = new_c - orig_c
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
        self._clear_hover()
        self._clear_preview()
        self._tag(context)

    def _commit_drag(self, context):
        kind = self._drag["kind"]
        bpy.ops.ed.undo_push(
            message="Axis Extrude: Move Edge" if kind == 'edge' else "Axis Extrude: Move Vertex")
        self._end_drag(context, restore=False)

    # ---- extrude / loop cut preview -----------------------------------------

    def _pick_candidate(self):
        """The locked candidate, or the offset-th from the sorted list."""
        if self._edge_locked and self._locked_candidate is not None:
            return self._locked_candidate
        if not self._last_candidates:
            return None
        return self._last_candidates[self._cycle_offset % len(self._last_candidates)]

    def _compute_extrude(self, context, view, obj, target, candidate, scope, coords, hold_active):
        """
        Final extrusion for `candidate` (and its group in ALL mode):
          snap to a vertex / vertex line wins; otherwise 10 cm steps while
          LMB is held, or the minimum distance on a quick click.
        Returns dict(edges, axis, orig_coord, final_coord, snap, step) or None.
        """
        me = obj.data
        ei = candidate["index"]
        if ei >= len(me.edges):
            return None
        axis = candidate["extrude_axis"]
        edges = self._edge_group(obj, ei, scope, coords, for_extrude=True)
        # The edges' own verts sit at the extrude origin — snapping to them
        # would collapse the extrusion, so they are excluded.
        exclude = {vi for e in edges for vi in me.edges[e].vertices}
        orig_c = coords[me.edges[ei].vertices[0]][axis]
        snap = snap_axis_coord(view, obj, target, axis, scope.plane, coords, exclude)

        state = _get_state(context)
        step = False
        if snap is not None:
            final_c = snap[0]
        else:
            final_c = target[axis]
            delta = final_c - orig_c
            if hold_active:
                min_cm = 10.0 if state.enforce_min_distance else 0.0
                snapped_delta, _cm = snap_distance_to_step(delta, 10.0, context, min_cm=min_cm)
                final_c = orig_c + snapped_delta
                step = True
            elif state.enforce_min_distance:
                min_dist = cm_to_blender_units(context, 10.0)
                if abs(delta) < min_dist:
                    final_c = orig_c + (-min_dist if delta < 0 else min_dist)
        return {"edges": edges, "axis": axis, "orig_coord": orig_c,
                "final_coord": final_c, "snap": snap, "step": step}

    def _loopcut_snap_target(self, view, obj, target, cut_axis, scope, coords):
        """Locked loop cut: vertex / alignment snap on the cut axis.
        Returns (adjusted_target, snap_or_None)."""
        snap = snap_axis_coord(view, obj, target, cut_axis, scope.plane, coords)
        adjusted = Vector(target)
        if snap is not None:
            adjusted[cut_axis] = snap[0]
        return adjusted, snap

    def _build_preview(self, context, view, obj, target, scope, coords):
        """Recompute preview lines, snap indicator and label for the cursor."""
        self._dim_info = None
        plane = scope.plane
        restrict = scope.mode == 'NEAREST'

        # Hover on a grabbable vertex / edge midpoint takes priority
        if (self._hover_vertex is not None or self._hover_edge is not None) and self._drag is None:
            self._preview_points = []
            self._preview_snap_point = None
            return

        # Locked loop cut — frozen face and axis, snap on the cut axis
        if self._edge_locked and self._locked_mode == 'loopcut':
            face_idx = self._locked_face_index
            cut_axis = self._locked_long_axis
            if face_idx is not None and cut_axis is not None:
                adjusted, snap = self._loopcut_snap_target(view, obj, target, cut_axis, scope, coords)
                pts = compute_loop_cut_preview(obj, face_idx, adjusted, plane, restrict, coords,
                                               forced_cut_axis=cut_axis)
                if not pts and snap is not None:
                    # Snapped onto the face boundary (no cut possible) — use raw cursor
                    snap = None
                    pts = compute_loop_cut_preview(obj, face_idx, target, plane, restrict, coords,
                                                   forced_cut_axis=cut_axis)
                if pts:
                    self._preview_points = pts
                    self._set_snap_preview(snap, pts)
                    return

        # Loop-cut preview when the cursor is inside a face
        if not self._edge_locked and not self._lmb_active_extrude:
            face_idx = find_face_at_point(obj, target, scope, coords)
            if face_idx is not None:
                pts = compute_loop_cut_preview(obj, face_idx, target, plane, restrict, coords)
                if pts:
                    self._preview_points = pts
                    self._set_snap_preview(None, None)
                    return

        if self._lmb_active_extrude and self._hold_candidate is not None:
            candidate = self._hold_candidate
        else:
            candidate = self._pick_candidate()
        if candidate is None:
            self._clear_preview()
            return

        hold_active = False
        if self._lmb_active_extrude:
            held = time.time() - self._lmb_press_time
            moved = (target - self._lmb_press_pos).length if self._lmb_press_pos else 0.0
            hold_active = held >= 0.15 or moved >= cm_to_blender_units(context, 1.0)

        ex = self._compute_extrude(context, view, obj, target, candidate, scope, coords, hold_active)
        if ex is None:
            self._clear_preview()
            return

        me = obj.data
        axis = ex["axis"]
        final_c = ex["final_coord"]
        pts = []
        main_old = main_new = None
        for ei in ex["edges"]:
            e = me.edges[ei]
            a = coords[e.vertices[0]]
            b = coords[e.vertices[1]]
            na = Vector(a)
            nb = Vector(b)
            na[axis] = final_c
            nb[axis] = final_c
            pts += [a, na, b, nb, na, nb]
            if ei == candidate["index"]:
                main_old, main_new = (a, b), (na, nb)

        self._preview_points = pts
        self._set_snap_preview(ex["snap"], list(main_new))
        cm_value = blender_units_to_cm(context, abs(final_c - ex["orig_coord"]))
        self._dim_info = ((main_old[0] + main_old[1]) * 0.5, (main_new[0] + main_new[1]) * 0.5,
                          cm_value, ex["step"])
        self._count_label = len(ex["edges"])

    # ---- event handlers ------------------------------------------------------

    def _toggle_lock(self, context, view, obj, scope):
        if self._drag is not None:
            return
        if self._edge_locked:
            self._unlock()
        else:
            target = self._last_mouse_world
            coords = mesh_world_coords(obj)
            plane = scope.plane
            if target is not None:
                # Prefer locking the face (loop cut mode) if cursor is inside one
                face_idx = find_face_at_point(obj, target, scope, coords)
                if face_idx is not None:
                    info = determine_cut_axis_from_face(obj, face_idx, target, plane, coords)
                    if info is not None:
                        self._edge_locked = True
                        self._locked_mode = 'loopcut'
                        self._locked_face_index = face_idx
                        self._locked_long_axis = info[0]
                        self._locked_plane_n = plane.n
            if not self._edge_locked:
                current = self._pick_candidate()
                if current is not None:
                    self._edge_locked = True
                    self._locked_mode = 'edge'
                    self._locked_candidate = dict(current)
                    self._locked_plane_n = plane.n
        if self._last_mouse_world is not None:
            coords = mesh_world_coords(obj)
            self._update_hover(view, obj, self._last_mouse_world, scope, coords)
            self._build_preview(context, view, obj, self._last_mouse_world, scope, coords)
        self._tag(context)

    def _handle_rmb(self, context, view, plane, event, px, in_view):
        if event.value == 'PRESS':
            # Moving a vertex/edge → cancel the move (restore) instead of quitting
            if self._drag is not None:
                self._end_drag(context, restore=True)
                return {'RUNNING_MODAL'}
            if not in_view:
                return {'PASS_THROUGH'}
            if self._lmb_active_extrude:
                self._reset_extrude()
            self._gesture = {"start": Vector(px), "current": Vector(px)}
            self._clear_hover()
            self._clear_preview()
            self._tag(context)
            return {'RUNNING_MODAL'}

        if event.value == 'RELEASE':
            if self._gesture is None:
                return {'PASS_THROUGH'}
            delta = px - self._gesture["start"]
            self._gesture = None
            if delta.length >= self.GESTURE_PIXELS:
                self._switch_view(context, view, gesture_view(delta, plane.view_type))
                self._tag(context)
                return {'RUNNING_MODAL'}
            # Plain right-click — exit, as before
            self._cancel(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'} if self._gesture is not None else {'PASS_THROUGH'}

    def _handle_mousemove(self, context, view, obj, scope, px, in_view):
        self._last_mouse_px = px
        if self._gesture is not None:
            self._gesture["current"] = Vector(px)
            self._tag(context)
            return {'RUNNING_MODAL'}

        mode = _get_state(context).depth_mode
        if in_view:
            self._hud_hover = hud_hit(hud_layout(context, view.region, scope.plane, mode, self._pin), px)
        else:
            self._hud_hover = None

        target = self._mouse_world(view, px, scope)
        if target is None:
            self._tag(context)
            return {'PASS_THROUGH'}
        self._last_mouse_world = target
        coords = mesh_world_coords(obj)

        if self._drag is not None:
            self._update_drag(context, view, obj, px, target, coords)
            self._tag(context)
            return {'RUNNING_MODAL'}

        if (not in_view or self._hud_hover is not None) and not self._lmb_active_extrude:
            self._clear_hover()
            self._clear_preview()
            self._tag(context)
            return {'PASS_THROUGH'}

        # When locked, keep the candidate frozen — only refresh preview
        if not self._edge_locked:
            self._last_candidates = find_edge_candidates(obj, target, scope, coords)
            self._cycle_offset = 0
        self._update_hover(view, obj, target, scope, coords)
        self._build_preview(context, view, obj, target, scope, coords)
        self._tag(context)
        return {'PASS_THROUGH'}

    def _handle_lmb_press(self, context, view, obj, scope, px, in_view):
        # Picked-up vertex/edge — this click drops it (commits the move)
        if self._drag is not None:
            target = self._mouse_world(view, px, scope)
            if target is not None:
                self._update_drag(context, view, obj, px, target, mesh_world_coords(obj))
            self._commit_drag(context)
            return {'RUNNING_MODAL'}

        if not in_view:
            return {'PASS_THROUGH'}

        mode = _get_state(context).depth_mode
        hud_id = hud_hit(hud_layout(context, view.region, scope.plane, mode, self._pin), px)
        if hud_id is not None:
            self._hud_press = True
            self._hud_action(context, view, scope.plane, hud_id)
            self._tag(context)
            return {'RUNNING_MODAL'}

        target = self._mouse_world(view, px, scope)
        if target is None:
            return {'RUNNING_MODAL'}
        coords = mesh_world_coords(obj)
        plane = scope.plane
        restrict = scope.mode == 'NEAREST'

        # Press on a vertex / edge midpoint — start moving it
        # (drag, or quick click = pick up)
        if not self._edge_locked:
            hit = self._find_grab_target(view, obj, target, scope, coords)
            if hit is not None:
                self._start_drag(px, obj, scope, coords, *hit)
                self._tag(context)
                return {'RUNNING_MODAL'}

        # Locked loop cut — frozen face + axis, with snap
        if self._edge_locked and self._locked_mode == 'loopcut':
            face_idx = self._locked_face_index
            cut_axis = self._locked_long_axis
            if face_idx is not None and cut_axis is not None:
                adjusted, snap = self._loopcut_snap_target(view, obj, target, cut_axis, scope, coords)
                ok = loop_cut_face_at_point(obj, face_idx, adjusted, plane, restrict, coords,
                                            forced_cut_axis=cut_axis)
                if not ok and snap is not None:
                    # Snapped onto the face boundary — cut at the raw cursor instead
                    ok = loop_cut_face_at_point(obj, face_idx, target, plane, restrict, coords,
                                                forced_cut_axis=cut_axis)
                if ok:
                    bpy.ops.ed.undo_push(message="Axis Extrude: Loop Cut")
                else:
                    self.report({'WARNING'}, "Loop cut failed")
            self._unlock()
            self._last_candidates = []
            self._clear_preview()
            self._tag(context)
            return {'RUNNING_MODAL'}

        # Cursor inside a face (not locked) → loop cut
        if not self._edge_locked:
            face_idx = find_face_at_point(obj, target, scope, coords)
            if face_idx is not None:
                ok = loop_cut_face_at_point(obj, face_idx, target, plane, restrict, coords)
                if ok:
                    bpy.ops.ed.undo_push(message="Axis Extrude: Loop Cut")
                else:
                    self.report({'WARNING'}, "Loop cut failed")
                self._last_candidates = []
                self._clear_preview()
                self._tag(context)
                return {'RUNNING_MODAL'}
            self._last_candidates = find_edge_candidates(obj, target, scope, coords)
            self._cycle_offset = 0
            self._last_mouse_world = target

        candidate = self._pick_candidate()
        if candidate is None:
            self.report({'WARNING'}, "No edge found")
            return {'RUNNING_MODAL'}

        # Enter "extrude press" state — RELEASE commits.
        # Holding / dragging switches the preview to 10 cm steps.
        self._lmb_active_extrude = True
        self._lmb_press_time = time.time()
        self._lmb_press_pos = Vector(target)
        self._hold_candidate = dict(candidate)
        self._build_preview(context, view, obj, target, scope, coords)
        self._tag(context)
        return {'RUNNING_MODAL'}

    def _handle_lmb_release(self, context, view, obj, scope, px):
        if self._hud_press:
            self._hud_press = False
            return {'RUNNING_MODAL'}

        # Vertex/edge press: dragged → commit; released in place → pick it up
        if self._drag is not None and self._drag["mode"] == 'press':
            if self._drag["started"]:
                self._commit_drag(context)
            else:
                self._drag["mode"] = 'pickup'
            return {'RUNNING_MODAL'}

        if not self._lmb_active_extrude:
            return {'PASS_THROUGH'}

        target = self._mouse_world(view, px, scope)
        candidate = self._hold_candidate
        if target is None or candidate is None:
            self._reset_extrude()
            self._tag(context)
            return {'RUNNING_MODAL'}

        # Quick click: release within 150 ms without meaningful movement
        held = time.time() - self._lmb_press_time
        moved = (target - self._lmb_press_pos).length if self._lmb_press_pos else 0.0
        is_quick_click = held < 0.15 and moved < cm_to_blender_units(context, 1.0)

        coords = mesh_world_coords(obj)
        ex = self._compute_extrude(context, view, obj, target, candidate, scope, coords,
                                   hold_active=not is_quick_click)
        ok = ex is not None and extrude_edges_to_coord(obj, ex["edges"], ex["axis"], ex["final_coord"])
        if ok:
            bpy.ops.ed.undo_push(message="Axis Extrude: Extrude")
        else:
            self.report({'WARNING'}, "Extrude failed")

        # Reset state — auto-unlock after click
        self._unlock()
        self._reset_extrude()
        self._last_candidates = []
        self._cycle_offset = 0
        self._tag(context)
        return {'RUNNING_MODAL'}

    # ---- modal loop --------------------------------------------------------

    def modal(self, context, event):
        state = _get_state(context)

        # User turned it off from the panel
        if not state.running:
            self._cancel(context)
            return {'CANCELLED'}

        obj = context.active_object
        if obj is None or obj.type != 'MESH' or context.area is None:
            self._cancel(context)
            return {'CANCELLED'}

        self._area_ptr = context.area.as_pointer()
        view = self._get_view(context)
        if view.region is None or view.rv3d is None:
            return {'PASS_THROUGH'}

        scope = self._scope(context, view)
        self._refresh_on_plane_change(scope.plane)
        px = Vector((event.mouse_x - view.region.x, event.mouse_y - view.region.y))
        in_view = self._in_viewport(context, event, view)
        etype, evalue = event.type, event.value

        if etype == 'ESC' and evalue == 'PRESS':
            if self._drag is not None:
                self._end_drag(context, restore=True)
                return {'RUNNING_MODAL'}
            self._cancel(context)
            return {'CANCELLED'}

        if etype == 'RIGHTMOUSE':
            return self._handle_rmb(context, view, scope.plane, event, px, in_view)

        # MMB double-click — look straight at the wall under the cursor
        if etype == 'MIDDLEMOUSE' and evalue == 'DOUBLE_CLICK' and in_view:
            if self._drag is None:
                self._align_to_wall(context, view, obj, px)
                self._tag(context)
            return {'RUNNING_MODAL'}

        if etype == 'X' and evalue == 'PRESS' and in_view:
            self._toggle_lock(context, view, obj, scope)
            return {'RUNNING_MODAL'}

        # TIMER — refresh preview during LMB hold so the 10 cm snap kicks in after a delay
        if etype == 'TIMER':
            if self._lmb_active_extrude and self._last_mouse_world is not None:
                self._build_preview(context, view, obj, self._last_mouse_world, scope,
                                    mesh_world_coords(obj))
                self._tag(context)
            return {'PASS_THROUGH'}

        if etype == 'MOUSEMOVE':
            return self._handle_mousemove(context, view, obj, scope, px, in_view)

        # A fast second click arrives as DOUBLE_CLICK — treat it as a normal press
        if etype == 'LEFTMOUSE' and evalue in {'PRESS', 'DOUBLE_CLICK'}:
            return self._handle_lmb_press(context, view, obj, scope, px, in_view)

        if etype == 'LEFTMOUSE' and evalue == 'RELEASE':
            return self._handle_lmb_release(context, view, obj, scope, px)

        # Keys over panels / other editors, and view navigation, go to Blender
        if not in_view:
            return {'PASS_THROUGH'}
        if etype in NAV_EVENTS or etype.startswith(('NUMPAD_', 'NDOF_', 'TRACKPAD')):
            return {'PASS_THROUGH'}
        return {'RUNNING_MODAL'}


def _remove_draw_handlers():
    if _draw_handle["handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle["handle"], 'WINDOW')
        except Exception:
            pass
        _draw_handle["handle"] = None
    if _draw_handle["text_handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle["text_handle"], 'WINDOW')
        except Exception:
            pass
        _draw_handle["text_handle"] = None


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
        layout.label(text="Axis Extrude — click-to-extrude tool for plans, elevations and sections.")
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
        layout.label(text="Overlapping elements:")
        layout.row().prop(state, "depth_mode", expand=True)

        layout.separator()
        box = layout.box()
        box.label(text="How it works:")
        box.label(text="• Top (XY), front (XZ), side (YZ) views")
        box.label(text="• Move mouse — nearest edge highlights")
        box.label(text="• Click — extrude to cursor")
        box.label(text="• Press X — lock/unlock selection")
        box.label(text="• Click inside a face — loop cut")
        box.label(text="• Drag a vertex — move on axis (10 cm steps)")
        box.label(text="• Click a vertex — pick up, click to drop")
        box.label(text="• Drag edge midpoint (square) — move edge")
        box.label(text="• Snaps to nearby vertex")
        box.label(text="• Aligns with vertex lines (dashed)")
        box.label(text="• Bottom buttons — view / depth mode")
        box.label(text="• RMB drag ↑↓←→ — switch view")
        box.label(text="• MMB double-click wall — face it")
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
    depth_mode: bpy.props.EnumProperty(
        name="Elementy",
        description="Na których elementach działać, gdy w widoku pokrywają się ze sobą",
        items=(
            ('NEAREST', "Najbliższe",
             "Tylko element najbliższy widokowi (albo wskazana dwuklikiem MMB ściana)"),
            ('ALL', "Wszystkie",
             "Wszystkie elementy, które w widoku pokrywają się ze sobą"),
        ),
        default='NEAREST',
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
    _remove_draw_handlers()
    del bpy.types.WindowManager.axis_extrude_state
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
