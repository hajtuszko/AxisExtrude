bl_info = {
    "name": "Axis Extrude",
    "author": "Axis Extrude",
    "version": (1, 0, 0),
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


def extrude_edge_to_point(obj, edge_index, extrude_axis, target_world):
    """
    Extrude the given edge and snap the new verts so the extruded quad extends
    to `target_world` along `extrude_axis` (world axis). Other axis is locked
    to the original edge's value (snap-to-world).
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

    # Extrude the edge
    ret = bmesh.ops.extrude_edge_only(bm, edges=[edge])
    new_geom = ret['geom']
    new_verts = [g for g in new_geom if isinstance(g, bmesh.types.BMVert)]

    if len(new_verts) != 2:
        bm.free()
        return False

    # Match new verts to original verts by proximity (extrude_edge_only puts
    # them in the same position as the originals initially)
    orig_verts = list(edge.verts)
    # Build pairing: new_vert -> original world co
    pairs = []
    for nv in new_verts:
        nv_world = mw @ nv.co
        # Find closest original
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
        """Return the currently selected candidate from the cached list, or None."""
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

    # ---- modal loop --------------------------------------------------------

    def modal(self, context, event):
        state = _get_state(context)

        # User turned it off from the panel
        if not state.running:
            self._cancel(context)
            return {'CANCELLED'}

        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            self._cancel(context)
            return {'CANCELLED'}

        if event.type == 'ESC' and event.value == 'PRESS':
            self._cancel(context)
            return {'CANCELLED'}

        # Cycle edge candidate with X
        if event.type == 'X' and event.value == 'PRESS':
            if self._last_candidates:
                self._cycle_offset = (self._cycle_offset + 1) % len(self._last_candidates)
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
                # Rebuild candidate list on movement; reset cycle offset
                self._last_candidates = find_edge_candidates(obj, target)
                self._cycle_offset = 0
                self._last_mouse_world = target
                self._build_preview(context, obj, target)
                if context.area:
                    context.area.tag_redraw()

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            target = mouse_to_world_xy(context, event)
            obj = context.active_object
            if target is None or obj is None:
                return {'RUNNING_MODAL'}

            # Make sure candidate list is fresh
            if not self._last_candidates or self._last_mouse_world != target:
                self._last_candidates = find_edge_candidates(obj, target)
                # Keep cycle_offset — user may have pressed X before click
                if self._cycle_offset >= len(self._last_candidates):
                    self._cycle_offset = 0
                self._last_mouse_world = target

            candidate = self._pick_candidate()
            if candidate is None:
                self.report({'WARNING'}, "No edge found")
                return {'RUNNING_MODAL'}

            final_target, _snap = self._resolve_target(
                context, obj, target, candidate
            )

            ok = extrude_edge_to_point(
                obj, candidate["index"], candidate["extrude_axis"], final_target
            )
            if not ok:
                self.report({'WARNING'}, "Extrude failed")

            self._preview_points = []
            self._preview_snap_point = None
            self._last_candidates = []
            self._cycle_offset = 0
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
        box = layout.box()
        box.label(text="How it works:")
        box.label(text="• Top view (XY) 2D workflow")
        box.label(text="• Move mouse — nearest edge highlights")
        box.label(text="• Click — extrude to cursor")
        box.label(text="• Press X — cycle edge candidate")
        box.label(text="• Snaps to nearby vertex on extrude axis")
        box.label(text="• Right-click — disable")


# ---------------------------------------------------------------------------
# State PropertyGroup
# ---------------------------------------------------------------------------

class AXISEXTRUDE_state(bpy.types.PropertyGroup):
    running: bpy.props.BoolProperty(
        name="Running",
        default=False,
    )


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

classes = (
    AXISEXTRUDE_state,
    AXISEXTRUDE_OT_modal,
    AXISEXTRUDE_OT_toggle,
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
