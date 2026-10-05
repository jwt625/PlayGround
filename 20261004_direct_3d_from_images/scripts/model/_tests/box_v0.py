"""Pipeline test model: a solid box with the measured case rim size (97.5 x 201.7 x 35 mm)."""
import bpy

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.0175))
o = bpy.context.active_object
o.name = "case_box"
o.scale = (0.2017, 0.0975, 0.035)
bpy.ops.object.transform_apply(scale=True)
