import bpy
import os
import sys
import json
import time
import random
import numpy as np
from random import sample

import bpy, bpy_extras
import glob

def clear_scene():
    '''Clear existing objects in scene'''
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def set_viewport_shading(mode):
    '''Makes color/texture viewable in viewport'''
    areas = bpy.context.workspace.screens[0].areas
    for area in areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = mode

def add_camera_light():
    bpy.ops.object.light_add(
            type='SPOT', 
            location=(4.2,22.413,26.948), 
            rotation=(np.radians(-21.3), np.radians(40.8), np.radians(48.7))
            )
    bpy.context.object.data.energy = 4000
    bpy.ops.object.camera_add(
            location=(35.268, 20.493, 15.886), 
            rotation=(np.radians(76.78),0,np.radians(-240))
            )
    bpy.context.scene.camera = bpy.context.object
    return bpy.context.object

def set_render_settings(engine, render_size, generate_masks=True):
    scene = bpy.context.scene
    scene.world.color = (1, 1, 1)
    scene.render.resolution_percentage = 100
    scene.render.engine = engine
    render_width, render_height = render_size
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.use_nodes = True
    scene.render.image_settings.file_format='PNG'
    scene.view_settings.exposure = 3.25
    if engine == "BLENDER_EEVEE":
        scene.eevee.taa_samples = 1
        scene.eevee.taa_render_samples = 1
    elif engine == 'CYCLES':   
        scene.render.image_settings.file_format='PNG'
        # Some settings to make the rendering go faster, can comment out 
        scene.cycles.samples = 128
        scene.cycles.max_bounces = 1
        scene.cycles.min_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transmission_bounces = 1
        scene.cycles.volume_bounces = 1
        scene.cycles.transparent_max_bounces = 1
        scene.cycles.transparent_min_bounces = 1
        scene.view_layers["View Layer"].use_pass_object_index = True
        scene.render.tile_x = 16
        scene.render.tile_y = 16

def colorize(obj, color, metallic=0, roughness=1):
    '''Add color to object'''
    if '%sColor'%obj.name in bpy.data.materials:
        mat = bpy.data.materials['%sColor'%obj.name]
    else:
        mat = bpy.data.materials.new(name="%sColor"%obj.name)
        mat.use_nodes = True
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = color # color
    mat.node_tree.nodes["Principled BSDF"].inputs[4].default_value = metallic # roughness
    mat.node_tree.nodes["Principled BSDF"].inputs[7].default_value = roughness # roughness
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    set_viewport_shading('MATERIAL')

def load_grasp(path_to_obj, ind):
    bpy.ops.import_scene.obj(filepath=path_to_obj)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    # ['round_pad.obj', 'round_pad.obj:AV3RVMMTY1NP',
    #         'Spot', 'test_36.obj', 'test_36.obj:AV0BXHCQA1BP',
    #         'test_36.obj:AV1QSROAZYTC', 'test_36.obj:AV20MLUXD1PG',
    #         'test_36.obj:AV2K9B63Z3EP']

    base_obj = bpy.data.objects['base.stl']
    colorize(base_obj, (0.2,0.2,0.2,1))
    for obj_name in bpy.data.objects.keys():
        if obj_name[:len("round_pad.obj")] == "round_pad.obj":
            # round pad backing
            obj = bpy.data.objects[obj_name]
            colorize(obj, (0.5,0.5,0.5,1))
        elif obj_name == f"test_{ind}.obj":
            # target object
            obj = bpy.data.objects[obj_name]
            colorize(obj, (0.0,0.0,0.0,1), roughness=0.0)
            # # Uncomment if you want blue-colored meshes
            # colorize(obj, (0.2,0.6,1.0,1), roughness=0.0)
        elif obj_name[:len(f"test_{ind}.obj")] == f"test_{ind}.obj":
            obj = bpy.data.objects[obj_name]
            if obj.dimensions[1] > 2: # heuristic for checking if backing or deformable
                colorize(obj, (0.3,0.3,0.3,1), metallic=0.9)
            else:
                colorize(obj, (0.5,0.5,0.5,1))

    bpy.ops.mesh.primitive_plane_add(size=30)

def render_grasp(ind, obj_path, image_path, eevee=False):
    clear_scene()
    camera = add_camera_light()
    render_size = (640,640)
    if eevee:
        # faster, lighter-weight rendering engine
        set_render_settings('BLENDER_EEVEE', render_size) 
    else:
        set_render_settings('CYCLES', render_size) # slower

    load_grasp(obj_path, ind)
    os.makedirs(os.path.join(outdir, 'images'), exist_ok=True)
    bpy.context.scene.render.filepath = os.path.join(image_path, "%05d.jpg"%(ind))
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    import sys
    outdir = sys.argv[4]

    grasp_dir = os.path.join(outdir, 'objs')
    image_dir = os.path.join(outdir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    for obj_path in glob.glob(os.path.join(grasp_dir, '*.obj')):
        ind = int(os.path.basename(obj_path).split('.')[0])
        render_grasp(ind, obj_path, image_dir)
