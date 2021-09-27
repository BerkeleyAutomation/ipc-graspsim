import sys
import trimesh
import pandas as pd
import numpy as np
import subprocess, glob, os, json
from autolab_core import YamlConfig
from pose_rv import GaussianPoseRV

masses = {
    'bar_clamp': 18.5, 'book': 34.8, 'bowl': 15.7, 'cat': 9.9, 'cube_3cm': 3.0, 'endstop_holder': 16.3,
    'engine_part': 28.6, 'fan_extruder': 7.4, 'gearbox': 7.4, 'large_marker': 3.2, 'mount1': 10.4, 
    'mug': 20.2, 'nozzle': 6.3, 'part1': 6.8, 'part3': 13.6, 'pawn': 18.6, 'pear': 6.5, 'pipe_connector': 23.5, 
    'sardines': 11.2, 'sulfur_neutron': 7.0, 'vase': 13.1, 'yoda': 20.4
    }

# Nifty helper; in sim, y defined normal to ground, but in our original data z is normal to ground
yz_swap = trimesh.transformations.rotation_matrix(-np.pi/2, (1, 0, 0))

def get_grasp_tfs(obj_name, grasp_ind):
    basedir = os.path.join(os.path.dirname(__file__), '../../dexgrasp_data')

    # Get grasp information (gripper pose info, object pose info)
    with open(os.path.join(basedir, 'phys_grasps_json/%s.json' % obj_name)) as foo:
        obj_cfg = json.load(foo)
    w2g = obj_cfg[grasp_ind*5]['grasp']
    w2o = obj_cfg[grasp_ind*5]['pose']

    # Introduce pose noise to gripper position
    sigma_trans, sigma_rot = 0.001, 0.003
    w2g = np.array(w2g) @ GaussianPoseRV(sigma_trans, sigma_rot).generate().matrix
    
    # Create list of transformations to keep track of gripper positions(0, 1) and object position (2)
    tf_lst = [np.eye(4)] * 3

    # From gripper pose, re-generate tooltip pose
    yumi_cfg = YamlConfig(os.path.join(basedir, 'yumi_meshes/yumi_metal_spline.yaml'))
    for i in range(2):
        g2t = np.eye(4)
        g2t[:3, :3] = np.array(yumi_cfg['params']['tooltip_poses'][i]['params']['rotation'])
        g2t[:3, 3] = np.array(yumi_cfg['params']['tooltip_poses'][i]['params']['translation'])
        tf_lst[i] = g2t @ tf_lst[i]

    # Apply initial world-to-gripper / tooltip transformations
    tf_lst[0] = yz_swap @ w2g @ tf_lst[0]
    tf_lst[1] = yz_swap @ w2g @ tf_lst[1]
    tf_lst[2] = yz_swap @ w2o @ tf_lst[2]

    # to center and put object above ground (1)
    obj = trimesh.load(os.path.join(basedir, 'object_meshes/%s.obj' % obj_name))
    obj.vertices = trimesh.transform_points(obj.vertices, tf_lst[2])

    obj_shift = np.eye(4); obj_shift[:3, 3] = -obj.centroid
    obj.vertices = trimesh.transform_points(obj.vertices, obj_shift)
    tf_lst[0] = obj_shift @ tf_lst[0]
    tf_lst[1] = obj_shift @ tf_lst[1]
    tf_lst[2] = obj_shift @ tf_lst[2]

    # to center and put object above ground (2)
    obj_shift = np.eye(4); obj_shift[1, 3] -= (obj.bounds[0, 1] - 0.001)
    obj.vertices = trimesh.transform_points(obj.vertices, obj_shift)
    tf_lst[0] = obj_shift @ tf_lst[0]
    tf_lst[1] = obj_shift @ tf_lst[1]
    tf_lst[2] = obj_shift @ tf_lst[2]

    # check for collisions; if collisions, open up grippers more until no collision.
    mat = [trimesh.load(os.path.join(basedir, 'yumi_meshes/round_pad.obj')) for _ in range(2)]
    if args.sharp:
        bound_box = trimesh.creation.box(mat[0].extents) 
        bound_box.vertices += (mat[0].centroid - bound_box.centroid)
        mat = [bound_box.copy() for _ in mat]
    mat[0].vertices = trimesh.transform_points(mat[0].vertices, tf_lst[0])
    mat[1].vertices = trimesh.transform_points(mat[1].vertices, tf_lst[1])

    obj = trimesh.load(os.path.join(basedir, 'object_meshes/%s.obj' % obj_name))
    obj.vertices = trimesh.transform_points(obj.vertices, tf_lst[2])

    eps = 0.001
    colman = trimesh.collision.CollisionManager()
    colman.add_object('t_0', mat[0])
    colman.add_object('t_1', mat[1])
    while colman.min_distance_single(obj) < 0.001:
        mat[0].vertices -= eps*tf_lst[0][:3, 2]
        mat[1].vertices -= eps*tf_lst[1][:3, 2]
        tf_lst[0] = trimesh.transformations.translation_matrix(-eps*tf_lst[0][:3, 2]) @ tf_lst[0]
        tf_lst[1] = trimesh.transformations.translation_matrix(-eps*tf_lst[1][:3, 2]) @ tf_lst[1]
        colman = trimesh.collision.CollisionManager()
        colman.add_object('t_0', mat[0])
        colman.add_object('t_1', mat[1])
    return tf_lst

def get_ipc_input(obj_name, tf_lst, output_basedir, duration, step, E, contact_mu, args):
    from datetime import datetime
    if output_basedir is not None:
        output_dir = os.path.join("output", output_basedir)
    else:
        output_dir = "output"
    output_dir = os.path.join(output_dir, "%s_%d_%s" % (
        obj_name, 
        args.grasp_ind, 
        datetime.now().isoformat()
        )); 
    os.makedirs(output_dir, exist_ok=True)

    pad_distance = np.linalg.norm(100 * tf_lst[0][:3, 3] - 100 * tf_lst[1][:3, 3])
    speed = max(1, (pad_distance)/2/duration)

    # everything is scaled up by 100.
    config = os.path.join(output_dir, 'grasp_config.txt')
    basedir = os.path.join(os.path.dirname(__file__), '../../dexgrasp_data')
    obj = trimesh.load(os.path.join(basedir, 'object_meshes/%s.obj' % obj_name))
    with open(config, 'w') as f:
        f.write("script grasp\n")
        f.write("shapes input 5\n")

        # Insert object into simulation scene
        f.write("%s/ipc_msh/%s.msh %f %f %f %f %f %f %f %f %f %f %f %f 100 100 100 material %d 1e10 0.3\n" % (
            basedir, 
            obj_name, 
            *(100 * tf_lst[2][:3, 3]), 
            *tf_lst[2][:3, :3].flatten(), 
            (masses[obj_name]/1000) / obj.volume
            ))

        # Insert gripper pads into simulation scene
        SCALING_FACTOR=80
        scaling = np.ones(3)
        pad_dir = os.path.join(basedir, 'yumi_meshes')
        vel_dir = np.zeros((2, 3))
        from copy import deepcopy
        orig_tf_lst = deepcopy(tf_lst)

        if args.sharp:
            pad_name = os.path.join(pad_dir, 'mat20x20.msh')

            # get "rectangular" jaws via filling out the extents
            round_pad = trimesh.load(os.path.join(basedir, 'yumi_meshes/round_pad.obj'))
            round_dims = round_pad.extents.copy()
            round_dims[1], round_dims[2] = round_dims[2], round_dims[1]

            sharp_mat = trimesh.load(os.path.join(basedir, 'yumi_meshes/mat20x20.obj'))
            sharp_dims = sharp_mat.extents
            scaling = np.array(round_dims) / np.array(sharp_dims)

            shift = np.array([round_pad.bounds[0, 0] - sharp_mat.bounds[0, 0]*scaling[0], 0, 0])

            tf_lst[0] = tf_lst[0] @ trimesh.transformations.translation_matrix(shift) @ yz_swap 
            tf_lst[1] = tf_lst[1] @ trimesh.transformations.translation_matrix(shift) @ yz_swap 

            # DBC constraint logic (select back surface to move w/ const vel)
            DBC_select = "-3 1 -3 3 1 3"

            # pad move directions
            vel_dir[0] = -tf_lst[0][:3, 1]
            vel_dir[1] = -tf_lst[1][:3, 1]
        else:
            pad_name = os.path.join(pad_dir, 'round_pad.msh')

            # pad move directions
            vel_dir[0] =  tf_lst[0][:3, 2]
            vel_dir[1] =  tf_lst[1][:3, 2]

            # DBC constraint logic (select back surface to move w/ const vel)
            DBC_select = "-3 -3 0 3 3 0"

        # Pad density + poisson ratio set w/ limited prior knowledge of silicone rubber
        for i in range(2):
            # insert gripper (deformable)
            f.write('%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f material 2300 %f 0.45 DBC %s %f %f %f 0 0 0\n' \
                % (pad_name, *(100 * tf_lst[i][:3, 3]), *tf_lst[i][:3, :3].flatten(), *(SCALING_FACTOR*scaling), E, DBC_select, *(speed*vel_dir[i])))
            # insert gripper jaw backings
            f.write('%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f linearVelocity %f %f %f\n' % (
                os.path.join(pad_dir, 'cube.obj'), 
                *(100 * (orig_tf_lst[i][:3, 3] - 0.0075*orig_tf_lst[i][:3, 1]- 0.005*orig_tf_lst[i][:3, 2] - 0.035*orig_tf_lst[i][:3, 0])),
                *orig_tf_lst[i][:3, :3].flatten(), 
                3.75, 1.5, 0.2, 
                *(speed*vel_dir[i])
                ))

        f.write("selfFric %f\n" % contact_mu)
        f.write("ground 0.1 0\n")
        f.write("time %f %f\n" % (duration, step))

    with open(os.path.join(output_dir, 'stdout.log'), 'w+') as log:
        proc = ["python3", "batch.py", config, "--output", output_dir]
        if not args.online:
            proc.append("--offline")
        # subprocess.run(proc, stdout=log)
        subprocess.run(proc)
    return output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_name")
    parser.add_argument("grasp_ind", type=int)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--sharp", action="store_true")
    parser.add_argument('--E', type=float, default=1e8)
    parser.add_argument('--contact_mu', type=float, default=0.3)
    parser.add_argument('--time', type=float, default=2)
    parser.add_argument('--step', type=float, default=0.02)
    parser.add_argument('--online', action="store_true")
    args = parser.parse_args()

    tf_lst = get_grasp_tfs(args.obj_name, args.grasp_ind)
    get_ipc_input(args.obj_name, tf_lst, args.output_dir, args.time, args.step, args.E, args.contact_mu, args)
