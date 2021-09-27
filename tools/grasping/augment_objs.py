import trimesh as tr
import trimesh.transformations as tra
import glob
import numpy as np

import pyrender as pr
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
import imageio
import os

def add_gripper_head(head, tf_lst):
    ret = head.copy()
    ret.vertices = tr.transform_points(ret.vertices, tf_lst[0])
    ret.vertices += (tf_lst[1][:3, 3] - tf_lst[0][:3, 3])/2
    ret.vertices += (tf_lst[0][:3, 2]) * -10
    return ret

def add_gripper_backs(msh, tf_lst):
    backs = [msh.copy(), msh.copy()]
    center = (tf_lst[1][:3, 3] + tf_lst[0][:3, 3])/2
    
    for i, back in enumerate(backs):
        back.vertices = tr.transform_points(back.vertices, tra.rotation_matrix(-np.pi/2, [0, 1, 0]))
        back.vertices -= back.extents[0]/2

        back.vertices = tr.transform_points(back.vertices, tf_lst[i])
        direction = np.sign(np.dot(tf_lst[i][:3, 3] - center, tf_lst[i][:3, 0]))
        back.vertices += tf_lst[i][:3, 0]*0.30*direction
        back.vertices += tf_lst[i][:3, 2]*1.70
    return backs

def load_initial(outdir, ind):
    # ordering: 
    # 0- object
    # 1- gripper 1 (deformable)
    # 2- gripper 1 (rigid backing)
    # 3- gripper 1 (deformable, postprocess)--> ADDED HERE
    # 4- gripper 2 (deformable)
    # 5- gripper 2 (rigid backing)
    # 6- gripper 1 (deformable, postprocess)--> ADDED HERE
    # 7- gripper head-------------------------> ADDED HERE

    grasp = tr.load(os.path.join(outdir, f'test_{ind}.obj'), process=False)
    grasp = grasp.split().tolist()
    for obj in grasp:
        obj.fix_normals()

    rounded_msh = tr.load('round_pad.obj')
    rounded_msh.vertices *= 80
    head_msh = tr.load('base.stl')
    head_msh.vertices *= 100

    backing_lst = [grasp[2], grasp[4]]
    gripper_poses = [np.linalg.inv(b.principal_inertia_transform) for b in backing_lst]

    head = add_gripper_head(head_msh, gripper_poses)
    rounded = add_gripper_backs(rounded_msh, gripper_poses)

    grasp = grasp[:3] + [rounded[0]] + grasp[3:5] + [rounded[1]] + [head]
    return grasp

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, 'objs'), exist_ok=True)
    for obj in tqdm(glob.glob(os.path.join(args.outdir, 'test_*.obj'))):
        ind = int(os.path.basename(obj).split('.')[0].split('_')[1])
        grasp = load_initial(args.outdir, ind)
        tr.Scene(grasp).export(os.path.join(args.outdir, 'objs', '%05d.obj'%ind))

