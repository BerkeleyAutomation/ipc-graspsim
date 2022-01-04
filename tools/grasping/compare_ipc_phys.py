import os
import trimesh
import pandas as pd
import numpy as np
import argparse
import json
from autolab_core import BinaryClassificationResult as bcr

num_grasp_sets = 25

def phys_quality(obj_name):
    with open(os.path.join(os.path.dirname(__file__), '../../dexgrasp_data', \
            'phys_grasps_json', '%s.json' % args.obj_name)) as f:
        grasp_data = json.load(f)
    grasp_qual = {}
    for grasp_ind in range(num_grasp_sets):
        ind = 5*grasp_ind
        success = [trial['result'] for trial in grasp_data[ind:(ind + 5)]]
        success, num_trials = sum(success), len(success)
        grasp_qual[grasp_ind] = (success, num_trials, success/num_trials)
    return grasp_qual

def check_collision(msh):
    col = trimesh.collision.CollisionManager()
    # 1. check that object is in contact with both tooltips
    col.add_object('obj', msh[0]); col.add_object('1', msh[2])
    min_dist = col.min_distance_internal()
    if min_dist > 1e-2:
        return 0
    col.remove_object('1'); col.add_object('2', msh[4])
    min_dist = col.min_distance_internal()
    if min_dist > 1e-2:
        return 0

    # 2. check that the two tooltips are not touching
    col = trimesh.collision.CollisionManager()
    col.add_object('1', msh[2]); col.add_object('2', msh[4])
    min_dist = col.min_distance_internal()
    if min_dist < 1e-2:
        return 0
    return 1

def ipc_quality(obj_name, out_basedir):
    # directory set up is: output/(output basedir, based on E, mu, etc)/samples
    outdir = os.path.join('output', out_basedir)
    grasp_qual = {}
    for grasp_ind in range(num_grasp_sets):
        sample_dirs = [os.path.join(outdir, d) \
                for d in os.listdir(outdir) \
                if (d[:len(args.obj_name)] == obj_name) and \
                (int(d.split('_')[-2]) == grasp_ind)]
        success, num_trials = 0, 0
        for data in sample_dirs:
            # indices: tooltips are 1 and 2, obj is 0
            fin_result = os.path.join(data, 'test_100.obj')
            try:
                finalResult = trimesh.load(fin_result).split() 
                assert len(finalResult) == 3
                success += check_collision(finalResult)
                num_trials += 1
            except:
                pass
        grasp_qual[grasp_ind] = (success, num_trials, \
                success/num_trials if num_trials != 0 else 0)
    return grasp_qual

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_name")
    parser.add_argument("basedir")
    args = parser.parse_args()
    ipc_qual = ipc_quality(args.obj_name, args.basedir)
    phy_qual = phys_quality(args.obj_name)

    recalls = []
    accuracy = []
    for t in np.arange(0.0, 1.1, 0.1):
        results = bcr(
                sum([round(5*i)*[1]+(5-round(5*i))*[0] for i in [x[2] for x in ipc_qual.values()]], []), 
                sum([round(5*i)*[1]+(5-round(5*i))*[0] for i in [x[2] for x in phy_qual.values()]], []),
                threshold=t)
        recalls.append(results.recall)
        accuracy.append(results.ap_score)

    ap, ar = round(results.ap_score, ndigits=2), round(np.mean(recalls), ndigits=2)
    print("IPC: %.3f %.3f %.3f" % (ap, ar, 2*(ap*ar)/(ap+ar)))
        
    ig_qual = pd.read_csv('~/cmk_deformable/deformable_object_grasping-isabella_dev/results/output_%s_1e8_0.4.txt' % args.obj_name, header=None, delimiter='\t', names=['name', 'ind', 'succ', 'time'])
    recalls = []
    accuracy = []
    for t in np.arange(0.0, 1.1, 0.1):
        results = bcr(
                ig_qual['succ'].to_numpy(), 
                sum([round(5*i)*[1]+(5-round(5*i))*[0] for i in [x[2] for x in phy_qual.values()]], []),
                threshold=t)
        recalls.append(results.recall)
        accuracy.append(results.ap_score)

    ap, ar = round(results.ap_score, ndigits=2), round(np.mean(recalls), ndigits=2)
    print("IG: %.3f %.3f %.3f" % (ap, ar, 2*(ap*ar)/(ap+ar)))
