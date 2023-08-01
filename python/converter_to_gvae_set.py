import argparse
from pysim import RayEnvManager
import numpy as np
import os
# from ray_train import RefLearner
import pickle5 as pickle
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
env = None
# This code is code for converting the raw motion data to the refined training dataset for Bidirectional GaitNet

# Raw Motion Path

parser.add_argument("--motion", type=str, default="/home/gait/BidirectionalGaitNet_Data_02/GridSampling/3rd/Raw_02")  # raw motion path
parser.add_argument("--save", type=str, default="/home/gait/BidirectionalGaitNet_Data_02/GridSampling/3rd/Data02")  # raw motion path
parser.add_argument("--env", type=str, default="/home/gait/BidirectionalGaitNet_Data_02/env.xml")
parser.add_argument("--name", type=str, default="refined_data")


# convert raw motion to refined motion (2cycles, 60 frames)
def convertToRefinedMotion(f, num_known_param, resolution=30, max_motions=1000000):
    # refined_motion : [motion, knownparam, truthparam]
    # return : list of refined_motion
    result = []

    phis = []
    for i in range(resolution * 3):
        phis.append(i / resolution)

    loaded_file = np.load(f)

    loaded_params = loaded_file["params"]
    loaded_motions = loaded_file["motions"]
    loaded_lengths = loaded_file["lengths"]

    i = 0
    for loaded_idx in range(len(loaded_lengths)):
        isInit = True
        prev_phi = None
        param = loaded_params[loaded_idx]


        if loaded_lengths[loaded_idx] > 140:
            # Motion preprocessing for motion phi
            prev_phi = -1
            phi_offset = -1
            for j in range(loaded_lengths[loaded_idx]):
                if prev_phi > loaded_motions[i + j][-1]:
                    phi_offset += 1
                prev_phi = loaded_motions[i+j][-1]
                loaded_motions[i+j][-1] += phi_offset

            phi_idx = 0
            motion_idx = 0
            refined_motion = [[], param]
            while phi_idx < len(phis) and motion_idx < loaded_lengths[loaded_idx] - 1:
                # from IPython import embed
                # embed()
                if loaded_motions[i+motion_idx][-1] <= phis[phi_idx] and phis[phi_idx] < loaded_motions[i+motion_idx+1][-1]:
                    w1 = loaded_motions[i+motion_idx+1][-1] - phis[phi_idx]
                    w2 = phis[phi_idx] - loaded_motions[i+motion_idx][-1]
                    # Interpolate six dof pos
                    v1 = loaded_motions[i+motion_idx][3:6] - loaded_motions[i+motion_idx-1][3:6]
                    v2 = loaded_motions[i+motion_idx+1][3:6] - loaded_motions[i+motion_idx][3:6]

                    p = (w1 * env.posToSixDof(loaded_motions[i+motion_idx][:-1]) + w2 * env.posToSixDof(loaded_motions[i+motion_idx+1][:-1])) / (w1 + w2)
                    v = (w1 * v1 + w2 * v2) / (w1 + w2)

                    p[6] = v[0]
                    p[8] = v[2]

                    # concatenate 'p' and 'param'
                    # refined_motion[0].append(np.concatenate((p, param[:num_known_param]), axis=0))
                    refined_motion[0].append(p)
                    phi_idx += 1
                else:
                    motion_idx += 1
            

            if len(refined_motion[0]) >= 60:
                result.append([refined_motion[0][:60], refined_motion[1]])
                if len(refined_motion[0]) == 90:
                    result.append([refined_motion[0][30:], refined_motion[1]])
            else:
                print("Error : ", len(refined_motion[0]))

        i += loaded_lengths[loaded_idx]
    return result

def save_motions(motions, params):
    np.savez_compressed("new_motion", motions=motions, params=params)


if __name__ == "__main__":
    args = parser.parse_args()

    ## print args information 
    print("motion path : ", args.motion)
    print("save path : ", args.save)


    train_filenames = os.listdir(args.motion)
    train_filenames.sort()

    file_idx = 0

    # Environment Loeading
    env = RayEnvManager(args.env)

    # Loading all motion from file Data PreProcessing
    file_idx = 0
    save_idx = 176
    results = []

    print(len(train_filenames), ' files are loaded ....... ')
    # while True:
    while file_idx < len(train_filenames):
        f = train_filenames[file_idx % len(train_filenames)]
        file_idx += 1
        if f[-4:] != ".npz":
            # print(path, ' is not npz file')
            continue
        path = args.motion + '/' + f

        results += convertToRefinedMotion(path, env.getNumKnownParam())

        if len(results) > 4096:
            res = results[:4096]
            motions = np.array([r[0] for r in res])
            params = np.array([r[1] for r in res])
            np.savez_compressed(args.save + "/" + args.name + "_" + str(save_idx), motions=motions, params=params)
            results = results[4096:]
            save_idx += 1
