import argparse
from pysim import RayEnvManager
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
import numpy as np
import torch
import os
import math
from forward_gaitnet import RefNN
import pickle5 as pickle
import random
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict

import time
import torch.optim as optim
from pathlib import Path

w_root_pos = 2.0
w_arm = 0.5

class RefLearner:
    def __init__(self, device, num_paramstate, ref_dof, phase_dof=1,
                 buffer_size=30000, learning_rate=1e-4, num_epochs=10, batch_size=128, model=None):
        self.device = device
        self.num_paramstate = num_paramstate
        self.ref_dof = ref_dof
        self.num_epochs = num_epochs
        self.ref_batch_size = batch_size
        self.default_learning_rate = learning_rate
        self.learning_rate = self.default_learning_rate
        print('learning rate : ', self.learning_rate,
              'num epochs : ', self.num_epochs)

        if model:
            self.model = model
        else:
            self.model = RefNN(self.num_paramstate + phase_dof,
                               self.ref_dof, self.device).to(self.device)

        parameters = self.model.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        for param in parameters:
            param.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        self.stats = {}
        self.model.train()

    def get_weights(self) -> Dict:
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def set_weights(self, weights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)

    def get_optimizer_weights(self) -> Dict:
        return self.optimizer.state_dict()

    def set_optimizer_weights(self, weights) -> None:
        self.optimizer.load_state_dict(weights)

    def get_model_weights(self, device=None) -> Dict:
        if device:
            return {k: v.to(device) for k, v in self.model.state_dict().items()}
        else:
            return self.model.state_dict()

    def save(self, name):
        path = Path(name)
        torch.save(self.model.state_dict(), path)
        torch.save(self.optimizer.state_dict(),
                   path.with_suffix(".opt" + path.suffix))

    def load(self, name):
        path = Path(name)
        self.model.load_state_dict(torch.load(path))
        self.optimizer.load_state_dict(torch.load(
            path.with_suffix(".opt" + path.suffix)))

    def validation(self, param_all, d_all) -> Dict:
        with torch.no_grad():
            d = self.model(param_all)
            validation_loss = 5.0 * (d_all - d).pow(2).mean()
            return validation_loss.cpu()

    def learn(self, param_all, d_all) -> Dict:

        converting_time = 0.0
        learning_time = 0.0
        start_time = time.perf_counter()

        assert (len(param_all) == len(d_all))
        idx_all = np.asarray(range(len(param_all)))

        converting_time = (time.perf_counter() - start_time) * 1000
        start_time = time.perf_counter()
        loss_avg = 0.

        for _ in range(self.num_epochs):
            np.random.shuffle(idx_all)
            loss_avg = 0
            for i in range(len(param_all) // self.ref_batch_size):
                mini_batch_idx = torch.from_numpy(
                    idx_all[i*self.ref_batch_size: (i+1)*self.ref_batch_size]).cuda()
                param = torch.index_select(param_all, 0, mini_batch_idx)
                d = torch.index_select(d_all, 0, mini_batch_idx)
                d_out = self.model(param)
                diff = d - d_out
                diff[:, 6:9] *= w_root_pos
                diff[:, 57:] *= w_arm

                loss = (5.0 * diff.pow(2)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                
                for param in self.model.parameters():
                    if param.grad != None:
                        param.grad.data.clamp_(-0.1, 0.1)

                self.optimizer.step()
                loss_avg += loss.cpu().detach().numpy().tolist()

        loss_ref = loss_avg / (len(param_all) // self.ref_batch_size)
        learning_time = (time.perf_counter() - start_time) * 1000

        time_stat = {'converting_time_ms': converting_time,
                     'learning_time_ms': learning_time}
        return {
            'num_tuples': len(param_all),
            'loss_distillation': loss_ref,
            'time': time_stat
        }


parser = argparse.ArgumentParser()

# Raw Motion Path
parser.add_argument("--motion", type=str, default="motion.txt")
parser.add_argument("--env", type=str, default="/home/gait/BidirectionalGaitNet_Data/GridSampling/3rd_rollout/env.xml")
parser.add_argument("--name", type=str, default="distillation")
parser.add_argument("--validation", action='store_true')

if __name__ == "__main__":
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parser.parse_args()
    
    f = open(args.motion, 'r')
    motions = f.readlines()
    train_filenames = []
    for motion in motions:
        motion = motion.replace('\n', '')
        train_filenames = train_filenames + [motion + '/' + fn for fn in os.listdir(motion)]
    
    train_filenames.sort()

    file_idx = 0

    # Environment Loading
    env = RayEnvManager(args.env)

    # Loading all motion from file Data PreProcessing
    buffers = [[], []]  # {Param, Phi}, {Pose}
    raw_buffers = [[], []]
    training_sets = [None, None]

    batch_size = 65536
    large_batch_scale = 100
    pos_dof = 0
    ref_learner = \
        RefLearner(torch.device("cuda"), len(env.getNormalizedParamState()), len(env.posToSixDof(env.getPositions())), 2, learning_rate=1e-5, batch_size=128, num_epochs=5)
    iter = 0

    # Writer Logging
    writer = SummaryWriter("distillation/" + args.name)

    tuple_size = 0
    print(len(train_filenames), ' files are loaded ....... ')

    phi = np.array([[math.sin(i * (1.0/30) * 2 * math.pi), math.cos(i * (1.0/30) * 2 * math.pi)] for i in range(60)])
    num_knownparam = env.getNumKnownParam()
    while True:
        random.shuffle(train_filenames)     
        num_tuple = 0
        while True:
            if raw_buffers[0] != None and (len(raw_buffers[0]) > batch_size * large_batch_scale):
                break
            else:
                f = train_filenames[file_idx % len(train_filenames)]
                file_idx += 1
                if file_idx > len(train_filenames):
                    print('All files are used')
                    file_idx %= len(train_filenames)
                if f[-4:] != ".npz":
                    continue

                path = f 
                print(path)
                loaded_file = np.load(path, allow_pickle=True)
                loaded_motions = loaded_file["motions"]
                loaded_params = loaded_file["params"]
                loaded_idx = 0
                
                i = 0
                              
                for loaded_idx in range(len(loaded_motions)):
                    param_matrix = np.repeat(env.getNormalizedParamStateFromParam(loaded_params[loaded_idx]), 60).reshape(-1, 60).transpose()
                    data_in = np.concatenate((param_matrix, phi), axis=1)
                    data_out = loaded_motions[loaded_idx][:,:]
                    raw_buffers[0] += list(data_in)  
                    raw_buffers[1] += list(data_out)  


        buffers[0] = torch.tensor(
            np.array(raw_buffers[0][:batch_size * large_batch_scale], dtype=np.float32))
        buffers[1] = torch.tensor(
            np.array(raw_buffers[1][:batch_size * large_batch_scale], dtype=np.float32))

        raw_buffers[0] = raw_buffers[0][batch_size * large_batch_scale:]
        raw_buffers[1] = raw_buffers[1][batch_size * large_batch_scale:]

        if True:
            idx_all = np.asarray(range(len(buffers[0])))
            np.random.shuffle(idx_all)

            for i in range(len(idx_all) // batch_size):
                mini_batch_idx = torch.from_numpy(
                    idx_all[i*batch_size: (i+1)*batch_size])
                training_sets[0] = torch.index_select(
                    buffers[0], 0, mini_batch_idx).cuda()
                training_sets[1] = torch.index_select(
                    buffers[1], 0, mini_batch_idx).cuda()
                tuple_size += len(training_sets[0])

                v_stat = ref_learner.learn(training_sets[0], training_sets[1])

                v_stat.pop('time')

                # while True:
                print('Iteration : ', iter, '\tRaw Buffer : ', len(raw_buffers[0]), '\tBuffer : ', len(buffers[0]), v_stat, '\tTuple Size : ', tuple_size)

                for v in v_stat:
                    writer.add_scalar(v, v_stat[v], iter)

                if iter % 50 == 0:
                    with open("distillation/" + args.name + "/" + args.name + "_" + str(iter), 'wb') as f:
                        state = {}
                        state["metadata"] = env.getMetadata()
                        state["is_cascaded"] = True
                        state["ref"] = ref_learner.get_weights()
                        pickle.dump(state, f)

                iter += 1
