import time
from pathlib import Path
from typing import List, Dict
import torch.optim as optim
from symbol import parameters
from forward_gaitnet import RefNN
import argparse
import pickle5 as pickle
import torch.nn.utils as torch_utils
from pysim import RayEnvManager
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
import numpy as np
import torch
import torch.nn as nn
import math
import random
from advanced_vae import AdvancedVAE
from torch.utils.tensorboard import SummaryWriter
import os

# Network Loading Function, f : (path) -> pi

w_v = 0.1
w_arm = 0.01
w_toe = 0.01
w_mse = 50.0
w_regul = 1
w_kl = 1E-3
w_weakness = 0.5


def loading_distilled_network(path, device):
    print('loading distilled network from', path)
    state = pickle.load(open(path, "rb"))
    env = RayEnvManager(state['metadata'])
    ref = RefNN(len(env.getNormalizedParamState()) + 2,
                len(env.posToSixDof(env.getPositions())), device)
    ref.load_state_dict(convert_to_torch_tensor(state['ref']))
    return ref, env

def loading_training_motion_fast(f):

    if True:
        print("loading ", f)
        loaded_file = np.load(f)

        loaded_param = loaded_file["params"]
        loaded_motion = loaded_file["motions"]

    return loaded_motion[:, :6073]


class VAELearner:
    def __init__(self, device,
                 pose_dof,
                 frame_num,
                 num_paramstate,
                 num_known_param,
                 Forward_GaitNet,
                 buffer_size=30000,
                 learning_rate=5e-5,
                 num_epochs=3,
                 batch_size=128,
                 encoder_hidden_dims=None,
                 decoder_hidden_dims=None,
                 model=None):
        self.device = device
        self.pose_dof = pose_dof
        self.frame_num = frame_num
        self.num_paramstate = num_paramstate

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.default_learning_rate = learning_rate
        self.learning_rate = self.default_learning_rate

        self.num_known_param = num_known_param
        if model:
            self.model = model
        else:
            self.model = AdvancedVAE(
                self.pose_dof, self.frame_num, self.num_known_param, self.num_paramstate, Forward_GaitNet)

        parameters = self.model.parameters()

        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        self.stats = {}
        self.model.train()

        self.diff_w = None

        # Weight initialization
        init_w = np.ones(self.pose_dof)

        # For Root Velocity
        init_w[6] *= w_v
        init_w[8] *= w_v

        init_w[63:] *= w_arm

        init_w[22:24] *= w_toe
        init_w[35:37] *= w_toe

        self.init_w = torch.tensor(
            np.repeat(init_w, 60), dtype=torch.float32, device=self.device)

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

    def learn(self, motion_all) -> Dict:

        converting_time = 0.0
        learning_time = 0.0
        start_time = time.perf_counter()

        idx_all = np.asarray(range(len(motion_all)))

        converting_time = (time.perf_counter() - start_time) * 1000
        start_time = time.perf_counter()
        loss_total = 0
        loss_recon = 0
        loss_kld = 0
        loss_regul = 0
        for _ in range(self.num_epochs):
            np.random.shuffle(idx_all)
            loss_total = 0
            loss_recon = 0
            loss_kld = 0
            loss_regul = 0
            for i in range(len(motion_all) // self.batch_size):
                mini_batch_idx = torch.from_numpy(
                    idx_all[i*self.batch_size: (i+1)*self.batch_size]).cuda()
                motion = torch.index_select(motion_all, 0, mini_batch_idx)
                input, recon, mu, log_var, conditions = self.model(motion)
                motion_diff = (input - recon)

                mse_loss = (motion_diff * self.init_w).pow(2).mean()

                kld_loss = torch.mean(-0.5 * torch.sum(1 +
                                      log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

                # condition diff
                condition_diff = torch.ones(
                    conditions.shape, device="cuda") - (conditions).cuda()
                weakness_num = (self.num_paramstate -
                                self.num_known_param) // 2
                condition_diff[:, weakness_num:] *= w_weakness
                regul_loss = condition_diff.pow(2).mean()

                # Multiply the weight

                loss = w_mse * mse_loss + w_regul * regul_loss + w_kl * kld_loss
                self.optimizer.zero_grad()
                loss.backward()
                
                # for param in self.model.parameters():
                #     if param.grad != None:
                #         param.grad.data.clamp_(-0.05, 0.05)
                torch_utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()
                loss_total += loss.cpu().detach().numpy().tolist()
                loss_recon += mse_loss.cpu().detach().numpy().tolist()
                loss_kld += kld_loss.cpu().detach().numpy().tolist()
                loss_regul += regul_loss.cpu().detach().numpy().tolist()

        loss_total = loss_total / (len(motion_all) // self.batch_size)
        loss_recon = loss_recon / (len(motion_all) // self.batch_size)
        loss_kld = loss_kld / (len(motion_all) // self.batch_size)
        loss_regul = loss_regul / (len(motion_all) // self.batch_size)

        learning_time = (time.perf_counter() - start_time) * 1000

        time_stat = {'converting_time_ms': converting_time,
                     'learning_time_ms': learning_time}
        return {
            'num_tuples': len(motion_all),
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_kld': loss_kld,
            'regul_loss': loss_regul,
        }


parser = argparse.ArgumentParser()
parser.add_argument("--fgn", type=str)
parser.add_argument("--motion", type=str, default="motion.txt")
parser.add_argument("--name", type=str, default="gvae_training")

# Main
if __name__ == "__main__":
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    args = parser.parse_args()
    fgn = args.fgn

    fgn, env = loading_distilled_network(fgn, device)

    # (1) Declare VAE Learner
    vae_learner = VAELearner(device, len(env.posToSixDof(env.getPositions())), 60, len(
        env.getNormalizedParamState()), env.getNumKnownParam(), fgn.cpu().state_dict(), learning_rate=1E-5, batch_size=32, num_epochs=10)

    # (2) Training Dataset Loading
    f = open(args.motion, 'r')
    motions = f.readlines()
    train_filenames = []
    for motion in motions:
        motion = motion.replace('\n', '')
        train_filenames = train_filenames + \
            [motion + '/' + fn for fn in os.listdir(motion)]

    # (3) (Validation Dataset Loading)

    # (4) Training
    writer = SummaryWriter("gvae/" + args.name)
    batch_size = 4096
    training_iter = 0
    used_episode = 0
    num_known_param = env.getNumKnownParam()
    buffers = []
    file_idx = 0
    random.shuffle(train_filenames)
    while True:

        # Collect Training Data from the files until the batch size is full
        if len(buffers) < batch_size:
            while True:

                if file_idx >= len(train_filenames):
                    file_idx %= len(train_filenames)
                    print('All file used', file_idx)
                    random.shuffle(train_filenames)

                if (len(train_filenames[file_idx]) < 5 or train_filenames[file_idx][-4:] != '.npz'):
                    file_idx += 1
                    continue
                f = np.load(train_filenames[file_idx], 'r')
                loaded_motions = f["motions"]
                loaded_params = f["params"]

                # Converting
                for i in range(len(loaded_params)):
                    buffers.append(np.concatenate((loaded_motions[i][:,:].flatten(
                    ), env.getNormalizedParamStateFromParam(loaded_params[i])[:num_known_param])))

                file_idx += 1

                if len(buffers) > batch_size * 100:
                    random.shuffle(buffers)
                    break

        # Training
        training_data = torch.tensor(
            np.array(buffers[:batch_size]), device="cuda")
        buffers = buffers[batch_size:]
        used_episode += len(training_data)
        stat = vae_learner.learn(training_data)
        print("Buffer Size : ", len(buffers), "\tIteration : ",
              training_iter, "\tUsed Episode : ", used_episode, '\t', stat)
        for v in stat:
            writer.add_scalar(v, stat[v], training_iter)

        if training_iter % 50 == 0:
            with open("gvae/" + args.name + "/" + args.name + "_" + str(training_iter), 'wb') as f:
                state = {}
                state["metadata"] = env.getMetadata()
                state["gvae"] = vae_learner.get_weights()
                pickle.dump(state, f)

        training_iter += 1
