from ray.rllib.utils.torch_ops import convert_to_torch_tensor
import pandas as pd
import matplotlib.pyplot as plt
import umap.plot
import umap
import numpy as np
import math
import torch
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
from torch.nn import functional as F
import pickle5 as pickle

Tensor = TypeVar('torch.tensor')

class AdvancedVAE(nn.Module):
    def __init__(self,
                 pose_dof: int,
                 frame_num: int,
                 #  latent_dim: int,
                 num_known_param: int,
                 num_paramstate: int,
                 Kinematic_Gaitnet=None,
                 encoder_hidden_dims: List = None,
                 predecoder_hidden_dims: List = None,
                 decoder_hidden_dims: List = None,
                 **kwargs):
        super(AdvancedVAE, self).__init__()

        self.latent_dim = 32
        self.num_paramstate = num_paramstate
        self.pose_dof = pose_dof
        self.frame_num = frame_num
        self.motion_dim = pose_dof * frame_num
        self.num_known_param = num_known_param  # Mesurable Conditions

        # Build Encoder
        modules = []
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 256, 256]
        input = self.motion_dim + num_known_param  # For Condition Parameter Size
        for h_dim in encoder_hidden_dims:
            modules.append(nn.Linear(input, h_dim))
            modules.append(nn.LeakyReLU(0.2, inplace=True))
            input = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(encoder_hidden_dims[-1], self.latent_dim)

        # # Build PreDecoder
        modules = []
        input = self.latent_dim + num_known_param  # For Condition Parameter Size
        if predecoder_hidden_dims is None:
            predecoder_hidden_dims = [256, 256, 256]
        for h_dim in predecoder_hidden_dims:
            modules.append(nn.Linear(input, h_dim))
            modules.append(nn.LeakyReLU(0.2, inplace=True))
            input = h_dim
        modules.append(nn.Linear(input, self.num_paramstate - num_known_param))
        modules.append(nn.Sigmoid())
        self.pre_decoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [512, 512, 512]
        input = self.num_paramstate + 2
        for h_dim in decoder_hidden_dims:
            modules.append(nn.Linear(input, h_dim))
            modules.append(nn.ReLU())
            input = h_dim
        modules.append(nn.Linear(input, self.pose_dof))
        self.decoder = nn.Sequential(*modules)

        weights = self.decoder.state_dict()
        keys = list(weights.keys())
        idx = 0
        if Kinematic_Gaitnet is not None:
            for w in Kinematic_Gaitnet:
                weights[keys[idx]] = Kinematic_Gaitnet[w]
                idx += 1
        self.decoder.load_state_dict(weights)

        # Freezing Decoder Parameter
        for param in self.decoder.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.cuda()

    def encode(self, input: Tensor, isRender=False) -> List[Tensor]:

        if isRender:
            input = torch.tensor(input, device="cuda")
        result = self.encoder(input)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var] if not isRender else (mu.cpu().detach().numpy(), (log_var * 0.5).exp().cpu().detach().numpy())

    def encode_to_condition(self, motion, isRender=False):
        if isRender:
            motion = torch.tensor([motion], device="cuda")
        mu, log_var = self.encode(motion)
        print(' mu : ', mu, ' log_var :', log_var)
        z = self.reparameterize(mu, log_var)
        return self.pre_decoder(z) if not isRender else self.pre_decoder(z)[0].cpu().detach().numpy()

    def decode_to_full_condition(self, z_):
        z_ = torch.tensor(z_, device="cuda")

        z = self.pre_decoder(z_).cpu().detach().numpy()
        param_state = np.ones(self.num_paramstate)
        param_state[:self.num_known_param] = z_[-self.num_known_param:].cpu().detach().numpy()
        param_state[self.num_known_param:] = z

        return param_state

    def gaitnet(self, z: Tensor, isRender=False) -> Tensor:
        if isRender:
            z = torch.tensor([z], device="cuda")

        z_dim = len(z.shape)

        if z_dim == 1:
            z = z.repeat(self.frame_num, 1)
        elif z_dim == 2:
            z = z.repeat(self.frame_num, 1, 1)

        phi = []
        for i in range(self.frame_num):
            angle = 4 * math.pi * (i / self.frame_num)
            phi_ = [math.sin(angle), math.cos(angle)]
            phi.append(phi_)
        phi = torch.tensor(phi, device="cuda")
        if z_dim == 2:
            phi = torch.transpose(phi.repeat(z.shape[1], 1, 1), 0, 1)
        input = torch.cat((z, phi), len(z.shape) - 1)
        result = self.decoder(input)
        # result[:, :, 6] = 0
        # result[:, :, 8] = 0
        result = result.transpose(0, 1).flatten(1, 2)
        return result if not isRender else result[0].cpu().detach().numpy()

    def decode(self, z_: Tensor, isRender=False) -> Tensor:
        if isRender:
            z_ = torch.tensor([z_], device="cuda")

        known_p = z_[:, -self.num_known_param:]

        z = self.pre_decoder(z_)
        z = torch.cat((known_p, z), dim=1)
        z_dim = len(z.shape)

        if z_dim == 1:
            z = z.repeat(self.frame_num, 1)
        elif z_dim == 2:
            z = z.repeat(self.frame_num, 1, 1)

        phi = []
        for i in range(self.frame_num):
            angle = 4 * math.pi * (i / self.frame_num)
            phi_ = [math.sin(angle), math.cos(angle)]
            phi.append(phi_)
        phi = torch.tensor(phi, device="cuda")
        if z_dim == 2:
            phi = torch.transpose(phi.repeat(z.shape[1], 1, 1), 0, 1)
        input = torch.cat((z, phi), len(z.shape) - 1)
        result = self.decoder(input)
        # result[:, :, 6] = 0
        # result[:, :, 8] = 0
        result = result.transpose(0, 1).flatten(1, 2)

        return result if not isRender else result[0].cpu().detach().numpy()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, motion):
        mu, log_var = self.encode(motion)
        known_param = motion[:, -self.num_known_param:]

        z = self.reparameterize(mu, log_var)
        z = torch.cat((z, known_param), dim=1)
        c = self.pre_decoder(z)
        c = torch.cat((known_param, c), dim=1)

        return motion[:, :-self.num_known_param], self.gaitnet(c), mu, log_var, c

    def render_forward(self, motion):

        motion = torch.tensor([motion], device="cuda")
        mu, log_var = self.encode(motion)
        known_param = motion[:, -self.num_known_param:]

        z = self.reparameterize(mu, log_var)
        z = torch.cat((z, known_param), dim=1)
        c = self.pre_decoder(z)
        c = torch.cat((known_param, c), dim=1)

        return self.gaitnet(c)[0].detach().cpu().numpy(), c[0].detach().cpu().numpy()

    # Sampling 1000 times and draw umap plot
    def sampling(self, motion, truth):

        motion = torch.tensor([motion], device="cuda")
        mu, log_var = self.encode(motion)
        known_param = motion[:, -self.num_known_param:]

        res = []
        for i in range(1000):
            z = self.reparameterize(mu, log_var)
            z = torch.cat((z, known_param), dim=1)
            c = self.pre_decoder(z)
            c = torch.cat((known_param, c), dim=1)

            res.append(c.detach().cpu().numpy()[0])

        res.append(truth)

        res = np.array(res)
        embedding = umap.UMAP(n_neighbors=5, random_state=42,
                              min_dist=0.00).fit_transform(res)

        plt.scatter(embedding[:, 0], embedding[:, 1],
                    color='gray', s=5.0, alpha=0.5)
        plt.scatter(embedding[-1, 0], embedding[-1, 1], color='r', s=10.0)

        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('UMAP Plot')
        plt.show()

        return

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    def load(self, path):
        print('load gait_vae nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save gait_vae nn {}'.format(path))
        torch.save(self.state_dict(), path)


def load_gaitvae(checkpoint_path, pose_dof, frame_num, num_knownparam, num_paramstate):
    print('called load_gaitvae')
    state = pickle.load(open(checkpoint_path, "rb"))
    GaitVAE = AdvancedVAE(pose_dof, frame_num, num_knownparam, num_paramstate)
    w = convert_to_torch_tensor(state['gvae'])
    GaitVAE.load_state_dict(w)
    return GaitVAE
