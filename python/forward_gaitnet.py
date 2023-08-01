
import torch
import torch.nn as nn
import numpy as np
import pickle5 as pickle
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from pysim import RayEnvManager

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(
    self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

class RefNN(nn.Module):
    def __init__(self, num_paramstate, ref_dof, device=None):
        super(RefNN, self).__init__()
        self.num_paramstate = num_paramstate
        num_h1 = 512
        num_h2 = 512
        num_h3 = 512

        self.fc = nn.Sequential(
            nn.Linear(self.num_paramstate, num_h1),
            nn.ReLU(),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, num_h3),
            nn.ReLU(),
            nn.Linear(num_h3, ref_dof),
        )
        if device != 'cpu' and torch.cuda.is_available():
            self.cuda()
            self.device = device
        else:
            self.device = 'cpu'

        self.fc.apply(weights_init)

    def forward(self, param_state):
        # param_state -= 1.0
        # param_state *= -1
        v_out = self.fc.forward(param_state)
        return v_out

    def get_action(self, s):
        s = np.array(s, dtype=np.float32)
        ts = torch.tensor(s)
        p = self.forward(ts)
        return p.detach().numpy()

    def get_displacement(self, param_state):
        with torch.no_grad():
            ts = torch.tensor(param_state).to(self.device)
            v_out = self.forward(ts)
            return v_out.cpu().detach().numpy()[0]

    def load(self, path):
        print('load ref nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save ref nn {}'.format(path))
        torch.save(self.state_dict(), path)


def load_FGN(checkpoint_file, num_paramstates, ref_dof):
    state = pickle.load(open(checkpoint_file, "rb"))
    env = RayEnvManager(state['metadata'])
    num_paramstates = len(env.getParamState())
    ref_dof = len(env.posToSixDof(env.getPositions()))
    ref = RefNN(num_paramstates + 2, ref_dof, 'cpu')
    ref.load_state_dict(convert_to_torch_tensor(state['ref']))
    return ref, state['metadata']


def load_motions_from_file(motion_file_path, num_known_param):
    
    result_motion = []
    result_param = []

    motion_file = np.load(motion_file_path)
    # from IPython import embed; embed()

    motions = motion_file["motions"]
    params = motion_file["params"]
    
    for i in range(len(motions)):
        if(len(motions[i]) == 6060): ## Hard coding 
            result_motion.append(motions[i])
            result_param.append(params[i])
        else:
            result_motion.append(motions[i].flatten())
            result_param.append(params[i])

    return np.array(result_param), np.array(result_motion)
