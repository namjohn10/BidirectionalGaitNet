from IPython import embed
from math import fabs
import torch
import torch.nn as nn
import numpy as np
import pickle5 as pickle

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
import torch.nn.functional as F

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


class MuscleNN(nn.Module):
    def __init__(self, num_total_muscle_related_dofs, num_dofs, num_muscles, is_cpu=False, is_cascaded=False):
        super(MuscleNN, self).__init__()

        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs  # Exclude Joint Root dof
        self.num_muscles = num_muscles
        self.isCuda = False
        self.isCascaded = is_cascaded

        num_h1 = 256
        num_h2 = 256
        num_h3 = 256

        self.fc = nn.Sequential(
            nn.Linear(num_total_muscle_related_dofs+num_dofs +
                      (num_muscles + 1 if self.isCascaded else 0), num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3, num_muscles),
        )

        # Normalization
        self.std_muscle_tau = torch.ones(
            self.num_total_muscle_related_dofs) * 200
        self.std_tau = torch.ones(self.num_dofs) * 200

        if torch.cuda.is_available() and not is_cpu:
            self.isCuda = True
            self.std_tau = self.std_tau.cuda()
            self.std_muscle_tau = self.std_muscle_tau.cuda()
            self.cuda()

        self.fc.apply(weights_init)

    def forward_with_prev_out_wo_relu(self, muscle_tau, tau, prev_out, weight=1.0):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        if type(prev_out) == np.ndarray:
            with torch.no_grad():
                prev_out = torch.FloatTensor(prev_out)
                out = prev_out + weight * \
                    self.fc.forward(
                        torch.cat([0.5 * prev_out, weight, muscle_tau, tau], dim=-1))
                return out
        else:
            out = prev_out + weight * \
                self.fc.forward(
                    torch.cat([0.5 * prev_out, weight, muscle_tau, tau], dim=-1))
            return out

    def forward_wo_relu(self, muscle_tau, tau):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
        return out

    def forward(self, muscle_tau, tau):
        return torch.relu(torch.tanh(self.forward_wo_relu(muscle_tau, tau)))

    def forward_with_prev_out(self, muscle_tau, tau, prev_out, weight=1.0):
        return torch.relu(torch.tanh(self.forward_with_prev_out_wo_relu(muscle_tau, tau, prev_out, weight)))

    def unnormalized_no_grad_forward(self, muscle_tau, tau, prev_out=None, numpyout=False, weight=None):
        with torch.no_grad():
            if type(self.std_muscle_tau) == torch.Tensor and type(muscle_tau) != torch.Tensor:
                if self.isCuda:
                    muscle_tau = torch.FloatTensor(muscle_tau).cuda()
                else:
                    muscle_tau = torch.FloatTensor(muscle_tau)

            if type(self.std_tau) == torch.Tensor and type(tau) != torch.Tensor:
                if self.isCuda:
                    tau = torch.FloatTensor(tau).cuda()
                else:
                    tau = torch.FloatTensor(tau)

            if type(weight) != type(None):
                if self.isCuda:
                    weight = torch.FloatTensor([weight]).cuda()
                else:
                    weight = torch.FloatTensor([weight])

            muscle_tau = muscle_tau / self.std_muscle_tau
            tau = tau / self.std_tau

            if type(prev_out) == type(None) and type(weight) == type(None):
                out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
            else:
                if self.isCuda:
                    prev_out = torch.FloatTensor(prev_out).cuda()
                else:
                    prev_out = torch.FloatTensor(prev_out)

                if type(weight) == type(None):
                    print('Weight Error')
                    exit(-1)
                out = self.fc.forward(
                    torch.cat([0.5 * prev_out, weight, muscle_tau, tau], dim=-1))

            if numpyout:
                out = out.cpu().numpy()

            return out

    def forward_filter(self, unnormalized_activation):
        return torch.relu(torch.tanh(torch.FloatTensor(unnormalized_activation))).cpu().numpy()

    def load(self, path):
        print('load muscle nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(torch.Tensorpath))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save muscle nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_activation(self, muscle_tau, tau):
        act = self.forward(torch.FloatTensor(muscle_tau.reshape(1, -1)),
                           torch.FloatTensor(tau.reshape(1, -1)))
        return act.cpu().detach().numpy()[0]


class SimulationNN(nn.Module):
    def __init__(self, num_states, num_actions, learningStd=False):
        nn.Module.__init__(self)
        self.num_states = num_states
        self.num_actions = num_actions

        self.num_h1 = 512
        self.num_h2 = 512
        self.num_h3 = 512

        self.log_std = None
        init_log_std = 1.0 * torch.ones(num_actions)
        init_log_std[18:] *= 0.5 ## For Upper Body

        ## For Cascading 
        init_log_std[-1] = 1.0

        if learningStd:
            self.log_std = nn.Parameter(init_log_std)
        else:
            self.log_std = init_log_std

        self.p_fc = nn.Sequential(
            nn.Linear(self.num_states, self.num_h1),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h1, self.num_h2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h2, self.num_h3),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h3, self.num_actions),
        )

        self.v_fc = nn.Sequential(
            nn.Linear(self.num_states, self.num_h1),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h1, self.num_h2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h2, self.num_h3),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h3, 1),
        )

        self.reset()

        if torch.cuda.is_available():
            if not learningStd:
                self.log_std = self.log_std.cuda()
            self.cuda()

    def reset(self):
        self.p_fc.apply(weights_init)
        self.v_fc.apply(weights_init)

    def forward(self, x):
        p_out = MultiVariateNormal(self.p_fc.forward(x), self.log_std.exp())
        v_out = self.v_fc.forward(x)
        return p_out, v_out

    def load(self, path):
        print('load simulation nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save simulation nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_action(self, s):
        ts = torch.tensor(s)
        p, _ = self.forward(ts)
        return p.loc.cpu().detach().numpy()

    def get_value(self, s):
        ts = torch.tensor(s)
        _, v = self.forward(ts)
        return v.cpu().detach().numpy()

    def get_random_action(self, s):
        # print(self.log_std)
        ts = torch.tensor(s)
        p, _ = self.forward(ts)
        return p.sample().cpu().detach().numpy()

    def get_noise(self):
        return self.log_std.exp().mean().item()


class RolloutNNRay(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, {}, "RolloutNNRay")
        nn.Module.__init__(self)

        self.num_states = np.prod(obs_space.shape)
        self.num_actions = np.prod(action_space.shape)

        self.action_dist_loc = torch.zeros(self.num_actions)
        self.action_dist_scale = torch.zeros(self.num_actions)
        self._value = None
        self.dummy_param = nn.Parameter(torch.ones(self.num_actions))

    # @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)

        action_tensor = None
        if torch.cuda.is_available():
            action_tensor = torch.zeros(
                obs.shape[0], 2 * self.num_actions).cuda()
            self._value = torch.zeros(obs.shape[0], 1).cuda()
        else:
            action_tensor = torch.zeros(obs.shape[0], 2 * self.num_actions)
            self._value = torch.zeros(obs.shape[0], 1)

        return action_tensor, state

    # @override(TorchModelV2)
    def value_function(self):
        return self._value.squeeze(1)  # self._value.squeeze(1)


class SimulationNN_Ray(TorchModelV2, SimulationNN):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        num_states = np.prod(obs_space.shape)
        num_actions = np.prod(action_space.shape)
        SimulationNN.__init__(
            self, num_states, num_actions, kwargs['learningStd'])
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, {}, "SimulationNN_Ray")
        num_outputs = 2 * np.prod(action_space.shape)
        self._value = None

    def get_value(self, obs):
        with torch.no_grad():
            _, v = SimulationNN.forward(self, obs)
            return v

    # @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)
        action_dist, self._value = SimulationNN.forward(self, x)
        action_tensor = torch.cat(
            [action_dist.loc, action_dist.scale.log()], dim=1)
        return action_tensor, state

    # @override(TorchModelV2)
    def value_function(self):
        return self._value.squeeze(1)

    def reset(self):
        SimulationNN.reset(self)

    def vf_reset(self):
        SimulationNN.vf_reset(self)

    def pi_reset(self):
        SimulationNN.pi_reset(self)


class PolicyNN:
    def __init__(self, num_states, num_actions, policy_state, filter_state, device, learningStd=False):

        self.policy = SimulationNN(
            num_states, num_actions, learningStd).to(device)

        self.policy.log_std = self.policy.log_std.to(device)
        self.policy.load_state_dict(convert_to_torch_tensor(policy_state))
        self.policy.eval()
        self.filter = filter_state
        # self.cascading_type = cascading_type

    def get_filter(self):
        return self.filter.copy()

    def get_value(self, obs):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        v = self.policy.get_value(obs)
        return v

    def get_value_function_weight(self):
        return self.policy.value_function_state_dict()

    def get_action(self, obs, is_random=False):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        return self.policy.get_action(obs) if not is_random else self.policy.get_random_action(obs)

    def get_filtered_obs(self, obs):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        return obs

    def weight_filter(self, unnormalized, beta):
        scale_factor = 1000.0
        return torch.sigmoid(torch.tensor([scale_factor * (unnormalized - beta)])).numpy()[0]

    def state_dict(self):
        state = {}
        state["weight"] = (self.policy.state_dict())
        state["filter"] = self.filter
        return state

    def soft_load_state_dict(self, _state_dict):
        self.policy.soft_load_state_dict(_state_dict)


def generating_muscle_nn(num_total_muscle_related_dofs, num_dof, num_muscles, is_cpu=True, is_cascaded=False):
    muscle = MuscleNN(num_total_muscle_related_dofs, num_dof,
                      num_muscles, is_cpu, is_cascaded)
    return muscle


def loading_metadata(path):
    state = pickle.load(open(path, "rb"))
    # print(state["metadata"])
    return state["metadata"] if "metadata" in state.keys() else None


def loading_network(path, num_states=0, num_actions=0,
                    use_musclenet=False, num_actuator_action=0, num_muscles=0, num_total_muscle_related_dofs=0,
                    device="cpu"):

    state = pickle.load(open(path, "rb"))
    worker_state = pickle.loads(state["worker"])
    policy_state = worker_state["state"]['default_policy']['weights']
    filter_state = worker_state["filters"]['default_policy']

    device = torch.device(device)
    learningStd = ('log_std' in policy_state.keys())
    policy = PolicyNN(num_states, num_actions, policy_state,
                      filter_state, device, learningStd)

    muscle = None
    if use_musclenet:
        muscle = MuscleNN(num_total_muscle_related_dofs, num_actuator_action, num_muscles, is_cpu=True, is_cascaded=(
            state["cascading"] if "cascading" in state.keys() else False))
        muscle.load_state_dict(convert_to_torch_tensor(state['muscle']))

    return policy, muscle
