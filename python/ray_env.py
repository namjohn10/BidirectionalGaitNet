from pysim import RayEnvManager
from IPython import embed
import numpy as np
import gym
import ray
from ray.rllib.utils.torch_ops import convert_to_torch_tensor


class MyEnv(gym.Env):
    def __init__(self, metadata):
        self.env = RayEnvManager(metadata)

        self.env.updateParamState()
        self.env.reset()
        self.obs = self.env.getState()

        self.num_state = len(self.obs)
        self.num_action = len(self.env.getAction())
        self.metadata = self.env.getMetadata()

        self.observation_space = gym.spaces.Box(
            low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_state,))
        self.action_space = gym.spaces.Box(
            low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_action,))

        self.use_cascading = self.env.getUseCascading()

        # For Mass Actuactor (2-Level)
        self.isTwoLevelActuactor = self.env.isTwoLevelController()

        if self.isTwoLevelActuactor:
            if not self.use_cascading:
                self.muscle_tuples = [[], [], []]
            else:
                self.muscle_tuples = [[], [], [], [], []]
        self.param_count = 0

        # For rollout
        self.is_rollout = False
        self.rollout_count = 0
        self.episode_buffer = [self.env.getParamState(), []]
        self.file_buffer = []
        self.idx = 0
        self.save_iter = 0

    def reset(self):
        if self.param_count > 300 or self.is_rollout:
            self.env.updateParamState()
            self.param_count = 0
        self.env.reset()
        self.obs = self.env.getState()

        if self.is_rollout:
            self.rollout_count = 0
            if (len(self.episode_buffer[1]) > 0):
                self.file_buffer.append(
                    [np.array(self.episode_buffer[0]), np.array(self.episode_buffer[1])])
            self.episode_buffer = [self.env.getParamState(), []]

        if len(self.file_buffer) > 10:
            params = np.array([p[0] for p in self.file_buffer])
            lengths = np.array([len(p[1]) for p in self.file_buffer])
            motions = np.concatenate([np.array(p[1])
                                     for p in self.file_buffer])

            path = './rollout/' + str(self.idx) + '_' + str(self.save_iter)
            self.save_iter += 1
            print('Saving the file ' + path + '.....')
            np.savez_compressed(path, lengths=lengths,
                                params=params, motions=motions)
            self.file_buffer = []

        return self.obs

    def step(self, action):

        if self.is_rollout:
            action *= 0.0
        self.env.setAction(action)
        self.env.step()
        self.param_count += 1

        if not self.is_rollout:
            self.obs = self.env.getState()
            reward = self.env.getReward()
        else:
            reward = 0.0

        done = False
        info = {}
        info['end'] = self.env.isEOE()

        if info['end'] != 0:
            done = True

        if self.isTwoLevelActuactor and not self.is_rollout:
            mt = self.env.getRandomMuscleTuple()
            for i in range(len(mt)):
                self.muscle_tuples[i].append(mt[i])

        if self.is_rollout:
            self.rollout_count += 1
            self.episode_buffer[1].append(
                np.append(self.env.getPositions(), self.env.getNormalizedPhase()))
            if self.rollout_count >= 150:
                done = True
                self.rollout_count = 0

        return self.obs, reward, done, info

    def get_muscle_tuple(self, idx):
        assert (self.isTwoLevelActuactor)
        res = np.array(self.muscle_tuples[idx], dtype=np.float32)
        if self.isTwoLevelActuactor:
            self.muscle_tuples[idx] = []  # = [[],[],[]]
        return res

    def load_muscle_model_weights(self, w):
        self.env.setMuscleNetworkWeight(convert_to_torch_tensor(ray.get(w)))

    def set_is_rollout(self):
        self.is_rollout = True

    def set_idx(self, idx):
        self.idx = idx


def createEnv():
    return MyEnv()


if __name__ == "__main__":
    print("MAIN")
    e = MyEnv("../data/env.xml")
    e.reset()
