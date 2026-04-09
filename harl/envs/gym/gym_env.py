import copy
import gym
import numpy as np
from collections.abc import Iterable
from harl.envs.gym.lbforaging_custom import register_custom_lbforaging_envs
try:
    import gymnasium
except ImportError:
    gymnasium = None


class GYMEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]
        self.lbforaging = self._is_lbforaging_scenario(self.scenario)
        if self.lbforaging:
            register_custom_lbforaging_envs()
            # DISSCv2 configs use "lbforaging:<env-id>" while custom registration uses "<env-id>".
            self.scenario = self.scenario.split(":", 1)[-1]
            if gymnasium is None:
                raise ImportError(
                    "gymnasium is required for lbforaging scenarios. Install gymnasium and lbforaging."
                )
            self.env = gymnasium.make(self.scenario)
        else:
            self.env = gym.make(self.scenario)
        self.n_agents = 1
        self.share_observation_space = [self.env.observation_space]
        self.observation_space = [self.env.observation_space]
        self.action_space = [self.env.action_space]
        if self.env.action_space.__class__.__name__ == "Box":
            self.discrete = False
        else:
            self.discrete = True
        if self.lbforaging:
            self.n_agents = len(self.env.observation_space)
            self.share_observation_space = list(self.env.observation_space)
            self.observation_space = list(self.env.observation_space)
            self.action_space = list(self.env.action_space)
            self.discrete = True

    @staticmethod
    def _is_lbforaging_scenario(scenario):
        return scenario.startswith("lbforaging") or "Foraging" in scenario

    @staticmethod
    def _reset_unpack(reset_output):
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            return reset_output[0]
        return reset_output

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if self.lbforaging:
            step_out = self.env.step(actions.flatten())
            if len(step_out) == 5:
                obs, rew, terminated, truncated, info = step_out
                done = np.logical_or(terminated, truncated)
                if np.all(done) and np.all(np.asarray(truncated)):
                    info["bad_transition"] = True
            else:
                obs, rew, done, info = step_out
                if np.all(done) and info.get("TimeLimit.truncated", False):
                    info["bad_transition"] = True
            if not isinstance(done, Iterable):
                done = [done] * self.n_agents
            rew = [[float(np.sum(rew))]] * self.n_agents
            return obs, obs, rew, done, [info], self.get_avail_actions()

        if self.discrete:
            step_out = self.env.step(actions.flatten()[0])
        else:
            step_out = self.env.step(actions[0])
        if len(step_out) == 5:
            obs, rew, terminated, truncated, info = step_out
            done = terminated or truncated
            if done and truncated:
                info["bad_transition"] = True
        else:
            obs, rew, done, info = step_out
            if done and info.get("TimeLimit.truncated", False):
                info["bad_transition"] = True
        return [obs], [obs], [[rew]], [done], [info], self.get_avail_actions()

    def reset(self):
        """Returns initial observations and states"""
        reset_out = self.env.reset()
        obs = [self._reset_unpack(reset_out)]
        s_obs = copy.deepcopy(obs)
        if self.lbforaging:
            return obs[0], s_obs[0], self.get_avail_actions()
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.lbforaging:
            return [[1] * self.action_space[0].n for _ in range(self.n_agents)]
        if self.discrete:
            avail_actions = [[1] * self.action_space[0].n]
            return avail_actions
        else:
            return None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        else:
            self.env.reset(seed=seed)
