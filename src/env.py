"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp
from tqdm import tqdm
import json


class Monitor:
    def __init__(self, width, height, saved_path):

        # self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
        #                 "-pix_fmt", "rgb24", "-r", "30", "-i", "-", "scale=800:600",   "-an", "-vcodec", "mpeg4", saved_path]
        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s",
                         "{}X{}".format(width, height), "-pix_fmt", "rgb24", "-r", "30", "-i", "-",
                           "-vf", "scale=800:600", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())
        # Non-blocking write + progress
        if not hasattr(self, "_pbar"):
            try:
                self._pbar = tqdm(total=None, desc="Recording", unit="frame", dynamic_ncols=True, mininterval=0.5)
            except Exception:
                self._pbar = None
        if getattr(self, "_pbar", None) is not None:
            self._pbar.update(1)


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, _, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        score_now = float(info.get("score", 0))
        reward = (score_now - self.curr_score) / 40.0
        self.curr_score = score_now

        self.current_x = info["x_pos"]
        return state, reward, done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


def create_train_env(world, stage, actions, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)
    env = CustomReward(env, world, stage, monitor)
    env = CustomSkipFrame(env)
    return env

# kept the actinon_type argument for backward compatibility
class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs, output_path=None, actions_json="Actions.json"):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []
        # if action_type == "right":
        #     actions = RIGHT_ONLY
        # elif action_type == "simple":
        #     actions = SIMPLE_MOVEMENT
        # else:
        #     actions = COMPLEX_MOVEMENT
        with open(actions_json, "r") as f:
            actions = json.load(f)["actions"]
        self.envs = [create_train_env(world, stage, actions, output_path=output_path) for _ in range(num_envs)]
        self.num_states = self.envs[0].observation_space.shape[0]
        self.num_actions = len(actions)
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.processes.append(process)
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == "reset":
                self.env_conns[index].send(self.envs[index].reset())
            elif request == "close":
                self.env_conns[index].close()
                break
            else:
                raise NotImplementedError
    
    def close(self):
        for c in self.agent_conns:
            print("closing agent con: " + str(c))
            c.send(("close", None))
            c.close()
        for p in self.processes:
            print("closing process: " + str(p))
            p.join(timeout = 5)
            if p.is_alive():
                p.terminate()
    

