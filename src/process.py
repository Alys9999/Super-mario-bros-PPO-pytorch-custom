"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import json
import os
from collections import deque

import torch
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from src.env import create_train_env
from src.model import PPO
from src.recorder import Recorder


def eval(opt, global_model, num_states, num_actions):
    torch.manual_seed(123)
    # if opt.action_type == "right":
    #     actions = RIGHT_ONLY
    # elif opt.action_type == "simple":
    #     actions = SIMPLE_MOVEMENT
    # else:
    #     actions = COMPLEX_MOVEMENT
    with open("Actions.json", "r") as f:
        actions = json.load(f)["actions"]
    env = create_train_env(opt.world, opt.stage, actions)
    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()
    state = torch.from_numpy(env.reset())
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)

        # Uncomment following lines if you want to save model whenever level is completed
        # if info["flag_get"]:
        #     print("Finished")
        #     torch.save(local_model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_step))

        # env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
        if torch.cuda.is_available():
            state = state.cuda()


def _get_actions(action_type):
    if action_type == "right":
        return list(RIGHT_ONLY)
    if action_type == "simple":
        return list(SIMPLE_MOVEMENT)
    if action_type == "complex":
        return list(COMPLEX_MOVEMENT)
    actions_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Actions.json"))
    try:
        with open(actions_path, "r", encoding="utf-8") as f:
            actions_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return actions_config.get("actions", [])


def record_run(
    opt,
    model_path: str,
    num_episodes: int = 1,
    recordings_root: str = "recordings",
    frame_skip: int = 1,
    quality: str = "high",
) -> None:
    action_type = getattr(opt, "action_type", "custom")
    actions = _get_actions(action_type)
    if not actions:
        raise ValueError(f"No actions available for action_type '{action_type}'")
    env = create_train_env(opt.world, opt.stage, actions)
    num_states = env.observation_space.shape[0]
    num_actions = len(actions)

    model = PPO(num_states, num_actions)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    model.eval()

    recorder = Recorder(recordings_root=recordings_root, frame_skip=frame_skip, quality=quality)
    try:
        recorder.start_recording(
            {
                "world": opt.world,
                "stage": opt.stage,
                "model_path": model_path,
                "action_type": getattr(opt, "action_type", "custom"),
                "num_episodes": num_episodes,
                "actions_override": actions,
                # External recorder alignment
                "buttons_order_ext": ["action", "jump", "left", "right", "down"],
                "user_name": "Zy",
                "naming_format": "user_fxxx_axxx_ntxxx.png",
            }
        )

        state = env.reset()
        episode = 0
        timestep = 0

        while episode < num_episodes:
            state_tensor = torch.from_numpy(state).float()
            if use_cuda:
                state_tensor = state_tensor.cuda()

            with torch.no_grad():
                logits, _ = model(state_tensor)
                policy = F.softmax(logits, dim=1)
                action = torch.argmax(policy, dim=1).item()

            next_state, reward, done, info = env.step(action)

            frame = None
            try:
                frame = env.render(mode="rgb_array")
            except Exception:
                frame = None

            action_buttons = actions[action] if 0 <= action < len(actions) else []

            recorder.record_step(
                frame=frame,
                action_idx=action,
                action_buttons=action_buttons,
                reward=float(reward),
                done=bool(done),
                info=info,
                timestep=timestep,
            )

            timestep += 1
            state = next_state

            if done:
                episode += 1
                if episode < num_episodes:
                    state = env.reset()
    finally:
        recorder.stop_recording()
        env.close()
