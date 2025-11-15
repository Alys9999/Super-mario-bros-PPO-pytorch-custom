"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
import torch; torch.backends.cudnn.benchmark = True

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil
import math



def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.0001, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--max_updates", type=int, default=101, help="Stop after this many PPO updates (episodes)")

    parser.add_argument("--resume_from", type=str, default="trained_models/ppo_super_mario_bros_1_1", help="Path to checkpoint .pt to resume from")
    # trained_models/ppo_super_mario_bros_1_1
        # Time proximity shaping
    parser.add_argument('--time_bonus_weight', type=float, default=1.0,
                        help='Weight for the time proximity Gaussian bonus')
    parser.add_argument('--time_center', type=float, default=390.0,
                        help='Target time for Gaussian bonus (seconds)')
    parser.add_argument('--time_sigma', type=float, default=10.0,
                        help='Std for Gaussian in seconds')
    parser.add_argument('--time_source', type=str, choices=['env_remaining', 'elapsed_steps'],
                        default='env_remaining', help='Use env info["time"] or approximate elapsed')

    # Action diversity shaping
    parser.add_argument('--novelty_weight', type=float, default=0.3,
                        help='Per-action novelty reward weight')
    parser.add_argument('--novelty_mode', type=str, choices=['geometric', 'harmonic'], default='geometric',
                        help='Novelty decay: geometric (alpha**c) or harmonic (1/(1+c))')
    parser.add_argument('--novelty_alpha', type=float, default=0.95,
                        help='Alpha for geometric novelty decay, 0<alpha<1')
    parser.add_argument('--novelty_cap', type=int, default=3,
                        help='If >0, stop novelty after M uses per action per episode')
    parser.add_argument('--coverage_bonus', type=float, default=1.0,
                        help='One-time bonus when all actions are used in an episode')


    args = parser.parse_args()
    return args


def train(opt):
    noop_penalty = 0.01


    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.world, opt.stage, "Actions.json", opt.num_processes)
    model = PPO(envs.num_states, envs.num_actions)

    curr_episode = 0
    if opt.resume_from != "None":
        print("Loading model from " + str(opt.resume_from))
        ckpt = torch.load(opt.resume_from, map_location="cpu")
        # handle plain state_dict or dict checkpoint
        if isinstance(ckpt, dict) and "model" in ckpt:
            print("There is a checkpoint dict, loading model and state")
            model.load_state_dict(ckpt["model"])
            curr_episode = int(ckpt.get("episode", 0))
        else:
            print("Loading checkpoint state_dict only")
            model.load_state_dict(ckpt)

    model.share_memory()
    process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
    process.start()

    if torch.cuda.is_available():
        model.cuda()



    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))


    if torch.cuda.is_available():
        curr_states = curr_states.cuda()


    # Episode-local action diversity tracking (per env process)
    ep_action_counts = torch.zeros(opt.num_processes, envs.num_actions, dtype=torch.long)
    ep_action_seen = torch.zeros(opt.num_processes, envs.num_actions, dtype=torch.bool)
    ep_coverage_paid = torch.zeros(opt.num_processes, dtype=torch.bool)
    ep_step_counts = torch.zeros(opt.num_processes, dtype=torch.long)  # used if time_source=elapsed_steps
    ep_prev_x = torch.zeros(opt.num_processes, dtype=torch.float32) # for distance-based time bonus
    # constants for elapsed time approximation (skip=4, ~60 fps)
    ep_time_potential = torch.zeros(opt.num_processes, dtype=torch.float32)
    frameskip = 4
    fps = 60.0


    
    while True:
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(model.state_dict(),
                       "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            torch.save(model.state_dict(),
                       "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            done_flags = [bool(d) for d in done]
            state = torch.from_numpy(np.concatenate(state, 0))
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            




            # Episode time step count (for elapsed time option)
            ep_step_counts += 1

            # Compute per-env novelty + coverage + (optional) time bonus
            act_ids = action.detach().cpu().tolist()

            novelty_terms = []
            coverage_terms = []
            time_terms = []
            NOOP_ID = 0 
            for idx, a in enumerate(act_ids):
                # Novelty bonus
                if a == NOOP_ID:
                    # No novelty for NOOP
                    nov = 0.0
                else:
                    c = int(ep_action_counts[idx, a].item())
                    if opt.novelty_cap > 0 and c >= opt.novelty_cap:
                        nov = 0.0
                    else:
                        if opt.novelty_mode == 'geometric':
                            nov = opt.novelty_weight * (opt.novelty_alpha ** c)
                        else:
                            nov = opt.novelty_weight / (1.0 + c)
                    ep_action_counts[idx, a] += 1
                    ep_action_seen[idx, a] = True
                novelty_terms.append(float(nov))

                # # Mark usage and maybe coverage bonus
                # ep_action_counts[idx, a] += 1
                # ep_action_seen[idx, a] = True
                # if (not ep_coverage_paid[idx]) and bool(ep_action_seen[idx].all().item()):
                #     coverage_terms.append(float(opt.coverage_bonus))
                #     ep_coverage_paid[idx] = True
                # else:
                #     coverage_terms.append(0.0)
                # Coverage bonus (ignore NOOP in the “all used” check)
                coverage_bonus_step = 0.0
                if not ep_coverage_paid[idx]:
                    seen_useful = ep_action_seen[idx].clone()
                    # Mark NOOP as already satisfied so it never blocks coverage
                    if seen_useful.numel() > NOOP_ID:
                        seen_useful[NOOP_ID] = True
                    if bool(seen_useful.all().item()):
                        coverage_bonus_step = float(opt.coverage_bonus)
                        ep_coverage_paid[idx] = True
                coverage_terms.append(coverage_bonus_step)

                # Time proximity bonus as stepwise potential difference (instant reward)
                if opt.time_source == 'env_remaining':
                    t_val = info[idx].get('time', None)
                    if t_val is not None and opt.time_sigma > 0:
                        z = (float(t_val) - opt.time_center) / opt.time_sigma
                        phi_now = opt.time_bonus_weight * math.exp(-0.5 * (z * z))
                    else:
                        phi_now = 0.0
                else:
                    # approximate elapsed seconds: steps * frameskip / fps
                    t_elapsed = float(ep_step_counts[idx].item()) * (frameskip / fps)
                    if opt.time_sigma > 0:
                        z = (t_elapsed - opt.time_center) / opt.time_sigma
                        phi_now = opt.time_bonus_weight * math.exp(-0.5 * (z * z))
                    else:
                        phi_now = 0.0

                t_bonus = float(phi_now - ep_time_potential[idx].item())
                ep_time_potential[idx] = phi_now
                time_terms.append(t_bonus)


            # Add the extras to the reward tensor (device-aware)
            extras = torch.tensor(
                [n + c + t for n, c, t in zip(novelty_terms, coverage_terms, time_terms)],
                device=reward.device, dtype=reward.dtype
            )
            reward = reward + extras
            # Idle penalty based on lack of horizontal progress (discourages standing still)
            for idx, d in enumerate(done_flags):
                x_raw = info[idx].get('x_pos', 0)
                try:
                    x_now = float(getattr(x_raw, "item", lambda: x_raw)())
                except Exception:
                    x_now = 0.0

                if noop_penalty > 0 and not d:
                    delta_x = x_now - ep_prev_x[idx].item()
                    # penalize only near-zero forward progress; no penalty if delta_x < 0 (going back)
                    idle_eps = 0.1
                    if (reward[idx].item() <= 0.0) and (0.0 <= delta_x < idle_eps):
                        reward[idx] -= noop_penalty

                # Update tracker, and reset on episode end
                ep_prev_x[idx] = 0.0 if d else x_now

            # Reset per-episode trackers for envs that just ended
            for idx, d in enumerate(done_flags):
                if d:
                    ep_action_counts[idx].zero_()
                    ep_action_seen[idx].zero_()
                    ep_coverage_paid[idx] = False
                    ep_step_counts[idx] = 0
                    ep_time_potential[idx] = 0.0























            rewards.append(reward)
            dones.append(done)
            curr_states = state
            rollout_mean_reward = torch.stack(rewards).mean().item()
            sum_rewards = torch.stack(rewards).sum(dim=0)

        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[
                                                       batch_indices]))
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        print("Episode: {}. Total loss: {:.4f}. Mean reward: {:.3f}, Sum rewards: {:3f}".format(curr_episode, total_loss.item(), rollout_mean_reward, sum_rewards.mean().item()))
        if opt.max_updates is not None and curr_episode >= opt.max_updates:
            print(f"Reached max_updates={opt.max_updates}. Stopping training.")
            break
    print("closing envs")
    envs.close()
    print("closing train process")
    process.join(timeout=5)
    if process.is_alive(): process.terminate()

    exit()



if __name__ == "__main__":
    opt = get_args()
    train(opt)
