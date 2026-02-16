import argparse
import pathlib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vae'))
from model import VAE
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect trajectories for diffusion training.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--wandb_path", type=str, default=None, help="Wandb run path")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_rollouts", type=int, default=100)
parser.add_argument("--rollout_length", type=int, default=150)
parser.add_argument("--output", type=str, default="diffusion_trajectories.pt")
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--motion_name", type=str, default=None)
parser.add_argument("--model_path", type=str, required=True, help="Path to VAE checkpoint")
parser.add_argument("--ou_theta", type=float, default=0.8, help="OU noise mean reversion rate")
parser.add_argument("--ou_sigma", type=float, default=0.1, help="OU noise scale")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import whole_body_tracking.tasks  # noqa: F401

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ornstein-Uhlenbeck Noise generating pipeline with parameters as set in the paper
class OUNoise:
    def __init__(self, shape, theta=0.8, sigma=0.1, mu=0.0, dt=1.0, device='cuda'):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.device = device
        self.shape = shape
        self.reset()
    
    # Reset the current state of the OU noise
    def reset(self):
        self.state = torch.zeros(self.shape, device=self.device)
    
    # Calculate the OU noise given a randomly sampled noise
    def sample(self):
        noise = torch.randn(self.shape, device=self.device)
        # Current state is calculated as previous state + theta * (mu - previous state) * time change + sigma * sqrt(time) * standard guassian noise
        self.state = self.state + self.theta * (self.mu - self.state) * self.dt + self.sigma * torch.sqrt(torch.tensor(self.dt)) * noise
        return self.state


def parse_observations(obs):
    '''
    function to parse through the raw observations received from isaac sim and convert them into understandable observations
    '''
    return {
        'target_joint_pos': obs[..., 0:29],
        'target_joint_vel': obs[..., 29:58],
        'pose_error_pos': obs[..., 58:61],
        'pose_error_ori': obs[..., 61:67],
        'base_lin_vel': obs[..., 67:70],
        'base_ang_vel': obs[..., 70:73],
        'joint_pos': obs[..., 73:102],
        'joint_vel': obs[..., 102:131],
        'last_action': obs[..., 131:160]
    }

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def collect_diffusion_data(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg.scene.num_envs = args_cli.num_envs
    motion_file_path = args_cli.motion_file

    # wandb for extracting motion file
    if motion_file_path is None and args_cli.motion_name:
        import wandb
        api = wandb.Api()
        artifact_name = f"pattheboy135/csv_to_npz/{args_cli.motion_name}:latest"
        art = api.artifact(artifact_name, type='motions')
        motion_file_path = str(pathlib.Path(art.download()) / "motion.npz")

    if motion_file_path is None:
        raise ValueError("Motion file required")
    
    if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'motion'):
        env_cfg.commands.motion.motion_file = motion_file_path

    # make the gym environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    
    num_envs = env.num_envs
    action_dim = env.num_actions

    # create the VAE model with loaded checkpoint
    model = VAE(64, 29, 32, 96).to(device)
    checkpoint = torch.load(args_cli.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # load in the OU noise
    ou_noise = OUNoise(
        shape=(num_envs, action_dim),
        theta=args_cli.ou_theta,
        sigma=args_cli.ou_sigma,
    )

    # used for collecting trajectories
    collected_trajectories = []
    num_successful = 0
    total_attempts = 0
    all_survival_steps = []
    
    print(f"collecting {args_cli.num_rollouts} rollouts with OU noise (theta={args_cli.ou_theta}, sigma={args_cli.ou_sigma})", flush=True)
    print(f"target: {args_cli.rollout_length} steps minimum", flush=True)

    while num_successful < args_cli.num_rollouts:
        obs, _ = env.get_observations()
        # reset OU noise
        ou_noise.reset()
        total_attempts += 1
        
        # track whether or not the environment ended and if we saved the trajectory
        trajectory_data = [[] for _ in range(num_envs)]
        env_done = [False] * num_envs
        env_saved = [False] * num_envs
        env_survival_steps = [0] * num_envs
        
        with torch.inference_mode():
            for step in range(args_cli.rollout_length * 2):
                parsed = parse_observations(obs)
                
                # save parsed states into variables to pass into the VAE
                joint_states = torch.cat([parsed['target_joint_pos'], parsed['target_joint_vel']], dim=-1)
                anchor_pose = torch.cat([parsed['base_lin_vel'], parsed['base_ang_vel']], dim=-1)
                conditioning = torch.cat([
                    parsed['joint_pos'], parsed['joint_vel'],
                    parsed['pose_error_pos'], parsed['pose_error_ori'],
                    parsed['last_action']
                ], dim=-1)

                # sample from VAE and add noise
                mean, logvar, z, vae_action = model(joint_states, anchor_pose, conditioning)          
                noise = ou_noise.sample()
                noisy_action = vae_action + noise

                # update how long each episode has survived so far
                for env_idx in range(num_envs):
                    if not env_done[env_idx]:
                        env_survival_steps[env_idx] = step + 1

                # record the data for the environments that are still running
                for env_idx in range(num_envs):
                    if not env_done[env_idx]:
                        trajectory_data[env_idx].append({
                            'joint_pos': parsed['joint_pos'][env_idx].cpu().clone(),
                            'joint_vel': parsed['joint_vel'][env_idx].cpu().clone(),
                            'base_lin_vel': parsed['base_lin_vel'][env_idx].cpu().clone(),
                            'base_ang_vel': parsed['base_ang_vel'][env_idx].cpu().clone(),
                            'pose_error_pos': parsed['pose_error_pos'][env_idx].cpu().clone(),
                            'pose_error_ori': parsed['pose_error_ori'][env_idx].cpu().clone(),
                            'target_joint_pos': parsed['target_joint_pos'][env_idx].cpu().clone(),
                            'target_joint_vel': parsed['target_joint_vel'][env_idx].cpu().clone(),
                            'latent': z[env_idx].cpu().clone(),
                            'action': vae_action[env_idx].cpu().clone(),
                            'last_action': parsed['last_action'][env_idx].cpu().clone(),
                        })

                obs, _, dones, _ = env.step(noisy_action)
                
                if step == 0:
                    continue

                # check to see which environments have failed and weren't already checked
                for env_idx in range(num_envs):
                    if dones[env_idx] and not env_done[env_idx]:
                        env_done[env_idx] = True
                        
                        # if the trajectory exceeds rollout length, then we can save it
                        if len(trajectory_data[env_idx]) >= args_cli.rollout_length and not env_saved[env_idx]:
                            # we only want to save the rollout length to our collected trajectories
                            collected_trajectories.append(trajectory_data[env_idx][:args_cli.rollout_length])
                            env_saved[env_idx] = True
                            num_successful += 1
                            print(f"{len(trajectory_data[env_idx])} steps", flush=True)
                            
                            if num_successful >= args_cli.num_rollouts:
                                break
                
                if num_successful >= args_cli.num_rollouts:
                    break
        
        # save any trajectories that passed our threshold
        for env_idx in range(num_envs):
            if num_successful >= args_cli.num_rollouts:
                break
            all_survival_steps.append(env_survival_steps[env_idx])
            
            if not env_saved[env_idx] and len(trajectory_data[env_idx]) >= args_cli.rollout_length:
                collected_trajectories.append(trajectory_data[env_idx][:args_cli.rollout_length])
                num_successful += 1
                print(f"env {env_idx}: {len(trajectory_data[env_idx])} steps (truncated to {args_cli.rollout_length}) | Total: {num_successful}/{args_cli.num_rollouts}", flush=True)
        
        # batch summary
        batch_saved = sum(1 for i in range(num_envs) if len(trajectory_data[i]) >= args_cli.rollout_length)
        print(f"\n[Batch {total_attempts}] Saved: {batch_saved}/{num_envs} | Total: {num_successful}/{args_cli.num_rollouts}", flush=True)
        
        if all_survival_steps:
            avg_survival = sum(all_survival_steps) / len(all_survival_steps)
            print(f"Avg survival: {avg_survival:.1f} | Min: {min(all_survival_steps)} | Max: {max(all_survival_steps)}", flush=True)
        print("-" * 60, flush=True)

    # Final summary
    print(f"Total trajectories collected: {num_successful}")
    print(f"Total batches: {total_attempts}")
    print(f"Success rate: {num_successful / (total_attempts * num_envs) * 100:.1f}%")

    save_diffusion_dataset(collected_trajectories, args_cli.output)
    env.close()
    print(f"Saved {len(collected_trajectories)} trajectories to {args_cli.output}")


def save_diffusion_dataset(trajectories, save_path):
    if not trajectories:
        print("No trajectories to save")
        return
        
    processed = {
        'joint_pos': [],
        'joint_vel': [],
        'base_lin_vel': [],
        'base_ang_vel': [],
        'pose_error_pos': [],
        'pose_error_ori': [],
        'target_joint_pos': [],
        'target_joint_vel': [],
        'latent': [],
        'action': [],
        'last_action': [],
    }
    
    for traj in trajectories:
        for key in processed.keys():
            stacked = torch.stack([step[key] for step in traj])
            processed[key].append(stacked)
    
    for key in processed.keys():
        processed[key] = torch.stack(processed[key])
    
    processed['num_trajectories'] = len(trajectories)
    processed['trajectory_length'] = len(trajectories[0]) if trajectories else 0
    
    torch.save(processed, save_path)

if __name__ == "__main__":
    collect_diffusion_data()
    simulation_app.close()