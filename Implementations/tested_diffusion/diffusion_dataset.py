import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiffusionDataset(Dataset):
    def __init__(self, filePaths, history_length, prediction_length, trajectory_length = 125):
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.trajectory_length = trajectory_length
        self.window_length = self.history_length + 1 + self.prediction_length
        data_container = []

        self.all_states = []
        self.all_latents = []

        self.valid_starts = trajectory_length - self.window_length + 1
        self.num_trajectories = 0

        # Load all of the filepaths and concatenate their states and latents
        for filepath in filePaths:
            data = torch.load(filepath)
            self.num_trajectories += data['num_trajectories']
            joint_pos = data['joint_pos'][:, :trajectory_length]
            joint_vel = data['joint_vel'][:, :trajectory_length]
            base_lin_vel = data['base_lin_vel'][:, :trajectory_length]
            base_ang_vel = data['base_ang_vel'][:, :trajectory_length]
            latents = data['latent'][:, :trajectory_length]

            states = torch.cat([joint_pos, joint_vel, base_lin_vel, base_ang_vel], dim = -1)
            self.all_states.append(states)
            self.all_latents.append(latents)
        
        # store all of the states and latents along the first dimension so that we can sample easier
        # with sliding window
        self.states = torch.cat(self.all_states, dim =0)
        self.latents = torch.cat(self.all_latents, dim =0)

    def __len__(self):
        return self.num_trajectories * self.valid_starts
    
    def __getitem__(self, index):
        '''
        we concatenated the trajectories along a their trajectory index so their dimensionality is
        [total_trajectories, traj_length, dim]
        '''

        # index should call both the trajectory number and where in the trajectory to start
        trajectory_num = index // self.valid_starts
        trajectory_start = index % self.valid_starts
        trajectory_end = trajectory_start + self.window_length

        state = self.states[trajectory_num, trajectory_start:trajectory_end]
        latent = self.latents[trajectory_num, trajectory_start:trajectory_end]
        return state, latent