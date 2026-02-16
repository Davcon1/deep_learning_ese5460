import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Diffusion_Scheduler():
# Use this to generate the new trajectories during the diffusion process
    def __init__(self,  beta_begin, beta_end, steps = 20, cosine = False):
        self.steps = steps

        # use linear scaling
        if not cosine:
            self.schedule = torch.linspace(beta_begin, beta_end, steps = steps).to(device)
            self.beta = self.schedule
            self.alpha = 1 - self.schedule
            self.cumulAlpha = torch.cumprod(self.alpha, dim = 0).to(device)
        # use cosine scaling
        else:
            lin_space = torch.linspace(0, steps, steps=steps + 1).to(device)
            cumulAlpha = torch.cos(((lin_space / steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            self.cumulAlpha = cumulAlpha / cumulAlpha[0]
            self.beta = torch.zeros_like(self.cumulAlpha)
            self.beta[1:] = 1 - (self.cumulAlpha[1:] / self.cumulAlpha[:-1]).clamp(min=0.0, max=0.999)
            self.alpha = 1 - self.beta[1:]
            self.cumulAlpha = torch.cumprod(self.alpha, dim=0).to(device)

    def add_noise_per_token(self, x0, timesteps):
        '''
        need to add noise based on the uniformly sampled noise for each of the latents and states
        '''
        #timesteps is of dimensionality (batch_size, seq_len)
        #x0 is of size (batch_size, seq_len, state_dim / latent_dim)
        cum_alpha_timesteps = self.cumulAlpha[timesteps].unsqueeze(-1)
        first_term = torch.sqrt(cum_alpha_timesteps)
        second_term = torch.sqrt(1 - cum_alpha_timesteps)
        eps = torch.randn_like(x0)

        noise = eps * second_term
        noised_data = first_term * x0 + noise
        return noised_data, eps
        
    def step_per_token( self, pred_x0, timesteps, sample, inpainting_mask, clean_data):
        '''
        need to handle per token steps --- this esentially means that each of the state / latent 
        tokens in the mdoel is at its own denoising step (where some may be clean)
        '''

        # subtract 1 from every timestep to get the previous timestep for each
        prev_timesteps = (timesteps - 1).clamp(min=0)

        # get the cummulated alpha for each
        cumulAlpha = self.cumulAlpha[timesteps].unsqueeze(-1)
        cumulAlphaPrev = self.cumulAlpha[prev_timesteps].unsqueeze(-1)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        pred_noise = (sample - torch.sqrt(cumulAlpha) * pred_x0) / torch.sqrt(1 - cumulAlpha)
        output = torch.sqrt(cumulAlphaPrev) * pred_x0 + torch.sqrt(1 - cumulAlphaPrev) * pred_noise

        # if an inpainting mask is used (at inference time)
        if inpainting_mask is not None and clean_data is not None:
            mask = inpainting_mask.unsqueeze(-1).float()
            output = mask * clean_data + (1 - mask) * output
        return output