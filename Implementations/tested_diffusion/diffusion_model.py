import torch
import torch.nn as nn
import math
from sin_embedding import SinusoidalEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionTransformer(nn.Module):
    def __init__(self, embedding_dims = 512,  history_length = 4, prediction_horizon = 8, state_dim = 64, latent_dim = 32,
                attention_heads = 8, num_layers = 6):
        super().__init__()
        # save parameters to use
        self.embedding_dims = embedding_dims
        self.history_length = history_length
        self.pred_horizon = prediction_horizon
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.total_sequence_length = 2 * (history_length + 1 + prediction_horizon)

        # define projections in
        self.project_state = nn.Linear(state_dim, embedding_dims)
        self.project_latent = nn.Linear(latent_dim, embedding_dims)

        # define projections out
        self.project_state_out = nn.Linear(embedding_dims, state_dim)
        self.project_latent_out = nn.Linear(embedding_dims, latent_dim)

        # position, token type, and time embeddings
        self.token_type_embed = nn.Embedding(2, embedding_dims)
        self.pos_embed = nn.Parameter(torch.randn(1, self.total_sequence_length, embedding_dims) * 0.02)
        self.time_embedding = SinusoidalEmbedding(embedding_dims)

        # actual transformer model
        encoding_layer = nn.TransformerEncoderLayer(
            d_model = embedding_dims,
            nhead = attention_heads,
            dim_feedforward=embedding_dims * 4,
            dropout = 0.1, 
            activation = 'gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoding_layer, num_layers)
    
    def build_sequence(self, states, latents):
        '''
        input is a sequence of states, and a sequence of latents
        essentially, we are concatenating the states and latents that are in 
        state_0, ..., state_k
        latent_0, ..., latent_k
        to the form
        (state_0, latent_0, ..., state_k, latent_k)
        '''

        batch_size = states.shape[0]
        num_timesteps = states.shape[1]
        # embedded state is of dimensionality [Batch, history_length + 1 + prediction_horizon, own_dim]
        embedded_states = self.project_state(states)
        embedded_latents = self.project_latent(latents)

        state_type = torch.zeros(batch_size, num_timesteps, dtype=torch.long, device=states.device)
        latent_type = torch.ones(batch_size, num_timesteps, dtype=torch.long, device=states.device)
        
        state_embeds = embedded_states + self.token_type_embed(state_type)
        latent_embeds = embedded_latents + self.token_type_embed(latent_type)

        output = torch.zeros((batch_size, self.total_sequence_length, self.embedding_dims), device=device)
        output[:, 0::2] = state_embeds
        output[:, 1::2] = latent_embeds

        return output
    
    def deconstruct_sequence(self, output):
        '''
        output is composed of a sequence of states and latents
        we need to do undo the operation that build sequence did
        '''
        batch_size = output.shape[0]

        state_embeds = output[:, 0::2]
        latent_embeds = output[:, 1::2]
        pred_states = self.project_state_out(state_embeds)
        pred_latents = self.project_latent_out(latent_embeds)

        return pred_states, pred_latents

    def forward(self, states, latents, diffusion_steps):
        # first construct the sequence into a form that can be processed
        h = self.build_sequence(states, latents)
        batch_size = h.shape[0]
        seq_len = h.shape[1]
        if diffusion_steps.dim() == 1:
            diffusion_steps = diffusion_steps.unsqueeze(1).expand(-1, seq_len)

        # add positional and time embedding
        h += self.pos_embed[:, :seq_len, :]
        time_embedding = self.time_embedding(diffusion_steps.float())
        h += time_embedding

        # run the sequence through the transformer
        h = self.transformer(h)

        #deconstruct the sequence
        pred_states, pred_latents = self.deconstruct_sequence(h)
        return pred_states, pred_latents