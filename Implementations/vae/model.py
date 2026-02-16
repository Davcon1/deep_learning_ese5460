import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dims, latent_dim=32, layers=[2048, 1024, 512]):
        super(Encoder, self).__init__()
        stored_layers = []
        stored_layers.append(nn.Linear(input_dims, layers[0]))
        stored_layers.append(nn.ELU())

        # generate network layers based on the list of parameters that we're going to pass
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            stored_layers.append(layer)
            activation = nn.ELU()
            stored_layers.append(activation)
        
        self.model = nn.Sequential(*stored_layers)
        self.logvar = nn.Linear(layers[-1], latent_dim)
        self.mean = nn.Linear(layers[-1], latent_dim)
    
    def forward(self, joint_states, anchor_pose):
        inputs = torch.cat([joint_states, anchor_pose], dim=-1)
        output = self.model(inputs)
        mean = self.mean(output)
        logvar = self.logvar(output)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, output_dims, input_dims, layers=[512, 1024, 2048]):
        super(Decoder, self).__init__()
        stored_layers = []
        stored_layers.append(nn.Linear(input_dims, layers[0]))
        stored_layers.append(nn.ELU())

        # same as encoder, generate the network based on list of hidden states
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            stored_layers.append(layer)
            activation = nn.ELU()
            stored_layers.append(activation)
        
        stored_layers.append(nn.Linear(layers[-1], output_dims))
        self.model = nn.Sequential(*stored_layers)
    
    def forward(self, latents, c):
        inputs = torch.cat([latents, c], dim=-1)
        return self.model(inputs)


class VAE(nn.Module):
    def __init__(self, input_dims, output_dims, latent_dims, c_dims, layers=[2048, 1024, 512]):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dims, latent_dims, layers)
        self.decoder = Decoder(output_dims, latent_dims + c_dims, layers[::-1])
    
    def reparameterize(self, mean, logvar):
        # use the reparametarization trick that we learn in class
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, joint_states, anchor_pose, c):
        mean, logvar = self.encoder(joint_states, anchor_pose)
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        # Sample z using reparameterization trick
        z = self.reparameterize(mean, logvar)

        # Decode
        output = self.decoder(z, c)
        
        # return mean, logvar (for loss calculation), z (for saving), and output
        return mean, logvar, z, output
    
    def kl_divergence(self, mu, logvar):
        mu = torch.clamp(mu, min=-10, max=10)
        logvar = torch.clamp(logvar, min=-10, max=2)
        # remember that mu and logvar are going to be of size [batch_size, latents]
        # we want to sum over the latents so we take the summation over the last dimension
        divergence_kl = -0.5 * torch.sum(1 - mu**2 + logvar - torch.exp(logvar), dim=-1)
        return torch.mean(divergence_kl)
        
    def loss(self, pred, actual, mu, logvar, beta):
        reconstruction_loss = torch.mean((pred - actual) ** 2)
        kl_divergence_loss = self.kl_divergence(mu, logvar)
        # calculate the loss with respect to the beta value that is used in the paper 
        total_loss = reconstruction_loss + kl_divergence_loss * beta
        return total_loss, reconstruction_loss, kl_divergence_loss