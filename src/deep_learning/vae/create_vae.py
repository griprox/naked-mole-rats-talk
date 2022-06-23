
import numpy as np
import os
import torch
from torch.distributions import LowRankMultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

#from ava.models.vae_dataset import SyllableDataset
#from ava.plotting.grid_plot import grid_plot


X_SHAPE = (128, 128)
"""Processed spectrogram shape: ``[freq_bins, time_bins]``"""
X_DIM = np.prod(X_SHAPE)
"""Processed spectrogram dimension: ``freq_bins * time_bins``"""



class VAE(nn.Module):
    
    
    def __init__(self, save_dir='', lr=1e-3, z_dim=32, model_precision=10.0, device_name="auto"):
        super(VAE, self).__init__()
        self.save_dir = save_dir
        self.lr = lr
        self.z_dim = z_dim
        self.model_precision = model_precision
        assert device_name != "cuda" or torch.cuda.is_available()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        if self.save_dir != '' and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self._build_network()
        self.optimizer = Adam(self.parameters(), lr=self.lr)
        self.epoch = 0
        self.loss = {'train':{}, 'test':{}}
        self.to(self.device)


    def _build_network(self):
        """Define all the network layers."""
        # Encoder
        
        self.conv1 = nn.Conv2d(1, 8, 3,1,padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3,2,padding=1)
        self.conv3 = nn.Conv2d(8, 16,3,1,padding=1)
        self.conv4 = nn.Conv2d(16,16,3,2,padding=1)
        self.conv5 = nn.Conv2d(16,24,3,1,padding=1)
        self.conv6 = nn.Conv2d(24,24,3,2,padding=1)
        self.conv7 = nn.Conv2d(24,32,3,1,padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(24)
        self.bn7 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(8192,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc31 = nn.Linear(256,64)
        self.fc32 = nn.Linear(256,64)
        self.fc33 = nn.Linear(256,64)
        self.fc41 = nn.Linear(64,self.z_dim)
        self.fc42 = nn.Linear(64,self.z_dim)
        self.fc43 = nn.Linear(64,self.z_dim)
        # Decoder
        self.fc5 = nn.Linear(self.z_dim,64)
        self.fc6 = nn.Linear(64,256)
        self.fc7 = nn.Linear(256,1024)
        self.fc8 = nn.Linear(1024,8192)
        self.convt1 = nn.ConvTranspose2d(32,24,3,1,padding=1)
        self.convt2 = nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1)
        self.convt3 = nn.ConvTranspose2d(24,16,3,1,padding=1)
        self.convt4 = nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1)
        self.convt5 = nn.ConvTranspose2d(16,8,3,1,padding=1)
        self.convt6 = nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1)
        self.convt7 = nn.ConvTranspose2d(8,1,3,1,padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(24)
        self.bn10 = nn.BatchNorm2d(24)
        self.bn11 = nn.BatchNorm2d(16)
        self.bn12 = nn.BatchNorm2d(16)
        self.bn13 = nn.BatchNorm2d(8)
        self.bn14 = nn.BatchNorm2d(8)


    def _get_layers(self):
        """Return a dictionary mapping names to network layers."""
        return {'fc1':self.fc1, 'fc2':self.fc2, 'fc31':self.fc31,
                'fc32':self.fc32, 'fc33':self.fc33, 'fc41':self.fc41,
                'fc42':self.fc42, 'fc43':self.fc43, 'fc5':self.fc5,
                'fc6':self.fc6, 'fc7':self.fc7, 'fc8':self.fc8, 'bn1':self.bn1,
                'bn2':self.bn2, 'bn3':self.bn3, 'bn4':self.bn4, 'bn5':self.bn5,
                'bn6':self.bn6, 'bn7':self.bn7, 'bn8':self.bn8, 'bn9':self.bn9,
                'bn10':self.bn10, 'bn11':self.bn11, 'bn12':self.bn12,
                'bn13':self.bn13, 'bn14':self.bn14, 'conv1':self.conv1,
                'conv2':self.conv2, 'conv3':self.conv3, 'conv4':self.conv4,
                'conv5':self.conv5, 'conv6':self.conv6, 'conv7':self.conv7,
                'convt1':self.convt1, 'convt2':self.convt2,
                'convt3':self.convt3, 'convt4':self.convt4,
                'convt5':self.convt5, 'convt6':self.convt6,
                'convt7':self.convt7}


    def encode(self, x):
        x = x.unsqueeze(1)
        #x = np.expand_dims(x, 1)
        #print(np.shape(x))
        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        x = F.relu(self.conv3(self.bn3(x)))
        x = F.relu(self.conv4(self.bn4(x)))
        x = F.relu(self.conv5(self.bn5(x)))
        x = F.relu(self.conv6(self.bn6(x)))
        x = F.relu(self.conv7(self.bn7(x)))
        x = x.view(-1, 8192)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.relu(self.fc31(x))
        mu = self.fc41(mu)
        u = F.relu(self.fc32(x))
        u = self.fc42(u).unsqueeze(-1) # Last dimension is rank \Sigma = 1.
        d = F.relu(self.fc33(x))
        d = torch.exp(self.fc43(d)) # d must be positive.
        return mu, u, d


    def decode(self, z):
        z = F.relu(self.fc5(z))
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = F.relu(self.fc8(z))
        z = z.view(-1,32,16,16)
        z = F.relu(self.convt1(self.bn8(z)))
        z = F.relu(self.convt2(self.bn9(z)))
        z = F.relu(self.convt3(self.bn10(z)))
        z = F.relu(self.convt4(self.bn11(z)))
        z = F.relu(self.convt5(self.bn12(z)))
        z = F.relu(self.convt6(self.bn13(z)))
        z = self.convt7(self.bn14(z))
        return z.view(-1, X_DIM)


    def forward(self, x, return_latent_rec=False):
        mu, u, d = self.encode(x)
        latent_dist = LowRankMultivariateNormal(mu, u, d)
        z = latent_dist.rsample()
        x_rec = self.decode(z)
        # E_{q(z|x)} p(z)
        elbo = -0.5 * (torch.sum(torch.pow(z,2)) + self.z_dim * np.log(2*np.pi))
        # E_{q(z|x)} p(x|z)
        pxz_term = -0.5 * X_DIM * (np.log(2*np.pi/self.model_precision))
        l2s = torch.sum(torch.pow(x.view(x.shape[0],-1) - x_rec, 2), dim=1)
        pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
        elbo = elbo + pxz_term
        # H[q(z|x)]
        elbo = elbo + torch.sum(latent_dist.entropy())
        if return_latent_rec:
            return -elbo, z.detach().cpu().numpy(), \
                x_rec.view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy()
        return -elbo


    def train_epoch(self, dataset):
        self.train()
        train_loss = 0.0
        for batch_idx, data in enumerate(dataset):
            self.optimizer.zero_grad()
            loss = self.forward(data)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        train_loss /= len(dataset)
        print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, train_loss))
        self.epoch += 1
        return train_loss


    def test_epoch(self, dataset):

        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataset):
                loss = self.forward(data)
                test_loss += loss.item()
        test_loss /= len(dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss
    
    def save_state(self, filename):
        """Save all the model parameters to the given file."""
        layers = self._get_layers()
        state = {}
        for layer_name in layers:
            state[layer_name] = layers[layer_name].state_dict()
        state['optimizer_state'] = self.optimizer.state_dict()
        state['loss'] = self.loss
        state['z_dim'] = self.z_dim
        state['epoch'] = self.epoch
        state['lr'] = self.lr
        state['save_dir'] = self.save_dir
        filename = os.path.join(self.save_dir, filename)
        torch.save(state, filename)
        
        
    def load_state(self, filename):
        """
        Load all the model parameters from the given ``.tar`` file.
        The ``.tar`` file should be written by `self.save_state`.
        Parameters
        ----------
        filename : str
            File containing a model state.
        Note
        ----
        - `self.lr`, `self.save_dir`, and `self.z_dim` are not loaded.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        assert checkpoint['z_dim'] == self.z_dim
        layers = self._get_layers()
        for layer_name in layers:
            layer = layers[layer_name]
            layer.load_state_dict(checkpoint[layer_name])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.loss = checkpoint['loss']
        self.epoch = checkpoint['epoch']

        
    
    def get_latent(self, dataset):
        latent = np.zeros((len(dataset), self.z_dim))
        i = 0
        for data in dataset:
            data = data.to(self.device)
            with torch.no_grad():
                mu, _, _ = self.encode(data)
            mu = mu.detach().cpu().numpy()
            latent[i:i+len(mu)] = mu
            i += len(mu)
        return latent

