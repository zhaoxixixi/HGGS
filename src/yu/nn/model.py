from typing import List
import numpy as np

import torch
from torch import nn
from itertools import product

class MLPModel(nn.Module):
    mlp: nn.Sequential

    def __init__(self, nn_layers: List[int], nn_norm: List[str] = None, p: float = 0.5,
                 init_strategy: str = 'kaiming'):
        super().__init__()
        assert init_strategy in ['kaiming', 'xiver'], 'Error Parameters Initial'
        self.init_strategy = init_strategy
        self.mlp = nn.Sequential()

        for i in range(1, len(nn_layers)):
            self.mlp.append(nn.Linear(nn_layers[i - 1], nn_layers[i]))
            # Batch Norm
            if nn_norm and i < len(nn_norm) and nn_norm[i]:
                norm = getattr(nn, nn_norm[i], None)
                if norm:
                    if p > 0:
                        self.mlp.append(nn.Dropout(p))
                    self.mlp.append(norm(nn_layers[i]))

            # ReLU
            if i != len(nn_layers) - 1:
                self.mlp.append(nn.ReLU())

        # print(self.mlp)
        # change the model precision
        self.mlp.double()

    def forward(self, x):
        ans = self.mlp(x)
        return ans

    def reset_weights(self, path=None):
        """ reset model weights to avoid weight leakage """
        if path:
            print('load pretrained model from:', path)
            self.load_state_dict(torch.load(path))
        def _reset_weights(m):
            for layer in self.children():
                if hasattr(layer, 'reset_parameters'):
                    if self.init_strategy == 'xiver':
                        nn.init.xavier_uniform(layer.weight.data)
                        nn.init.xavier_uniform(layer.bias.data)
                    else:
                        layer.reset_parameters()
        self.apply(_reset_weights)

    def save(self, save_file: str):
        torch.save(self.state_dict(), save_file)

    def get_last_layer_grad(self, inputs, targets) -> np.array:
        self.eval()
        with torch.no_grad():
            penu_out = inputs
            outputs = inputs
            for i in range(len(self.mlp)):
                if i == len(self.mlp) - 1:
                    outputs = self.mlp[i](penu_out)
                    break
                penu_out = self.mlp[i](penu_out)
            gradient_norm = torch.norm((outputs - targets), dim = -1).detach().cpu().numpy().reshape(-1, 1)
            penu_out = penu_out.detach().cpu().numpy()
            grad_embedding = gradient_norm * penu_out
        return grad_embedding
        # return self.mlp[-1].weight.grad

def get_model(nn_model, nn_layers, nn_norm, dropout, 
                      init_strategy):
    if nn_model == 'MLP':
        return MLPModel(nn_layers, nn_norm=nn_norm, p=dropout,
                     init_strategy=init_strategy)