from torch import nn
import torch
from einops.einops import rearrange, repeat
import random
import numpy as np
class Initializer(object):
    @staticmethod
    def manual_seed(seed):
        """
        Set all of random seed to seed.
        --------------------
        Arguments:
                seed (int): seed number.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def xavier_normal(module, lstm_forget_bias_init=2):
        """
        Xavier Gaussian initialization.
        """
        lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
        normal_classes = (nn.Conv2d, nn.Linear, nn.Embedding)
        recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
        if any([isinstance(module, cl) for cl in normal_classes]):
            nn.init.xavier_normal_(
                module.weight.data
            ) if module.weight.requires_grad else None
            try:
                module.bias.data.fill_(0) if module.bias.requires_grad else None
            except AttributeError:
                pass
        elif any([isinstance(module, cl) for cl in recurrent_classes]):
            for name, param in module.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(param.data) if param.requires_grad else None
                elif name.startswith("bias"):
                    if param.requires_grad:
                        hidden_size = param.size(0)
                        param.data.fill_(0)
                        param.data[
                            hidden_size // 4 : hidden_size // 2
                        ] = lstm_forget_bias_init

    @staticmethod
    def xavier_uniform(module, lstm_forget_bias_init=2):
        """
        Xavier Uniform initialization.
        """
        lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
        normal_classes = (nn.Conv2d, nn.Linear, nn.Embedding)
        recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
        if any([isinstance(module, cl) for cl in normal_classes]):
            nn.init.xavier_uniform_(
                module.weight.data
            ) if module.weight.requires_grad else None
            try:
                module.bias.data.fill_(0) if module.bias.requires_grad else None
            except AttributeError:
                pass
        elif any([isinstance(module, cl) for cl in recurrent_classes]):
            for name, param in module.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_uniform_(param.data) if param.requires_grad else None
                elif name.startswith("bias"):
                    if param.requires_grad:
                        hidden_size = param.size(0)
                        param.data.fill_(0)
                        param.data[
                            hidden_size // 4 : hidden_size // 2
                        ] = lstm_forget_bias_init

    @staticmethod
    def orthogonal(module, lstm_forget_bias_init=2):
        """
        Orthogonal initialization.
        """
        lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
        normal_classes = (nn.Conv2d, nn.Linear, nn.Embedding)
        recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
        if any([isinstance(module, cl) for cl in normal_classes]):
            nn.init.orthogonal_(
                module.weight.data
            ) if module.weight.requires_grad else None
            try:
                module.bias.data.fill_(0) if module.bias.requires_grad else None
            except AttributeError:
                pass
        elif any([isinstance(module, cl) for cl in recurrent_classes]):
            for name, param in module.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(param.data) if param.requires_grad else None
                elif name.startswith("bias"):
                    if param.requires_grad:
                        hidden_size = param.size(0)
                        param.data.fill_(0)
                        param.data[
                            hidden_size // 4 : hidden_size // 2
                        ] = lstm_forget_bias_init