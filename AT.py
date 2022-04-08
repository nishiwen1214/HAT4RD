import torch
import torch.nn as nn
import numpy as np

class AT():   #  Adversarial Training (Fast Gradient Method)
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def perturb(self, epsilon= 5, emb_name='embed.weight'):
        # The emb_name parameter should be replaced with the parameter name of embedding in the model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name == name:  # Find out the embed parameter param
                self.backup[name] = param.data.clone()  # Save the param before disturbing          
                norm = torch.norm(param.grad)  # L2 paradigm for computing gradients
                if norm != 0 and not torch.isnan(norm):
                    r_adv = epsilon * param.grad / norm  # r_adv = ϵ⋅g/||g||2
                    param.data += r_adv
    # Restore param
    def restore(self, emb_name='embed.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name == name: 
                assert name in self.backup
                param.data = self.backup[name]
#         self.backup = {}