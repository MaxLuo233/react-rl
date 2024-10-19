import torch
import torch.nn as nn

class PPOHead(nn.Module):
    """
    A simple linear layer as action head for action-level PPO.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_head = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=True),  
            nn.Tanh()
        )
    
    def forward(self, input_logits):
        last_one = input_logits.shape[1]
        needed_logits = input_logits[:, last_one-1, :]
        needed_logits = torch.unsqueeze(needed_logits, dim=1)
        prediction = self.action_head(needed_logits)
        return prediction
