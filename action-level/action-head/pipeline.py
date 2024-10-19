import os
import torch
import torch.nn as nn
from typing import *
from transformers.utils.dummy_pt_objects import PreTrainedModel

class Pipeline(nn.Module):
    '''
    Pipeline for viewport prediction.
    '''
    def __init__(self,
                plm: PreTrainedModel,
                loss_func = None,
                fut_window = None,
                device = 'cuda',
                embed_size = 1024,
                frequency = 5,
                using_multimodal = False,
                dataset = None
                ):
        """
        :param plm: the pretrained llm
        :param embed_size: the embed size of llm
        :param frequency: the frequency of dataset
        :param fut_window: future (prediction) window
        :param dataset: the dataset
        :param using_multimodal: adding multimodal image features (True/False)
        :param device: cuda or cpu
        """
        super().__init__()
        self.plm = plm
        self.using_multimodal = using_multimodal
        self.dataset = dataset
        self.device = device
        self.frequency = frequency
        self.embed_size = embed_size
        self.fut_window_length = fut_window

        self.embed_vp = nn.Linear(100, self.embed_size).to(device)
        self.embed_ln = nn.LayerNorm(self.embed_size).to(device)

        self.loaded_tensor_cache = {}
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.embed_vp, self.embed_multimodal, self.embed_ln, self.conv1d, self.plm.networking_head
        ])

        if loss_func is None:
            loss_func = nn.MSELoss()
        self.loss_fct = loss_func
        self.fut_window = fut_window
    
    def forward(self, batch, future, video_user_position, teacher_forcing=True) -> torch.Tensor:
        """
        :param batch: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the loss value for training
        """
        if teacher_forcing:
            pred = self.teaching_forcing(batch, future, video_user_position)
        else:
            pred = self.auto_regressive(batch, future, video_user_position)
        gt = future.to(pred.device)
        loss = self.loss_fct(pred, gt)
        return loss
    
    def auto_regressive(self, x, future, video_user_position) -> torch.Tensor:
        """
        auto-regressive generation
        
        :return: the loss value for training
        """
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.embed_vp(x[:,1,:]).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:  # we make using multimodal image features as an option, as not all datasets provide video information.
            mapped_tensor = self.get_multimodal_information(video_user_position)
            x = torch.cat([mapped_tensor, x], dim=1)

        x = self.embed_ln(x)

        outputlist = []
        for _ in range(self.fut_window_length):
            outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device))
            outputlist.append(outputs.logits)
            x = torch.cat((x, self.embed_vp(self.conv1d(outputs.logits)).unsqueeze(1)), dim=1)

        pred = torch.cat(outputlist, dim=1)
        return pred
    
    def teaching_forcing(self, x, future, video_user_position) -> torch.Tensor:
        """
        teaching-forcing generation

        :param x: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the return value by llm
        """

        x = torch.cat((x, future), dim=1)
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.embed_vp(x[:, i, :]).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        x = self.embed_ln(x)

        outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device), teacher_forcing=True)
        return outputs.logits
    
    def inference(self, batch, future, video_user_info) -> torch.Tensor:
        """
        Inference function. Use it for testing.
        """
        pred = self.auto_regressive(batch, future, video_user_info)
        gt = future.to(pred.device)
        return pred, gt
    
    