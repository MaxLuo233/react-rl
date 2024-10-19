from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from torch.distributions import Categorical


class MistralPPOHeadModel(MistralForCausalLM):
    """
    Mistral with action head.s
    This class is implemented based on MistralForCausalLM.
    """
    _tied_weights_keys = ["action_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.action_head = None
        self.actor = None
        self.critic = None

    def get_action_head(self):
        return self.action_head

    def set_action_head(self, action_head):
        self.action_head = action_head

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_len: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Reimplement forward method for action-level.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(  # self.model输出的是embedding_size
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # 使用Input_embeds
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        prediction = self.action_head(hidden_states)

        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(prediction.view(-1,self.action_head.output_dim),
                             labels.view(-1,self.action_head.output_dim))
        
        if not return_dict:
            return loss

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=logits,
            logits=prediction,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )