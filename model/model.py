"""
reference: https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (BertPreTrainedModel, BertModel, BertConfig,
                          RobertaModel, PreTrainedModel, RobertaConfig)

# for debugging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class RobertaForShuffleRandomThreeWayClassification(RobertaPreTrainedModel):
    """RoBERTa model for pre-training with three-way shuffled/random/non-replaced classification.

    References:
        https://huggingface.co/transformers/model_doc/roberta.html?highlight=roberta#transformers.RobertaModel
    """
    def __init__(self, config):
        super().__init__(config)

        # if add_pooling_layer is `True`, this will add a dense layer 
        # + `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # for token-level three-way classification
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, 3) # 0 (non-replaced), 1 (shuffled), 2 (random)

        self.init_weights()
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        shuffle_random_mask=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hs)
        logits = self.dense(sequence_output) # -> (bs, seq_len, 3)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 3), shuffle_random_mask.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output)


class RobertaForFirstCharPrediction(RobertaPreTrainedModel):
    """RoBERTa model for pre-training with masked first character classification."""
    def __init__(self, config):
        super().__init__(config)

        # if add_pooling_layer is `True`, this will add a dense layer 
        # + `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # for token-level four-way classification
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, 29)
        # -> 0~25: alphabet, 26: digit, 27: punctuation, 28: exception

        self.init_weights()
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        masked_word_labels=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hs)
        logits = self.dense(sequence_output) # -> (bs, seq_len, 29)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 29), masked_word_labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output)


class RobertaForAsciiValuePrediction(RobertaPreTrainedModel):
    """RoBERTa model for pre-training with masked ascii value classification."""
    def __init__(self, config):
        super().__init__(config)

        # if add_pooling_layer is `True`, this will add a dense layer 
        # + `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # for token-level four-way classification
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, 5)
        # -> 0~4: classes

        self.init_weights()
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        masked_word_labels=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hs)
        logits = self.dense(sequence_output) # -> (bs, seq_len, 5)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 5), masked_word_labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output)


class RobertaForRandomValuePrediction(RobertaPreTrainedModel):
    """RoBERTa model for pre-training with masked completely random value classification."""
    def __init__(self, config):
        super().__init__(config)

        # if add_pooling_layer is `True`, this will add a dense layer 
        # + `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # for token-level four-way classification
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, 5)
        # -> 0~4: classes

        self.init_weights()
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        masked_word_labels=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hs)
        logits = self.dense(sequence_output) # -> (bs, seq_len, 5)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 5), masked_word_labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output)
