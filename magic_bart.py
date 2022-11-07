import collections
import logging
import math
import os
import pdb
import numpy as np
import random
import subprocess
import sys
from transformers import BartTokenizerFast
from transformers.trainer_pt_utils import (find_batch_size, nested_concat,
                                           nested_numpify, IterableDatasetShard,
                                           nested_truncate)
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Tuple, Union, Dict, Any, List, NamedTuple
from module import *
# from module_copy import *
from attr import dataclass
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize)
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils_base import PaddingStrategy

import torch
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
from packaging import version
from transformers import BartForCausalLM

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput, \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartConfig,
    BartEncoder,
    BartPretrainedModel,
    _expand_mask, _make_causal_mask,
    BartLearnedPositionalEmbedding, BartAttention,
)
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import numpy as np



class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]
    q_e_concat: Union[np.ndarray, Tuple[np.ndarray]]


class MyBartConfig(BartConfig):
    def __init__(self, margin_model=False,
                 **kwargs):
        super(MyBartConfig, self).__init__(**kwargs)
        self.margin_model = margin_model


class MyBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = MyBartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            entity_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            # addi_source=None,
            # addi_source_attention_mask=None,
            # addi_source_encoder_outputs=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # addi_source_encoder_outputs = self.encoder(
            #     input_ids=addi_source,
            #     attention_mask=addi_source_attention_mask
            # )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            # addi_source_encoder_outputs = BaseModelOutput(
            #     last_hidden_state=addi_source_encoder_outputs[0],
            #     hidden_states=addi_source_encoder_outputs[1] if len(addi_source_encoder_outputs) > 1 else None,
            #     attentions=addi_source_encoder_outputs[2] if len(addi_source_encoder_outputs) > 2 else None,
            # )

        if entity_outputs is not None:
            entity_attention_mask = 1 - entity_outputs[:, :, 0].eq(0).int()

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            entity_hidden_states=entity_outputs,
            encoder_attention_mask=attention_mask,
            entity_attention_mask=entity_attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class MyBart(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: MyBartConfig):
        super().__init__(config)
        self.config = config
        self.model = MyBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.lm_model = BartForCausalLM.from_pretrained('checkpoint-8972', add_cross_attention=False)

        self.init_weights()
        self.graph = GraphBartEncoder(config)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            #
            questions_inputs=None,
            map_entity_spans=None,
            map_sent_spans=None,
            sent_entity_edge=None,
            answer_index=None,
            answers=None,
            entity_outputs=None,
            questions_output=None

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        q_e_concat = None
        if encoder_outputs is None:
            encoder_outputs = self.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if questions_output is None:
            questions_inputs = questions_inputs.view(-1, 15)  # (batch*10,15)
            question_mask = 1 - questions_inputs.eq(1).float()
            questions_output = self.get_encoder()(
                input_ids=questions_inputs,
                attention_mask=question_mask,
            )[0]  # [batch*10,15,1024]
            question_mask = question_mask.view(input_ids.shape[0], -1, 15)
            questions_output = questions_output.view(input_ids.shape[0], -1, 15, 1024)
            # questions_output = torch.mean(questions_output, 1)  # [batch*10,1024]

        if entity_outputs is None :
            entity_outputs, q_e_concat = self.graph(document_output=encoder_outputs[0],
                                                    map_entity_spans=map_entity_spans,
                                                    map_sent_spans=map_sent_spans,
                                                    sent_entity_edge=sent_entity_edge,
                                                    questions_output=questions_output,
                                                    question_mask=question_mask)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            entity_outputs=entity_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # dict (['last_hidden_state', 'past_key_values', 'encoder_last_hidden_state']
        if self.model.config.margin_model:
            with torch.no_grad():
                tmp_labels = labels.clone()
                tmp_labels.masked_fill_(tmp_labels == -100, 1)
                tmp_labels = shift_tokens_right(
                    tmp_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                zero_logits = self.lm_model(tmp_labels).logits
        else:
            zero_logits = 'None'
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  # [1,vocab_size]

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            zero_logits=zero_logits,
            q_e_concat=q_e_concat
        )

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs) -> Dict[
        str, Any]:
        encoder = self.get_encoder()
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if
            not argument.startswith("decoder_") and not 'use_cache' in argument and (
                not argument in ['questions_inputs', 'map_entity_spans', 'map_sent_spans',
                                 'sent_entity_edge']
            )
        }
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        questions_inputs = model_kwargs['questions_inputs']
        questions_inputs = questions_inputs.view(-1, 15)  # (batch*10,15)
        question_mask = 1 - questions_inputs.eq(1).float()
        questions_output = self.get_encoder()(
            input_ids=questions_inputs,
            attention_mask=question_mask,
        )[0]  # [batch*10,15,1024]
        question_mask = question_mask.view(input_ids.shape[0], -1, 15)
        questions_output = questions_output.view(input_ids.shape[0], -1, 15, 1024)
        model_kwargs["entity_outputs"] = self.graph(
            document_output=model_kwargs["encoder_outputs"][0],
            questions_output=questions_output,
            map_entity_spans=model_kwargs['map_entity_spans'],
            map_sent_spans=model_kwargs['map_sent_spans'],
            sent_entity_edge=model_kwargs['sent_entity_edge'],
            question_mask=question_mask)[0]
        model_kwargs["questions_output"] = questions_output
        model_kwargs["question_mask"] = question_mask

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            entity_outputs=None,
            questions_output=None,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

            entity_outputs = entity_outputs.index_select(
                0, expanded_return_idx.to(entity_outputs.device)
            )
            model_kwargs["entity_outputs"] = entity_outputs

            questions_output = questions_output.index_select(
                0, expanded_return_idx.to(questions_output.device)
            )
            model_kwargs["questions_output"] = questions_output

        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "entity_outputs": kwargs['entity_outputs'],
            "questions_output": kwargs['questions_output']

        }

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@dataclass
class MyDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    model_args: object = None

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        questions_inputs = [feature["questions_inputs"] for feature in features] if "questions_inputs" in features[
            0].keys() else None
        if questions_inputs is not None:
            new_questions_inputs = []
            max_question_number = max(len(l) for l in questions_inputs)
            # if max_question_number>max_qa:
            #     max_question_number = max_qa
            pad_questions = [self.tokenizer.pad_token_id for _ in range(15)]
            for questions_input in questions_inputs:
                questions_input = questions_input[:max_question_number]
                while len(questions_input) < max_question_number:
                    questions_input.append(pad_questions)
                new_questions_inputs.append(questions_input)

        map_entity_spans = [feature["map_entity_spans"] for feature in features] if "map_entity_spans" in features[
            0].keys() else None
        if map_entity_spans is not None:
            new_map_entity_spans = []
            max_entity_number = max(len(l) for l in map_entity_spans)
            for map_entity_span in map_entity_spans:
                while len(map_entity_span) < max_entity_number:
                    map_entity_span.append([0, 0])
                new_map_entity_spans.append(map_entity_span)

        map_sent_spans = [feature["map_sent_spans"] for feature in features] if "map_sent_spans" in features[
            0].keys() else None
        if map_sent_spans is not None:
            new_map_sent_spans = []
            max_sent_number = max(len(l) for l in map_sent_spans)
            for map_sent_span in map_sent_spans:
                while len(map_sent_span) < max_sent_number:
                    map_sent_span.append([0, 0])
                new_map_sent_spans.append(map_sent_span)

        sent_entity_edges = [feature["sent_entity_edge"] for feature in features] if "sent_entity_edge" in features[
            0].keys() else None
        if sent_entity_edges is not None:
            new_sent_entity_edges = []
            max_sent_number = max(len(l) for l in sent_entity_edges)
            for sent_entity_edge in sent_entity_edges:
                sent_entity_edge = np.array(sent_entity_edge)
                result = np.zeros((max_sent_number, max_entity_number))
                result[:sent_entity_edge.shape[0], :sent_entity_edge.shape[1]] = sent_entity_edge
                new_sent_entity_edges.append(result.tolist())

        answer_indexs = [feature["answer_index"] for feature in features] if "answer_index" in features[
            0].keys() else None
        if answer_indexs is not None:
            new_answer_indexs = []
            max_answer_number = max(len(l) for l in answer_indexs)
            # if max_answer_number>max_qa:
            #     max_answer_number = max_qa

            for case_index, answer_index in enumerate(answer_indexs):
                answer_index = answer_index[:max_answer_number]
                while len(answer_index) < max_answer_number:
                    answer_index.append(-1)
                if True in [each > max_entity_number for each in answer_index]:
                    pdb.set_trace()
                new_answer_indexs.append(answer_index)

        for f in features:
            for k in ['answer_index', 'sent_spans', 'sent_entity_edge', 'map_entity_spans', 'questions_inputs',
                      'questions_inputs_mask', 'answers']:
                if k in f:
                    del f[k]

        to_return = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if questions_inputs is not None:
            to_return['questions_inputs'] = torch.tensor(new_questions_inputs)  # ([8, 10, 15])
            to_return['map_entity_spans'] = torch.tensor(new_map_entity_spans)  # ([8, 200, 2])
            to_return['map_sent_spans'] = torch.tensor(new_map_sent_spans)  # ([8, 58, 2])
            to_return['sent_entity_edge'] = torch.tensor(new_sent_entity_edges)  # ([8, 58, 2])
            to_return['answer_index'] = torch.tensor(new_answer_indexs)  # ([8, 10])
        return to_return


class MySeq2SeqTrainer(Seq2SeqTrainer):
    class LabelSmoother:
        """
        Adds label-smoothing on a pre-computed output from a Transformers model.

        Args:
            epsilon (`float`, *optional*, defaults to 0.1):
                The label smoothing factor.
            ignore_index (`int`, *optional*, defaults to -100):
                The index in the labels to ignore when computing the loss.
        """

        def __call__(self, model_output, target):
            epsilon: float = 0.1
            ignore_index: int = -100

            logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
            lprobs = nn.functional.log_softmax(logits, dim=-1)
            if target.dim() == lprobs.dim() - 1:
                target = target.unsqueeze(-1)
            target = torch.clamp(target, min=0)
            nll_loss = -lprobs.gather(dim=-1, index=target)
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            if ignore_index is not None:
                pad_mask = target.eq(ignore_index)
                nll_loss.masked_fill_(pad_mask, 0.)
                smooth_loss.masked_fill_(pad_mask, 0.)
            else:
                nll_loss = nll_loss.squeeze(-1)
                smooth_loss = smooth_loss.squeeze(-1)
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            eps_i = epsilon / lprobs.size(-1)
            loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
            return loss


    def compute_margin_loss(self, zero_logits, new_logits, labels):
        if labels.dim() == zero_logits.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(-100)
        labels = torch.clamp(labels, min=0)

        zero_logits = nn.functional.softmax(zero_logits, dim=-1)
        zero_logits = zero_logits.gather(dim=-1, index=labels)
        zero_logits.masked_fill_(padding_mask, 0.0)  # [4, 84, 1]
        lm_preds = zero_logits.squeeze(2).contiguous()  # batch_size, len

        new_logits = nn.functional.softmax(new_logits, dim=-1)
        new_logits = new_logits.gather(dim=-1, index=labels)
        new_logits.masked_fill_(padding_mask, 0.0)  # [4, 84, 1]
        new_preds = new_logits.squeeze(2).contiguous()  # batch_size, len
        delta = new_preds - lm_preds

        new_lm = (1 - new_preds).mul(1 - (new_preds - lm_preds) ** 5) / 2  # [4, 84]
        padding_mask = padding_mask.squeeze(-1)
        new_lm.masked_fill_(padding_mask, 0.0)
        new_lm = new_lm.sum()
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        new_lm = new_lm / (num_active_elements)

        return delta, new_lm

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        output_attn = outputs.cross_attentions  # 12:([8, 16, 84, 95])
        q_e_concat = outputs['q_e_concat']  # ([8, 15, 95])
        q_e_concat = torch.mean(q_e_concat, 1)  # 8,95
        output_attn = output_attn[-1]
        output_attn = output_attn[:, -1, :, :]  # [8, 84, 95])
        output_attn = torch.mean(output_attn, 1)  # 8,95
        output_attn = F.relu(output_attn)
        output_attn = F.relu(output_attn)
        pad_mask = inputs['map_entity_spans'][:, :, 0].eq(0)
        cover_loss = torch.nn.functional.kl_div(output_attn, q_e_concat, reduction='none')
        num_active_elements = pad_mask.numel() - pad_mask.long().sum()
        cover_loss = cover_loss.masked_fill_(pad_mask, 0.).sum()
        cover_loss = cover_loss / (num_active_elements)
        loss += cover_loss

        if 'q_e_concat' in outputs.keys() and self.args.qa_loss:
            class_loss_fct = CrossEntropyLoss()
            class_label = inputs['answer_index'].view(-1)  # (8*14)
            class_mask = class_label.eq(-1)
            class_label = torch.clamp(class_label, min=0)
            entity_num = outputs['q_e_concat'].shape[-1]
            class_logits = outputs['q_e_concat'].view(-1, entity_num)
            if (class_label >= class_logits.shape[1]).sum() > 0:
                pdb.set_trace()
            class_loss = class_loss_fct(class_logits, class_label).sum()
            num_active_elements = class_mask.numel() - class_mask.long().sum()
            class_loss = class_loss / (num_active_elements)
            loss += class_loss * 10

        zero_logits = outputs['zero_logits']
        new_logits = outputs['logits']
        if zero_logits !='None':
            delta, new_lm = self.compute_margin_loss(zero_logits, new_logits, labels)
            loss += new_lm
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "repetition_penalty": 2.0,
            "no_repeat_ngram_size": 3,
            "questions_inputs": inputs['questions_inputs'],  # ([8, 10, 15])
            "map_entity_spans": inputs['map_entity_spans'],  # ([8, 10, 15])
            "map_sent_spans": inputs['map_sent_spans'],  # ([8, 10, 15])
            "sent_entity_edge": inputs['sent_entity_edge'],  # ([8, 10, 15])
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            q_e_concat = outputs['q_e_concat']
            # pdb.set_trace()
            # np.save('q_e_concat.npy', q_e_concat)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels, q_e_concat)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        qa_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_qa = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels, q_e_concat = self.prediction_step(model, inputs, prediction_loss_only,
                                                                    ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if q_e_concat is not None:
                q_e_concat = self._pad_across_processes(q_e_concat)
                q_e_concat = self._nested_gather(q_e_concat)
                if qa_host is not None:
                    if qa_host.shape[2] > q_e_concat.shape[2]:

                        pad_zeros = torch.zeros([q_e_concat.shape[0], qa_host.shape[1], qa_host.shape[2]]).to('cuda')
                        pad_zeros[:q_e_concat.shape[0], :q_e_concat.shape[1], :q_e_concat.shape[2]] = q_e_concat
                        q_e_concat = pad_zeros
                    elif qa_host.shape[2] < q_e_concat.shape[2]:

                        pad_zeros = torch.zeros([qa_host.shape[0], q_e_concat.shape[1], q_e_concat.shape[2]]).to('cuda')
                        pad_zeros[:qa_host.shape[0], :q_e_concat.shape[1], :qa_host.shape[2]] = qa_host
                        qa_host = pad_zeros

                qa_host = q_e_concat if qa_host is None else nested_concat(qa_host, q_e_concat,
                                                                           padding_index=0)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if qa_host is not None:
                    class_labels = nested_numpify(qa_host)
                    all_qa = (
                        class_labels if all_qa is None else nested_concat(all_qa, class_labels,
                                                                          padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, all_qa = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if qa_host is not None:
            class_labels = nested_numpify(qa_host)
            all_qa = class_labels if all_qa is None else nested_concat(all_qa, class_labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_qa is not None:
            all_qa = nested_truncate(all_qa, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, q_e_concat=all_qa))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


