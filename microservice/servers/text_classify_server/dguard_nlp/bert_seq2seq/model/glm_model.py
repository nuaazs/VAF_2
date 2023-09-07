# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
from torch import nn
import torch.nn.functional as F
from bert_seq2seq.model.blocks.glm_block import GLMBlock
from bert_seq2seq.model.utils import scaled_init_method, divide, unscaled_init_method
from bert_seq2seq.model.layers.embeddings import VocabParallelEmbedding
from bert_seq2seq.model.layers.embeddings import PositionalEmbedding
from bert_seq2seq.model.prompt import PromptSpell
from bert_seq2seq.model.utils import normal_init_method
from torch.nn import LayerNorm

print_rank_0 = print

if os.getenv('ENV_TYPE') == 'deepspeed':
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
else:
    from torch.utils.checkpoint import checkpoint


class GLMStack(torch.nn.Module):
    """GLM transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """

    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        max_sequence_length,
        max_memory_length,
        embedding_dropout_prob,
        attention_dropout_prob,
        output_dropout_prob,
        checkpoint_activations,
        checkpoint_num_layers=1,
        layernorm_epsilon=1.0e-5,
        init_method_std=0.02,
        use_scaled_init_for_output_weights=True,
        relative_encoding=False,
        block_position_encoding=False,
        performer=False,
        use_decoder_layer=False,
        attention_scale=1.0,
    ):
        super(GLMStack, self).__init__()
        self.hidden_size = hidden_size
        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length
        self.performer = performer
        self.use_decoder_layer = use_decoder_layer
        assert not (performer and relative_encoding)

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(
                0.0, init_method_std, num_layers)
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        if relative_encoding:
            # Relative position embedding
            self.position_embeddings = PositionalEmbedding(hidden_size)
            # Per attention head and per partition values.

            self.hidden_size_per_attention_head = divide(
                hidden_size, num_attention_heads)
            self.num_attention_heads_per_partition = num_attention_heads
            self.r_w_bias = torch.nn.Parameter(
                torch.Tensor(self.num_attention_heads_per_partition,
                             self.hidden_size_per_attention_head))
            self.r_w_bias.model_parallel = True
            self.r_r_bias = torch.nn.Parameter(
                torch.Tensor(self.num_attention_heads_per_partition,
                             self.hidden_size_per_attention_head))
            self.r_r_bias.model_parallel = True
            # Always initialize bias to zero.
            with torch.no_grad():
                self.r_w_bias.zero_()
                self.r_r_bias.zero_()
        else:
            # Position embedding (serial).
            if block_position_encoding:
                self.position_embeddings = torch.nn.Embedding(
                    max_sequence_length + 1, hidden_size)
                self.block_position_embeddings = torch.nn.Embedding(
                    max_sequence_length + 1, hidden_size)
                torch.nn.init.normal_(self.block_position_embeddings.weight,
                                      mean=0.0,
                                      std=init_method_std)
            else:
                self.position_embeddings = torch.nn.Embedding(
                    max_sequence_length, hidden_size)
            # Initialize the position embeddings.
            torch.nn.init.normal_(self.position_embeddings.weight,
                                  mean=0.0,
                                  std=init_method_std)

        def get_layer():
            return GLMBlock(hidden_size,
                            num_attention_heads,
                            attention_dropout_prob,
                            output_dropout_prob,
                            layernorm_epsilon,
                            unscaled_init_method(init_method_std),
                            output_layer_init_method=output_layer_init_method,
                            relative_encoding=relative_encoding,
                            performer=performer,
                            attention_scale=attention_scale)

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer() for _ in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(self,
                hidden_states,
                position_ids,
                attention_mask,
                memory_states=None,
                encoder_states=None,
                return_memory=False,
                detach_memory=True,
                checkpoint_activations=False):
        batch_size, query_length = hidden_states.size()[:2]
        memory_length = memory_states[0].size(1) if memory_states else 0

        key_length = query_length + memory_length
        # attention mask is the beginning postion of B region, \in [0, query_len)
        is_scalar = torch.numel(attention_mask) == 1
        is_sep = is_scalar or torch.numel(attention_mask) == batch_size
        if self.performer:
            assert is_scalar, 'attention_mask should be a scalar to indicate the seperation position.'
            assert memory_length == 0, 'Do not support transformer-xl.'
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            # conventional transformer
            def build_mask_matrix(seq_length, sep, memory_length=0):
                m = hidden_states.new_ones((1, seq_length, seq_length))
                m = torch.tril(m)
                if is_scalar:
                    m[0, :, :sep] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = torch.arange(seq_length,
                                       device=sep.device,
                                       dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                if memory_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = torch.cat((hidden_states.new_ones(
                        (batch_size, seq_length, memory_length)), m),
                                  dim=2)
                m = m.unsqueeze(1)
                return m

            if not self.performer:
                attention_mask = build_mask_matrix(query_length,
                                                   sep,
                                                   memory_length=memory_length)
        else:
            attention_mask = attention_mask[:, :, :,
                                            -query_length - memory_length:]

        if self.relative_encoding:
            position_sequence = torch.arange(key_length - 1,
                                             -1,
                                             -1.0,
                                             device=hidden_states.device,
                                             dtype=hidden_states.dtype)
            position_embeddings = self.position_embeddings(position_sequence)
            # Apply dropout
            position_embeddings = self.embedding_dropout(position_embeddings)
        else:
            if self.block_position_encoding:
                position_ids, block_position_ids = position_ids[:,
                                                                0], position_ids[:,
                                                                                 1]
            position_embeddings = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeddings
            if self.block_position_encoding:
                block_position_embeddings = self.block_position_embeddings(
                    block_position_ids)
                hidden_states = hidden_states + block_position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        def check_detach(_hidden_states):
            if detach_memory:
                return _hidden_states.detach()
            return _hidden_states

        if self.max_memory_length > 0 or return_memory:
            mem_layers = [check_detach(hidden_states)]
        else:
            mem_layers = []

        def custom(start, end):

            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_, inputs = inputs[0], inputs[1:]
                if self.relative_encoding:
                    inputs, mems_ = inputs[:4], inputs[4:]
                else:
                    inputs, mems_ = inputs[:1], inputs[1:]
                for i, layer in enumerate(layers_):
                    mem_i_ = mems_[i] if mems_ else None
                    x_ = layer(x_, *inputs, mem=mem_i_)
                    if self.max_memory_length > 0 or return_memory:
                        mem_layers.append(check_detach(x_))
                return x_

            return custom_forward

        if checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                args = [hidden_states, attention_mask
                        ] if not self.use_decoder_layer else [
                            hidden_states, encoder_states, attention_mask
                        ]
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                if memory_states:
                    args += memory_states[l:l + chunk_length]
                hidden_states = checkpoint(custom(l, l + chunk_length), *args)
                l += chunk_length
        else:
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask
                        ] if not self.use_decoder_layer else [
                            hidden_states, encoder_states, attention_mask
                        ]
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                mem_i = memory_states[i] if memory_states else None
                hidden_states = layer(*args, mem=mem_i)
                if self.max_memory_length > 0 or return_memory:
                    mem_layers.append(check_detach(hidden_states))

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if self.max_memory_length > 0 or return_memory:
            mem_layers = self.update_mems(mem_layers,
                                          memory_states,
                                          return_memory=return_memory)

        return (output, mem_layers)

    def update_mems(self, hiddens, mems, return_memory=False):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length
        if not return_memory:
            new_memory_length = min(self.max_memory_length, new_memory_length)
        new_mems = []
        # with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(
                    torch.cat((mems[i][:, -new_memory_length + query_length:],
                               hiddens[i]),
                              dim=1))
        return new_mems


class GLMModel(nn.Module):
    """GLM Language model.
    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self, num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers,
                 output_predict,
                 parallel_output,
                 relative_encoding,
                 block_position_encoding,
                 spell_length,
                 spell_func,
                 attention_scale):

        super(GLMModel, self).__init__()

        tune_prefix_layers = None

        self.parallel_output = parallel_output
        self.output_predict = output_predict
        self.hidden_size = hidden_size

        init_method = normal_init_method(std=0.02)

        self.word_embeddings = VocabParallelEmbedding(vocab_size,
                                                      hidden_size,
                                                      init_method=init_method)

        # Transformer
        self.transformer = GLMStack(
            num_layers,
            hidden_size,
            num_attention_heads,
            max_sequence_length,
            max_memory_length,
            embedding_dropout_prob,
            attention_dropout_prob,
            output_dropout_prob,
            checkpoint_activations,
            checkpoint_num_layers,
            attention_scale=attention_scale,
            relative_encoding=relative_encoding,
            block_position_encoding=block_position_encoding)

        if spell_length is not None:
            self.prompt_spell = PromptSpell(spell_length, self.hidden_size,
                                            spell_func)
        if tune_prefix_layers != None:
            print("the model is freezed!")
            self.freeze_transformer(tune_prefix_layers=tune_prefix_layers)

    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = "Freeze transformer"
        self.word_embeddings.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if tune_prefix_layers is not None:
            log_str += f" tune {tune_prefix_layers} prefix layers"
            for i in range(tune_prefix_layers):
                self.transformer.layers[i].requires_grad_(True)
        print_rank_0(log_str)

    def forward(self,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                mems=None,
                return_memory=False,
                detach_memory=True,
                prompt_pos=None,
                labels=None):
        '''
        Multi_token
        input_ids: 2 x num_choices x seq_length
        position_ids: 2 x num_choices x 2 x 256
        attention_mask: 2 x 3
        Single_token
        input_ids: batch_size x seq_length
        position_ids: 2 x 1 x 2 x 256
        attention_mask: 2 x 3
        '''
        # Embeddings.
        batch_size = input_ids.size(0)
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = torch.arange(batch_size,
                                       device=input_ids.device).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds
        # Transformer.
        transformer_output = self.transformer(embeddings,
                                              position_ids,
                                              attention_mask,
                                              mems,
                                              return_memory=return_memory,
                                              detach_memory=detach_memory)
        logits, hidden_layers = transformer_output

        logits_parallel = logits

        if self.output_predict:
            # Parallel logits.
            logits_parallel = F.linear(logits_parallel,
                                       self.word_embeddings.weight)

            if labels is not None:

                loss = F.cross_entropy(
                    logits_parallel.reshape(-1, logits_parallel.shape[-1]).contiguous().float(), labels.reshape(-1).long())

                if self.parallel_output:  # Put in different GPUs
                    return {
                        'logits': logits_parallel,
                        'loss': loss,
                        'hidden_states': hidden_layers
                    }
                else:
                    return {
                        "logits":
                            logits_parallel,
                        "loss":
                            loss,
                        "hidden_states":
                            hidden_layers
                    }
            else:
                if self.parallel_output:  # Put in different GPUs
                    return {
                        'logits': logits_parallel,
                        'hidden_states': hidden_layers
                    }
                else:
                    return {
                        "logits":
                            logits_parallel,
                        "hidden_states":
                            hidden_layers
                    }

        else:
            return {'logits': logits, 'hidden_states': hidden_layers}

    def load_weights_glm(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))
        if "module" in checkpoint:
            # ddp
            checkpoint = checkpoint["module"]
        checkpoint_load = {}
        for k, v in checkpoint.items():

            checkpoint_load[k[6:] if k[:5] == "model" else k] = v

        self.load_state_dict(checkpoint_load, strict=False)

        return checkpoint_load

    def load_weights(self, checkpoint_path):
        self.load_weights_glm(checkpoint_path)