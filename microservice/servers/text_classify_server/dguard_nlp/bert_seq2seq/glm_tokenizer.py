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
"""Utilities for using and training tokenizers (char, wordpiece, sentencepiece)"""
import numpy as np
import torch
import sentencepiece as spm
from collections import namedtuple
import itertools

print_rank_0 = print
COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))
TYPE_TUPLE = namedtuple('TypeToken', ('name', 'token', 'Id'))


class Encoder_SP:

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text):
        """
        text="...."
        """
        return self.sp.EncodeAsIds(text)

    def decode(self, tokens):
        """
        tokens=[x1,x2,...]
        """
        text = [int(token) for token in tokens]
        return self.sp.DecodeIds(text)

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]

    def convert_token_to_id(self, token):
        return self.sp.PieceToId(token)

    def convert_id_to_token(self, idx):
        return self.sp.IdToPiece(idx)


class CommandToken(object):
    def __init__(self, name, token, Id, lstrip=False, rstrip=False):
        self.name = name
        self.token = token
        self.Id = Id
        self.lstrip = lstrip
        self.rstrip = rstrip

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))

class TypeToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(TYPE_TUPLE(self.name, self.token, self.Id))


class GLMTokenizer:

    def __init__(self,
                 vocab_path=None,
                 add_block_symbols=True,
                 add_task_mask=True,
                 add_decoder_mask=False,
                 fix_command_token=True):
        """
        Args:
            add_block_symbols: (str):
                When add_block_symbol is True, a bunch of block-masking-related special tokens will be added to the vocab
            add_task_mask: (bool)
                when add_task_mask is True, the generation mask token and gap sentence mask token will be distinguished
            add_decoder_mask (bool)
                When add_decoder_mask is True, some tokens of the block spans will be masked for BERT, and a special token
                    for that will be added to vocab
            fix_command_token: (bool)
                When add_task_mask, setting fix_command
        """
        self.text_tokenizer = Encoder_SP(vocab_path)
        self.num_command_tokens = 0
        self.num_text_tokens = self.text_tokenizer.sp.vocab_size()
        self.num_tokens = self.num_text_tokens
        self.num_type_tokens = 2
        self.token_eos_id = self.num_text_tokens
        self.token_pad_id = self.num_text_tokens
        self.token_sep_id = self.num_text_tokens + 1
        self.token_cls_id = self.num_text_tokens + 2
        self.token_mask_id = self.num_text_tokens + 3
        self.token_unk_id = self.num_text_tokens + 4

        self._command_tokens = [
            CommandToken('pad', '<|endoftext|>', self.num_text_tokens),
            CommandToken('eos', '<|endoftext|>', self.num_text_tokens),
            CommandToken('sep', '[SEP]', self.num_text_tokens + 1),
            CommandToken('ENC', '[CLS]', self.num_text_tokens + 2),
            CommandToken('MASK',
                         '[MASK]',
                         self.num_text_tokens + 3,
                         lstrip=True),
            CommandToken('unk', '[UNK]', self.num_text_tokens + 4)
        ]

        self.num_tokens += 5
        self.num_command_tokens += 6
        if add_block_symbols:
            self.token_sop_id = self.num_tokens + 1
            self.token_eop_id = self.num_tokens + 2
            self._command_tokens.extend([
                CommandToken('sop', '<|startofpiece|>', self.num_tokens + 1),
                CommandToken('eop', '<|endofpiece|>', self.num_tokens + 2)
            ])
            if fix_command_token:
                self.num_tokens += 3
            else:
                self.num_tokens += 2
            self.num_command_tokens += 2
            if add_task_mask:
                if fix_command_token:
                    self.token_smask_id = self.num_tokens
                    self.token_gmask_id = self.num_tokens + 1
                    self._command_tokens.extend([
                        CommandToken('sMASK',
                                     '[sMASK]',
                                     self.num_tokens,
                                     lstrip=True),
                        CommandToken('gMASK',
                                     '[gMASK]',
                                     self.num_tokens + 1,
                                     lstrip=True)
                    ])
                else:
                    self.token_smask_id = self.num_tokens + 1
                    self.token_gmask_id = self.num_tokens
                    self._command_tokens.extend([
                        CommandToken('gMASK',
                                     '[gMASK]',
                                     self.num_tokens,
                                     lstrip=True),
                        CommandToken('sMASK',
                                     '[sMASK]',
                                     self.num_tokens + 1,
                                     lstrip=True)
                    ])
                self.num_tokens += 2
                self.num_command_tokens += 2

        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {
            tok.token: tok
            for tok in self._command_tokens
        }
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}
        if torch.cuda.is_available():
            print_rank_0({tok.name: tok.Id for tok in self._command_tokens})
        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {
            t: Id
            for Id, t in self.command_id_map.items()
        }

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def _encode(self, text):
        ids = self.text_tokenizer.encode(text)
        return ids

    def trunction(self, input_ids,
                  target_ids=None,
                  max_length=512,
                  src_pop_index=-1,
                  tgt_pop_index=-1):
        if target_ids is None:
            while len(input_ids) > max_length:
                input_ids.pop(src_pop_index)

        else :
            while len(input_ids) + len(target_ids) > max_length:
                if len(input_ids) > len(target_ids):
                    input_ids.pop(src_pop_index)
                else :
                    target_ids.pop(tgt_pop_index)

    def encode_plus(self, source_text,
                       target_text=None,
                       max_length=512,
                       prefix_flag="",
                       post_flag=""):

        cls_id = self.token_cls_id
        sop_id = self.token_sop_id
        eop_id = self.token_eop_id
        mask_id = self.get_command("gMASK").Id

        source_tokens = self.EncodeAsIds(" " + source_text)
        if prefix_flag != "":
            prefix_flag_ids = self.EncodeAsIds(prefix_flag)
            prompt = [cls_id] + prefix_flag_ids
        else :
            prompt = [cls_id]

        if post_flag != "":
            post_flag_ids = self.EncodeAsIds(post_flag)
            post_prompt = post_flag_ids
        else :
            post_prompt = []

        source_tokens = prompt + source_tokens + post_prompt + [mask_id]

        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))

        block_position_ids = [0] * len(source_tokens)
        mask_pos = source_tokens.index(mask_id)
        # sop_id 位置为1

        if target_text != None:
            target_tokens = self.EncodeAsIds(" " + target_text)
            # target_tokens = target_tokens + [eop_id]
            target_tokens = [sop_id] + target_tokens

            self.trunction(input_ids=source_tokens,
                           target_ids=target_tokens,
                           max_length=max_length-1,
                           src_pop_index=-2,
                           tgt_pop_index=-1)

            loss_mask = [1] * len(target_tokens)

            tokens = source_tokens + target_tokens + [eop_id]
            input_ids = tokens[:-1]
            labels = tokens[1:]

            loss_mask = [0] * len(source_tokens) + loss_mask
            position_ids += [mask_pos] * len(target_tokens)

            block_position_ids += list(range(1, len(target_tokens) + 1))
            position_ids = [position_ids, block_position_ids]

            sample = {'input_ids': input_ids,
                      'labels': labels,
                      'attention_mask': sep,
                      'loss_mask': loss_mask,
                      "position_ids": position_ids}
        else:
            # tokens = source_tokens + [sop_id]
            tokens = source_tokens
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]

            sample = {'input_ids': np.array(tokens, dtype=np.int64),
                      'attention_mask': np.array(sep, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64)}
        return sample

    def MultiWordId(self,exception=None):
        #get multi word tokens' ids
        #return ids list
        #exception token: string list
        result=[]
        for i in range(self.num_text_tokens):
            word=self.IdToToken(i)
            if exception:
                if word not in exception and len(word) > 2:
                    result.append(i)
            else:
                if len(word) > 2:
                    result.append(i)
        return result

    def EncodeAsIds(self, text, process_fn=None, maxlen=1024):
        """
        encode text using text tokenizer and shift Id values for command tokens
        """
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        def split_on_token(tok_extended: CommandToken, text):
            result = []
            tok = tok_extended.token
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # CommandToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # Strip white spaces on the right
                if tok_extended.rstrip and i > 0:
                    # A bit counter-intuitive but we strip the left of the string
                    # since tok_extended.rstrip means the special token is eating all white spaces on its right
                    sub_text = sub_text.lstrip()
                # Strip white spaces on the left
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()  # Opposite here

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self.text_tokenizer.encode(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._command_token_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (self._encode(token)
                     if token not in self._command_token_tokens else
                     [self.command_token_map[token].Id]
                     for token in tokenized_text)))

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)

        if len(Ids) > maxlen:
            Ids = Ids[:maxlen]
        return Ids
        # tokenization = Tokenization(Ids, processed_text, text)
        # tokenization.set_command_tokens(self._command_tokens)
        # return tokenization

    def CommandTokenIds(self,exception=None):
        #get command tokens' ids
        #return ids list
        #exception token: string list
        result=[]
        for s in self._command_tokens:
            if not exception or (exception and s.name not in exception):
                result.append(s.Id)
        return(result)

    def EncodeAsTokens(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        return tokens
        # tokenization = Tokenization(tokens, processed_text, text, asIds=False)
        # tokenization.set_command_tokens(self._command_tokens)
        # return tokenization
        # return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id, type_token=False):
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        elif Id in self.type_id_map:
            return self.type_id_map[Id].token
        else:
            return self.text_tokenizer.convert_id_to_token(int(Id))

    def TokenToId(self, token, type_token=False):
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        return self.text_tokenizer.convert_token_to_id(token)

    def DecodeIds(self, Ids, type_token=False):
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.
                            type_id_map[Id].token for Id in Ids)
        Ids = list(map(int, Ids))
        pieces = []
        last = 0
        for i, token_id in enumerate(Ids):
            if token_id in self.command_id_map:
                pieces.append(Ids[last:i])
                pieces.append(token_id)
                last = i + 1
        pieces.append(Ids[last:])
        text = ""
        for piece in pieces:
            if isinstance(piece, int):
                text += self.command_id_map[piece].token
            elif piece:
                text += self.text_tokenizer.decode(piece)
        return text

    def decode(self, ids):
        return self.DecodeIds(ids)


    def DecodeTokens(self, Tokens, type_token=False):
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t
                            for t in Tokens)
        # if isinstance(Tokens, Tokenization):
        #     Tokens = Tokens.tokenization
        return self.text_tokenizer.decode(
            [self.TokenToId(tok) for tok in Tokens])

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def get_type(self, name):
        """get type token corresponding to `name`"""
        return self.type_name_map[name]