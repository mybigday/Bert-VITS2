# coding=utf-8
# Copyright 2023 The Suno AI Authors and The HuggingFace Inc. team. All rights reserved.
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
"""
Processor class for Bark
"""

import json
import os
from typing import Optional, Dict

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging
from transformers.utils.hub import get_file_from_repo
from transformers import AutoTokenizer, PreTrainedTokenizer
import transformers


logger = logging.get_logger(__name__)

class PreTrainedTokenizerDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class AutoTokenizerDict(PreTrainedTokenizerDict):
    def from_pretrained(models, *args, **kwargs):
        return AutoTokenizerDict({
            k: AutoTokenizer.from_pretrained(v, *args, **kwargs) for k, v in models.items()
        })

transformers.AutoTokenizerDict = AutoTokenizerDict
transformers.PreTrainedTokenizerDict = PreTrainedTokenizerDict

class BertVits2Processor(ProcessorMixin):
    r"""
    Constructs a Bark processor which wraps a text tokenizer and optional Bark voice presets into a single processor.

    Args:
        tokenizers ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`].
        bert_tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`].

    """

    tokenizer_class = "AutoTokenizer"
    bert_tokenizers_class = "AutoTokenizerDict"
    attributes = ["tokenizer", "bert_tokenizers"]

    preset_shape = {
        "semantic_prompt": 1,
        "coarse_prompt": 2,
        "fine_prompt": 2,
    }

    def __init__(self, tokenizer: PreTrainedTokenizer, bert_tokenizers: PreTrainedTokenizerDict):
        super().__init__(tokenizer, bert_tokenizers)

    def __call__(
        self,
        text=None,
        language=None,
        return_tensors="pt",
        max_length=256,
        add_special_tokens=False,
        return_attention_mask=True,
        padding="longest",
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s). This method forwards the `text` and `kwargs`
        arguments to the AutoTokenizer's [`~AutoTokenizer.__call__`] to encode the text. The method also proposes a
        voice preset which is a dictionary of arrays that conditions `Bark`'s output. `kwargs` arguments are forwarded
        to the tokenizer and to `cached_file` method if `voice_preset` is a valid filename.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            Tuple([`BatchEncoding`], [`BatchFeature`]): A tuple composed of a [`BatchEncoding`], i.e the output of the
            `tokenizer` and a [`BatchFeature`], i.e the voice preset with the right tensors type.
        """

        encoded_text = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        bert_tokenizer = self.bert_tokenizers[language]
        bert_encoded_text = bert_tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        return {
            "input_ids": encoded_text["input_ids"],
            "attention_mask": encoded_text["attention_mask"],
            "bert_input_ids": bert_encoded_text["input_ids"],
            "bert_attention_mask": bert_encoded_text["attention_mask"],
        }
