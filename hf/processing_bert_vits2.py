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
Processor class for Bert VITS2
"""

import os
from typing import Optional, Dict
import re

from transformers.tokenization_utils_base import BatchEncoding
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging
from transformers.utils.hub import get_file_from_repo
from transformers import AutoTokenizer, PreTrainedTokenizer, TOKENIZER_MAPPING

# inject BertVits2Tokenizer
import transformers
from tokenization_bert_vits2 import BertVits2Tokenizer
transformers.BertVits2Tokenizer = BertVits2Tokenizer
TOKENIZER_MAPPING.register("bert_vits2", "BertVits2Tokenizer")

logger = logging.get_logger(__name__)

def chinese_number_to_words(text):
    out = ""
    if text[0] == "-":
        out += "負"
        text = text[1:]
    elif text[0] == "+":
        out += "正"
        text = text[1:]
    if "." in text:
        integer, decimal = text.split(".")
        out += chinese_number_to_words(integer)
        out += "點"
        for c in decimal:
            out += chinese_number_to_words(c)
        return out
    chinese_num = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    length = len(text)
    for i, c in enumerate(text):
        if c == "0" and out[-1] not in chinese_num:
            if i != length - 1 or length == 1:
                out += chinese_num[0]
        else:
            out += chinese_num[int(c)]
        if length - i == 2:
            out += "十"
        elif length - i == 3:
            out += "百"
        elif length - i == 4:
            out += "千"
        elif length - i == 5:
            out += "萬"
        elif length - i == 6:
            out += "十"
        elif length - i == 7:
            out += "百"
        elif length - i == 8:
            out += "千"
        elif length - i == 9:
            out += "億"
        elif length - i == 10:
            out += "十"
        elif length - i == 11:
            out += "百"
        elif length - i == 12:
            out += "千"
        elif length - i == 13:
            out += "兆"
        elif length - i == 14:
            out += "十"
        elif length - i == 15:
            out += "百"
        elif length - i == 16:
            out += "千"
        elif length - i == 17:
            out += "京"
    return out


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
    attributes = ["tokenizer"]

    def __init__(self, tokenizer: PreTrainedTokenizer, bert_tokenizers: Dict[str, PreTrainedTokenizer]):
        super().__init__(tokenizer)
        self.__bert_tokenizers = bert_tokenizers

    @property
    def bert_tokenizers(self):
        return self.__bert_tokenizers

    def preprocess_stage1(self, text, language=None):
        # normalize punctuation
        text = text.replace("，", ",").replace("。", ".").replace("？", "?").replace("！", "!").replace("...", "…")
        # normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # convert number to words
        if language == "zh":
            text = re.sub(r"[+-]?\d+", lambda x: chinese_number_to_words(x.group()), text)
        return text

    def preprocess_stage2(self, text, language=None):
        # normalize whitespace
        text = re.sub(r"\s", 'SP', text).strip()
        return text

    def __call__(
        self,
        text=None,
        language=None,
        return_tensors="pt",
        max_length=256,
        add_special_tokens=True,
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

        if language is None:
            raise ValueError("The language argument is required for BertVits2Processor.")
        
        if language not in self.bert_tokenizers:
            raise ValueError(f"Language '{language}' not supported by BertVits2Processor.")
        
        bert_text = self.preprocess_stage1(text, language)
        g2p_text = self.preprocess_stage2(bert_text, language)

        phone_text, tone_ids, lang_ids, word2ph = self.tokenizer.convert_g2p(g2p_text, language, add_special_tokens)

        encoded_text = self.tokenizer(
            phone_text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        bert_tokenizer = self.bert_tokenizers[language]
        bert_encoded_text = bert_tokenizer(
            bert_text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            add_special_tokens=add_special_tokens,
            return_token_type_ids=False,
            **kwargs,
        )

        return BatchEncoding({
            **encoded_text,
            **{ f"bert_{k}": v for k, v in bert_encoded_text.items() },
            "tone_ids": [tone_ids],
            "language_ids": [lang_ids],
            "word_to_phoneme": [word2ph],
        }, tensor_type=return_tensors)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor_dict, kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        processor_dict['bert_tokenizers'] = {
            key: AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder=val)
            for key, val in processor_dict['bert_tokenizers'].items()
        }
        return cls.from_args_and_dict(args, processor_dict, **kwargs)

    def save_pretrained(
        self,
        save_directory,
        **kwargs,
    ):
        """
        Save the processor to the `save_directory` directory. If the processor has been created from a
        repository, the method will push the model to the `save_directory` repository.

        Args:
            save_directory (`str`):
                Directory where the processor will be saved.
            push_to_hub (`bool`, `optional`, defaults to `False`):
                Whether or not to push the model to the Hugging Face Hub after saving it.
            kwargs:
                Additional attributes to be saved with the processor.
        """
        os.makedirs(save_directory, exist_ok=True)
        for language, tokenizer in self.bert_tokenizers.items():
            tokenizer.save_pretrained(os.path.join(save_directory, f"bert_{language}"))
        bert_tokenizers = self.bert_tokenizers
        self.bert_tokenizers = {language: f"bert_{language}" for language in self.bert_tokenizers}
        outputs = super().save_pretrained(save_directory, **kwargs)
        self.bert_tokenizers = bert_tokenizers
        return outputs
