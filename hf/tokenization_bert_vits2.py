# coding=utf-8
# Copyright 2023 The Kakao Enterprise Authors, the MMS-TTS Authors and the HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for VITS."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import is_phonemizer_available, logging


if is_phonemizer_available():
    import phonemizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}

def is_symbol(ch):
    return ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

class BertVits2Tokenizer(PreTrainedTokenizer):
    """
    Construct a VITS tokenizer. Also supports MMS-TTS.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        language (`str`, *optional*):
            Language identifier.
        add_blank (`bool`, *optional*, defaults to `True`):
            Whether to insert token id 0 in between the other tokens.
        normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the input text by removing all casing and punctuation.
        phonemize (`bool`, *optional*, defaults to `True`):
            Whether to convert the input text into phonemes.
        is_uroman (`bool`, *optional*, defaults to `False`):
            Whether the `uroman` Romanizer needs to be applied to the input text prior to tokenizing.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = [
        "input_ids",
        # "input_tones",
        "attention_mask",
    ]

    def __init__(
        self,
        vocab_file,
        pad_token="<pad>",
        unk_token="<unk>",
        space_token=None,
        languages=None,
        add_blank=True,
        **kwargs,
    ) -> None:
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.languages = languages
        self.add_blank = add_blank

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            space_token=space_token,
            languages=languages,
            add_blank=add_blank,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def zh_g2p(self, text: str) -> Tuple[str, List[int], List[int]]:
        """Converts a string of Chinese text into a list of phonemes and tones."""
        from pypinyin import lazy_pinyin, Style

        with open(os.path.join(os.path.dirname(__file__), "data", "zh_g2p.json"), encoding="utf-8") as f:
            g2p = json.load(f)

        phones = []
        tones = []
        word2ph = []

        initials = lazy_pinyin(text, neutral_tone_with_five=True, style=Style.INITIALS, tone_sandhi=True)
        finals = lazy_pinyin(text, neutral_tone_with_five=True, style=Style.FINALS_TONE3, tone_sandhi=True)

        for initial, final in zip(initials, finals):
            tone = 0
            if final[-1].isdigit():
                pinyin = initial + final[:-1]
                tone = int(final[-1])
                if initial:
                    pinyin = re.sub(r"uei$", "ui", pinyin)
                    pinyin = re.sub(r"iou$", "iu", pinyin)
                    pinyin = re.sub(r"uen$", "un", pinyin)
                else:
                    pinyin = re.sub(r"^ing$", "ying", pinyin)
                    pinyin = re.sub(r"^i$", "yi", pinyin)
                    pinyin = re.sub(r"^in$", "yin", pinyin)
                    pinyin = re.sub(r"^u$", "wu", pinyin)
                    pinyin = re.sub(r"^v", "yu", pinyin)
                    pinyin = re.sub(r"^e", "e", pinyin)
                    pinyin = re.sub(r"^i", "y", pinyin)
                    pinyin = re.sub(r"^u", "w", pinyin)
            else:
                pinyin = initial + final
            if initial == final:
                tone = 0
                phone = [initial]
            else:
                phone = g2p.get(pinyin, [self.unk_token])
                if phone[0] == self.unk_token:
                    tone = 0
                    phone = [self.unk_token]
            tones += [tone] * len(phone)
            phones += phone
            if initial != 'SP':
                word2ph.append(len(phone))
            else:
                word2ph[-1] += 1

        phones = "<|SEP|>".join(phones)
        return phones, tones, word2ph
            

    def convert_g2p(self, text: str, language: str, add_special_tokens: bool) -> Tuple[str, List[int], List[int]]:
        """Converts a string of text into a list of phonemes and tones."""
        if not is_phonemizer_available():
            raise ImportError("Phonemizer is not available. Please install it using `pip install phonemizer`.")

        if language.startswith("zh"):
            phones, tones, word2ph = self.zh_g2p(text)
        else:
            raise ValueError(f"Language '{language}' not supported by VITS.")

        lang_ids = [self.languages.index(language)] * len(tones)

        if self.add_blank:
            tones = self._add_blank(tones, 0)
            lang_ids = self._add_blank(lang_ids, 0)

            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1

        if add_special_tokens:
            word2ph = [0] + word2ph + [0]

        return phones, tones, lang_ids, word2ph

    def _add_blank(self, sequence: List[Union[str, int]], blank: Union[str, int]) -> List[Union[str, int]]:
        interspersed = [blank] * (len(sequence) * 2 + 1)
        interspersed[1::2] = sequence
        return interspersed

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string by inserting the `<pad>` token at the boundary between adjacent characters."""
        tokens = []

        if '<|SEP|>' in text:
            tokens = text.split('<|SEP|>')
        else: # fallback
            i = 0
            while i < len(text):
                found = False
                for j in range(min(len(text), i + 2), i, -1):
                    subtext = text[i:j]
                    if subtext in self.encoder:
                        tokens.append(subtext)
                        i = j
                        found = True
                        break
                if not found:
                    tokens.append(self.unk_token)
                    i += 1

        if self.add_blank:
            tokens = self._add_blank(tokens, self.pad_token)

        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        if self.add_blank and len(tokens) > 1:
            tokens = tokens[1::2]
        return "".join(tokens)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Union[Tuple[str], None]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)
