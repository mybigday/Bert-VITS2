# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Convert VITS checkpoint."""

import argparse
import json
import tempfile
import re

import torch
from huggingface_hub import hf_hub_download

# from transformers import VitsConfig, VitsModel, VitsTokenizer, logging
from modeling_bert_vits2 import BertVits2Model
from tokenization_bert_vits2 import BertVits2Tokenizer
from configuration_bert_vits2 import BertVits2Config
from processing_bert_vits2 import BertVits2Processor, AutoTokenizerDict, PreTrainedTokenizerDict
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import logging


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bert_vits2")

MAPPING_TEXT_ENCODER = {
    "enc_p.emb": "text_encoder.embed_tokens",
    "enc_p.bert_proj": "text_encoder.bert_projs.0",
    "enc_p.ja_bert_proj": "text_encoder.bert_projs.1", # len(bert) >= 2
    "enc_p.en_bert_proj": "text_encoder.bert_projs.2", # len(bert) >= 3
    "enc_p.encoder.attn_layers.*.conv_k": "text_encoder.encoder.layers.*.attention.k_proj",
    "enc_p.encoder.attn_layers.*.conv_v": "text_encoder.encoder.layers.*.attention.v_proj",
    "enc_p.encoder.attn_layers.*.conv_q": "text_encoder.encoder.layers.*.attention.q_proj",
    "enc_p.encoder.attn_layers.*.conv_o": "text_encoder.encoder.layers.*.attention.out_proj",
    "enc_p.encoder.attn_layers.*.emb_rel_k": "text_encoder.encoder.layers.*.attention.emb_rel_k",
    "enc_p.encoder.attn_layers.*.emb_rel_v": "text_encoder.encoder.layers.*.attention.emb_rel_v",
    "enc_p.encoder.norm_layers_1.*.gamma": "text_encoder.encoder.layers.*.layer_norm.weight",
    "enc_p.encoder.norm_layers_1.*.beta": "text_encoder.encoder.layers.*.layer_norm.bias",
    "enc_p.encoder.ffn_layers.*.conv_1": "text_encoder.encoder.layers.*.feed_forward.conv_1",
    "enc_p.encoder.ffn_layers.*.conv_2": "text_encoder.encoder.layers.*.feed_forward.conv_2",
    "enc_p.encoder.norm_layers_2.*.gamma": "text_encoder.encoder.layers.*.final_layer_norm.weight",
    "enc_p.encoder.norm_layers_2.*.beta": "text_encoder.encoder.layers.*.final_layer_norm.bias",
    "enc_p.proj": "text_encoder.project",
}
MAPPING_STOCHASTIC_DURATION_PREDICTOR = {
    "dp.pre": "duration_predictor.conv_pre",
    "dp.proj": "duration_predictor.conv_proj",
    "dp.convs.convs_sep.*": "duration_predictor.conv_dds.convs_dilated.*",
    "dp.convs.convs_1x1.*": "duration_predictor.conv_dds.convs_pointwise.*",
    "dp.convs.norms_1.*.gamma": "duration_predictor.conv_dds.norms_1.*.weight",
    "dp.convs.norms_1.*.beta": "duration_predictor.conv_dds.norms_1.*.bias",
    "dp.convs.norms_2.*.gamma": "duration_predictor.conv_dds.norms_2.*.weight",
    "dp.convs.norms_2.*.beta": "duration_predictor.conv_dds.norms_2.*.bias",
    "dp.flows.0.logs": "duration_predictor.flows.0.log_scale",
    "dp.flows.0.m": "duration_predictor.flows.0.translate",
    "dp.flows.*.pre": "duration_predictor.flows.*.conv_pre",
    "dp.flows.*.proj": "duration_predictor.flows.*.conv_proj",
    "dp.flows.*.convs.convs_1x1.0": "duration_predictor.flows.*.conv_dds.convs_pointwise.0",
    "dp.flows.*.convs.convs_1x1.1": "duration_predictor.flows.*.conv_dds.convs_pointwise.1",
    "dp.flows.*.convs.convs_1x1.2": "duration_predictor.flows.*.conv_dds.convs_pointwise.2",
    "dp.flows.*.convs.convs_sep.0": "duration_predictor.flows.*.conv_dds.convs_dilated.0",
    "dp.flows.*.convs.convs_sep.1": "duration_predictor.flows.*.conv_dds.convs_dilated.1",
    "dp.flows.*.convs.convs_sep.2": "duration_predictor.flows.*.conv_dds.convs_dilated.2",
    "dp.flows.*.convs.norms_1.0.gamma": "duration_predictor.flows.*.conv_dds.norms_1.0.weight",
    "dp.flows.*.convs.norms_1.0.beta": "duration_predictor.flows.*.conv_dds.norms_1.0.bias",
    "dp.flows.*.convs.norms_1.1.gamma": "duration_predictor.flows.*.conv_dds.norms_1.1.weight",
    "dp.flows.*.convs.norms_1.1.beta": "duration_predictor.flows.*.conv_dds.norms_1.1.bias",
    "dp.flows.*.convs.norms_1.2.gamma": "duration_predictor.flows.*.conv_dds.norms_1.2.weight",
    "dp.flows.*.convs.norms_1.2.beta": "duration_predictor.flows.*.conv_dds.norms_1.2.bias",
    "dp.flows.*.convs.norms_2.0.gamma": "duration_predictor.flows.*.conv_dds.norms_2.0.weight",
    "dp.flows.*.convs.norms_2.0.beta": "duration_predictor.flows.*.conv_dds.norms_2.0.bias",
    "dp.flows.*.convs.norms_2.1.gamma": "duration_predictor.flows.*.conv_dds.norms_2.1.weight",
    "dp.flows.*.convs.norms_2.1.beta": "duration_predictor.flows.*.conv_dds.norms_2.1.bias",
    "dp.flows.*.convs.norms_2.2.gamma": "duration_predictor.flows.*.conv_dds.norms_2.2.weight",
    "dp.flows.*.convs.norms_2.2.beta": "duration_predictor.flows.*.conv_dds.norms_2.2.bias",
    "dp.post_pre": "duration_predictor.post_conv_pre",
    "dp.post_proj": "duration_predictor.post_conv_proj",
    "dp.post_convs.convs_sep.*": "duration_predictor.post_conv_dds.convs_dilated.*",
    "dp.post_convs.convs_1x1.*": "duration_predictor.post_conv_dds.convs_pointwise.*",
    "dp.post_convs.norms_1.*.gamma": "duration_predictor.post_conv_dds.norms_1.*.weight",
    "dp.post_convs.norms_1.*.beta": "duration_predictor.post_conv_dds.norms_1.*.bias",
    "dp.post_convs.norms_2.*.gamma": "duration_predictor.post_conv_dds.norms_2.*.weight",
    "dp.post_convs.norms_2.*.beta": "duration_predictor.post_conv_dds.norms_2.*.bias",
    "dp.post_flows.0.logs": "duration_predictor.post_flows.0.log_scale",
    "dp.post_flows.0.m": "duration_predictor.post_flows.0.translate",
    "dp.post_flows.*.pre": "duration_predictor.post_flows.*.conv_pre",
    "dp.post_flows.*.proj": "duration_predictor.post_flows.*.conv_proj",
    "dp.post_flows.*.convs.convs_1x1.0": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.0",
    "dp.post_flows.*.convs.convs_1x1.1": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.1",
    "dp.post_flows.*.convs.convs_1x1.2": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.2",
    "dp.post_flows.*.convs.convs_sep.0": "duration_predictor.post_flows.*.conv_dds.convs_dilated.0",
    "dp.post_flows.*.convs.convs_sep.1": "duration_predictor.post_flows.*.conv_dds.convs_dilated.1",
    "dp.post_flows.*.convs.convs_sep.2": "duration_predictor.post_flows.*.conv_dds.convs_dilated.2",
    "dp.post_flows.*.convs.norms_1.0.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.0.weight",
    "dp.post_flows.*.convs.norms_1.0.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.0.bias",
    "dp.post_flows.*.convs.norms_1.1.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.1.weight",
    "dp.post_flows.*.convs.norms_1.1.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.1.bias",
    "dp.post_flows.*.convs.norms_1.2.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.2.weight",
    "dp.post_flows.*.convs.norms_1.2.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.2.bias",
    "dp.post_flows.*.convs.norms_2.0.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.0.weight",
    "dp.post_flows.*.convs.norms_2.0.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.0.bias",
    "dp.post_flows.*.convs.norms_2.1.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.1.weight",
    "dp.post_flows.*.convs.norms_2.1.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.1.bias",
    "dp.post_flows.*.convs.norms_2.2.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.2.weight",
    "dp.post_flows.*.convs.norms_2.2.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.2.bias",
    "dp.cond": "duration_predictor.cond",  # num_speakers > 1
}
MAPPING_FLOW = {
    "flow.flows.*.pre": "flow.flows.*.conv_pre",
    "flow.flows.*.enc.in_layers.0": "flow.flows.*.wavenet.in_layers.0",
    "flow.flows.*.enc.in_layers.1": "flow.flows.*.wavenet.in_layers.1",
    "flow.flows.*.enc.in_layers.2": "flow.flows.*.wavenet.in_layers.2",
    "flow.flows.*.enc.in_layers.3": "flow.flows.*.wavenet.in_layers.3",
    "flow.flows.*.enc.res_skip_layers.0": "flow.flows.*.wavenet.res_skip_layers.0",
    "flow.flows.*.enc.res_skip_layers.1": "flow.flows.*.wavenet.res_skip_layers.1",
    "flow.flows.*.enc.res_skip_layers.2": "flow.flows.*.wavenet.res_skip_layers.2",
    "flow.flows.*.enc.res_skip_layers.3": "flow.flows.*.wavenet.res_skip_layers.3",
    "flow.flows.*.enc.cond_layer": "flow.flows.*.wavenet.cond_layer",  # num_speakers > 1
    "flow.flows.*.post": "flow.flows.*.conv_post",
}
MAPPING_GENERATOR = {
    "dec.conv_pre": "decoder.conv_pre",
    "dec.ups.0": "decoder.upsampler.0",
    "dec.ups.1": "decoder.upsampler.1",
    "dec.ups.2": "decoder.upsampler.2",
    "dec.ups.3": "decoder.upsampler.3",
    "dec.resblocks.*.convs1.0": "decoder.resblocks.*.convs1.0",
    "dec.resblocks.*.convs1.1": "decoder.resblocks.*.convs1.1",
    "dec.resblocks.*.convs1.2": "decoder.resblocks.*.convs1.2",
    "dec.resblocks.*.convs2.0": "decoder.resblocks.*.convs2.0",
    "dec.resblocks.*.convs2.1": "decoder.resblocks.*.convs2.1",
    "dec.resblocks.*.convs2.2": "decoder.resblocks.*.convs2.2",
    "dec.conv_post": "decoder.conv_post",
    "dec.cond": "decoder.cond",  # num_speakers > 1
}
MAPPING_POSTERIOR_ENCODER = {
    "enc_q.pre": "posterior_encoder.conv_pre",
    "enc_q.enc.in_layers.*": "posterior_encoder.wavenet.in_layers.*",
    "enc_q.enc.res_skip_layers.*": "posterior_encoder.wavenet.res_skip_layers.*",
    "enc_q.enc.cond_layer": "posterior_encoder.wavenet.cond_layer",  # num_speakers > 1
    "enc_q.proj": "posterior_encoder.conv_proj",
}
MAPPING = {
    **MAPPING_TEXT_ENCODER,
    **MAPPING_STOCHASTIC_DURATION_PREDICTOR,
    **MAPPING_FLOW,
    **MAPPING_GENERATOR,
    **MAPPING_POSTERIOR_ENCODER,
    "emb_g": "embed_speaker",  # num_speakers > 1
}
TOP_LEVEL_KEYS = []
IGNORE_KEYS = []


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # strip off the kernel dimension at the end (original weights are Conv1d)
    if key.endswith(".k_proj") or key.endswith(".v_proj") or key.endswith(".q_proj") or key.endswith(".out_proj"):
        value = value.squeeze(-1)

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")


def should_ignore(name, ignore_keys):
    for key in ignore_keys:
        if key.endswith(".*"):
            if name.startswith(key[:-1]):
                return True
        elif ".*." in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        elif key in name:
            return True
    return False


def recursively_load_weights(fairseq_dict, hf_model):
    unused_weights = []

    for name, value in fairseq_dict.items():
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        is_used = False
        for key, mapped_key in MAPPING.items():
            # filter key liks glob pattern
            glob_key = key.replace(".", "\.").replace("*", ".*")
            if re.match(glob_key, name) is None:
                continue
            if key.endswith(".*"):
                key = key[:-1]
            elif "*" in key:
                prefix, suffix = key.split(".*.")
                if prefix in name and suffix in name:
                    key = suffix

            is_used = True
            if mapped_key.endswith(".*"):
                layer_index = name.split(key)[-1].split(".")[0]
                mapped_key = mapped_key.replace("*", layer_index)
            elif "*" in mapped_key:
                layer_index = name.split(key)[0].split(".")[-2]

                # remap the layer index since we removed the Flip layers
                if "flow.flows" in mapped_key:
                    layer_index = str(int(layer_index) // 2)
                if "duration_predictor.flows" in mapped_key or "duration_predictor.post_flows" in mapped_key:
                    layer_index = str(int(layer_index) // 2 + 1)

                mapped_key = mapped_key.replace("*", layer_index)
            if "weight_g" in name:
                weight_type = "weight_g"
            elif "weight_v" in name:
                weight_type = "weight_v"
            elif "bias" in name:
                weight_type = "bias"
            elif "weight" in name:
                weight_type = "weight"
            elif "running_mean" in name:
                weight_type = "running_mean"
            elif "running_var" in name:
                weight_type = "running_var"
            elif "num_batches_tracked" in name:
                weight_type = "num_batches_tracked"
            else:
                weight_type = None
            set_recursively(hf_model, mapped_key, value, name, weight_type)
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


def original_vits_vocab():
    punctuation = ["!", "?", "…", ",", ".", "'", "-"]
    pu_symbols = punctuation + ["SP", "UNK"]
    pad = "_"
    # chinese
    zh_symbols = [
        "E",
        "En",
        "a",
        "ai",
        "an",
        "ang",
        "ao",
        "b",
        "c",
        "ch",
        "d",
        "e",
        "ei",
        "en",
        "eng",
        "er",
        "f",
        "g",
        "h",
        "i",
        "i0",
        "ia",
        "ian",
        "iang",
        "iao",
        "ie",
        "in",
        "ing",
        "iong",
        "ir",
        "iu",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "ong",
        "ou",
        "p",
        "q",
        "r",
        "s",
        "sh",
        "t",
        "u",
        "ua",
        "uai",
        "uan",
        "uang",
        "ui",
        "un",
        "uo",
        "v",
        "van",
        "ve",
        "vn",
        "w",
        "x",
        "y",
        "z",
        "zh",
        "AA",
        "EE",
        "OO",
    ]
    num_zh_tones = 6
    # japanese
    ja_symbols = [
        "N",
        "a",
        "a:",
        "b",
        "by",
        "ch",
        "d",
        "dy",
        "e",
        "e:",
        "f",
        "g",
        "gy",
        "h",
        "hy",
        "i",
        "i:",
        "j",
        "k",
        "ky",
        "m",
        "my",
        "n",
        "ny",
        "o",
        "o:",
        "p",
        "py",
        "q",
        "r",
        "ry",
        "s",
        "sh",
        "t",
        "ts",
        "ty",
        "u",
        "u:",
        "w",
        "y",
        "z",
        "zy",
    ]
    num_ja_tones = 2
    # English
    en_symbols = [
        "aa",
        "ae",
        "ah",
        "ao",
        "aw",
        "ay",
        "b",
        "ch",
        "d",
        "dh",
        "eh",
        "er",
        "ey",
        "f",
        "g",
        "hh",
        "ih",
        "iy",
        "jh",
        "k",
        "l",
        "m",
        "n",
        "ng",
        "ow",
        "oy",
        "p",
        "r",
        "s",
        "sh",
        "t",
        "th",
        "uh",
        "uw",
        "V",
        "w",
        "y",
        "z",
        "zh",
    ]
    num_en_tones = 4
    # combine all symbols
    normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
    symbols = [pad] + normal_symbols + pu_symbols
    # combine all tones
    num_tones = num_zh_tones + num_ja_tones + num_en_tones
    return symbols, num_tones

@torch.no_grad()
def convert_checkpoint(
    pytorch_dump_folder_path,
    checkpoint_path=None,
    params_path=None,
    vocab_path=None,
    language=None,
    bert=None,
    sampling_rate=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    config = BertVits2Config()

    logger.info(f"***Converting model: {checkpoint_path}***")

    config.num_languages = len(language)

    with open(params_path, "r") as f:
        params = json.load(f)
        config.resblock_kernel_sizes = params['model']['resblock_kernel_sizes']
        config.resblock_dilation_sizes = params['model']['resblock_dilation_sizes']
        config.upsample_rates = params['model']['upsample_rates']
        config.upsample_kernel_sizes = params['model']['upsample_kernel_sizes']
        config.upsample_initial_channel = params['model']['upsample_initial_channel']
        config.hidden_size = params['model']['hidden_channels']
        config.duration_predictor_filter_channels = params['model']['filter_channels']
        config.duration_predictor_kernel_size = params['model']['kernel_size']
        config.duration_predictor_dropout = params['model']['p_dropout']
        config.num_hidden_layers = params['model']['n_layers']
        config.num_attention_heads = params['model']['n_heads']
        config.speaker_embedding_size = params['model'].get('gin_channels', 0)
        config.num_speakers = params['data']['n_speakers']
        config.cond_layer_index = params['model'].get('cond_layer_idx', 2) if config.speaker_embedding_size > 0 else config.num_hidden_layers
        config.spectrogram_bins = params['data']['filter_length'] // 2 + 1
        if sampling_rate:
            config.sampling_rate = sampling_rate
        else:
            config.sampling_rate = params['data']['sampling_rate']

    # original VITS checkpoint
    if vocab_path is None:
        symbols, n_tones = original_vits_vocab()
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        _pad = '_'
        _unk = 'UNK'
    else:
        # Save vocab as temporary json file
        symbols = [line.replace("\n", "") for line in open(vocab_path, encoding="utf-8").readlines()]
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        _pad = symbols[0]
        _unk = symbols[-1]
        n_tones = 12
    
    config.num_tones = n_tones
    config.vocab_size = len(symbols)
    config.bert_configs = [BertConfig.from_pretrained(bert_path) for bert_path in bert]

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, "w", encoding="utf-8") as f:
            f.write(json.dumps(symbol_to_id, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        tokenizer = BertVits2Tokenizer(tf.name, languages=language, pad_token=_pad, unk_token=_unk)
    
    processor = BertVits2Processor(
        tokenizer,
        bert_tokenizers=AutoTokenizerDict.from_pretrained(dict(zip(language, bert))),
    )

    config.vocab_size = len(symbols)
    model = BertVits2Model(config)

    # load bert weights
    for i, bert_path in enumerate(bert):
        bert_model = BertModel.from_pretrained(bert_path)
        model.bert_encoders[i].load_state_dict(bert_model.state_dict(), strict=False)

    # drop unsed mappings, if bert is not provided
    if len(bert) == 1:
        MAPPING.pop("enc_p.ja_bert_proj")
        MAPPING.pop("enc_p.en_bert_proj")
    elif len(bert) == 2:
        MAPPING.pop("enc_p.en_bert_proj")


    model.decoder.apply_weight_norm()

    orig_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    recursively_load_weights(orig_checkpoint["model"], model)

    model.decoder.remove_weight_norm()

    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Local path to original checkpoint")
    parser.add_argument("--vocab_path", default=None, type=str, help="Path to vocab.txt")
    parser.add_argument("--params_path", default=None, type=str, required=True, help="Path to original params.json")
    parser.add_argument("--language", default=None, type=str, action="append", help="Tokenizer language (three-letter code)")
    parser.add_argument("--bert", default=None, type=str, action="append", help="Repo or path to bert model")
    parser.add_argument(
        "--sampling_rate", default=None, type=int, help="Sampling rate on which the model was trained."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.pytorch_dump_folder_path,
        args.checkpoint_path,
        args.params_path,
        args.vocab_path,
        args.language,
        args.bert,
        args.sampling_rate,
        args.push_to_hub,
    )
