from modeling_bert_vits2 import BertVits2Model
from configuration_bert_vits2 import BertVits2Config
from processing_bert_vits2 import BertVits2Processor
from tokenization_bert_vits2 import BertVits2Tokenizer
from transformers import PreTrainedTokenizerFast
from tempfile import TemporaryDirectory
from pathlib import Path
import torch
import sys
import os

model_name = sys.argv[1]

save_to = Path(sys.argv[2])

processor = BertVits2Processor.from_pretrained(model_name)
model = BertVits2Model.from_pretrained(model_name)

class OnnxWrapper(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        kwargs.update(self.kwargs)
        return self.model(*args, **kwargs)["waveform"]

model = OnnxWrapper(model, speaker_id=0)

with torch.no_grad():
    inputs = processor("你好我是愛利", language="zh", return_tensors="pt")
    inputs = tuple(
        inputs[k]
        for k in ["input_ids", "tone_ids", "language_ids", "attention_mask", "word_to_phoneme", "bert_input_ids", "bert_attention_mask"]
    )
    os.makedirs(save_to, exist_ok=True)
    with TemporaryDirectory() as dir:
        torch.onnx.export(
            model,
            inputs,
            os.path.join(dir, "model.onnx"),
            opset_version=19,
            input_names=["input_ids", "tone_ids", "language_ids", "attention_mask", "word_to_phoneme", "bert_input_ids", "bert_attention_mask"],
            output_names=["waveform"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "phone_sequence"},
                "tone_ids": {0: "batch", 1: "phone_sequence"},
                "language_ids": {0: "batch", 1: "phone_sequence"},
                "attention_mask": {0: "batch", 1: "phone_sequence"},
                "word_to_phoneme": {0: "batch", 1: "bert_sequence"},
                "bert_input_ids": {0: "batch", 1: "bert_sequence"},
                "bert_attention_mask": {0: "batch", 1: "bert_sequence"},
                "waveform": {0: "batch", 1: "waveform"},
            },
            do_constant_folding=True,
            # use_external_data_format=True,
        )
        os.rename(os.path.join(dir, "model.onnx"), save_to / "model.onnx")
        # # use v2 export
        # export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
        # torch.onnx.dynamo_export(
        #     model,
        #     export_options=export_options,
        #     **inputs,
        # ).save(save_to / "model.onnx")
        print(f"ONNX model saved to {save_to / 'model.onnx'}")
    # save configs
    model.model.config.save_pretrained(save_to)
    # save tokenizer
    processor.save_pretrained(save_to)
