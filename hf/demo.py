from processing_bert_vits2 import BertVits2Processor
from modeling_bert_vits2 import BertVits2Model
from configuration_bert_vits2 import BertVits2Config
from tempfile import TemporaryDirectory
from scipy.io import wavfile
import torch
import sys
import os



model_name = sys.argv[1]
wavfile_path = sys.argv[2] if len(sys.argv) > 2 else None

processor = BertVits2Processor.from_pretrained(model_name)
model = BertVits2Model.from_pretrained(model_name)
config = BertVits2Config.from_pretrained(model_name)

sentence1 = "254 號，您的餐點好囉，請到櫃台取餐，謝謝"
sentence2 = "13 號，張 先生，您的美味餐點已經準備好囉，請到櫃台取餐"
sentence3 = "60 號，陳 小姐，您的美味餐點好囉，請到櫃台取餐，謝謝"
sentence4 = "60 號，陳 小姐，您的 赤炸厚切雞排 好囉，請到櫃台取餐"

with torch.no_grad():
    inputs = processor(sentence4, language="zh", return_tensors="pt")
    # print(inputs['input_ids'].shape)
    # print(inputs['bert_input_ids'].shape)
    # print(inputs['word_to_phoneme'].shape)
    # print(inputs['word_to_phoneme'].sum())
    # print(inputs['word_to_phoneme'])
    # print(inputs['input_ids'][0])
    # print(processor.tokenizer.decode(inputs['input_ids'][0]))
    # print(inputs['bert_input_ids'][0])
    result = model(**inputs, speaker_id=0)
    result = result["waveform"]


sr = model.config.sampling_rate
# sr = config.sampling_rate

if wavfile_path:
    wavfile.write(wavfile_path, sr, result[0].numpy())
else:
    with TemporaryDirectory() as dir:
        file = os.path.join(dir, "output.wav")
        wavfile.write(file, sr, result[0].numpy())
        os.system(f"aplay {file}")
