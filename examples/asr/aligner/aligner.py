import argparse
import copy
import json
import os
from test import find_matches, normalize

import numpy as np
import scipy.io.wavfile as wave
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import nemo.collections.asr as nemo_asr
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.utils import logging

parser = argparse.ArgumentParser(description="Token classification with pretrained BERT")
parser.add_argument("--output_dir", default='output', type=str, help='Path to output directory')
parser.add_argument("--audio", default='/mnt/sdb/DATA/sample/4831-25894-0009.wav', type=str, help='Path to audio file')
parser.add_argument(
    "--transcript",
    default='/mnt/sdb/DATA/sample/transcript.txt',
    type=str,
    help='Path to associated transcript with punctuation',
)
parser.add_argument(
    "--debug_mode", action="store_true", help="Enables debug mode with more info on data preprocessing and evaluation",
)

args = parser.parse_args()
logging.info(args)

if args.debug_mode:
    logging.setLevel("DEBUG")

os.makedirs(args.output_dir, exist_ok=True)

# sample rate, Hz
SAMPLE_RATE = 16000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En').to(device)

# Preserve a copy of the full config
cfg = copy.deepcopy(asr_model._cfg)
# print(OmegaConf.to_yaml(cfg))

# Make config overwrite-able
OmegaConf.set_struct(cfg.preprocessor, False)

# some changes for streaming scenario
cfg.preprocessor.params.dither = 0.0
cfg.preprocessor.params.pad_to = 0

# Disable config overwriting
OmegaConf.set_struct(cfg.preprocessor, True)

asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)

# Set model to inference mode
asr_model.eval()




# # inference method for audio signal (single instance)
# def infer_signal(model, signal):
#     data_layer.set_signal(signal)
#
#     batch = next(iter(data_loader))
#     audio_signal, audio_signal_len = batch
#     log_probs, encoded_len, predictions = model.forward(input_signal=audio_signal.to(device), input_signal_length=audio_signal_len.to(device))
#     return log_probs


def greedy_merge(pred, labels):
    blank_id = len(labels)
    prev_c = -1
    merged = ''
    for c in pred:
        if c != prev_c and c != blank_id:
            merged += labels[c]
        prev_c = c
    return merged


# data_layer = AudioDataLayer(sample_rate=cfg.preprocessor.params.sample_rate)
# data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

labels = asr_model.cfg.decoder['params']['vocabulary']
logging.debug(labels)
logging.debug(asr_model.cfg.preprocessor['params'])

with open(args.transcript, 'r') as f:
    original_text = f.read()

# ref_text = (
#     "as soon as her father was gone tessa flew about and put everything in nice order telling the "
#     "children she was going out for the day and they were to mind tommo's mother who would see about "
#     "the fire and the dinner for the good woman loved tessa and entered into her "
#     "little plans with all her heart"
# )
# punct_text = (
#     "As soon as her father was gone, Tessa flew about and put everything in nice order, telling the "
#     "children she was going out for the day. And they were to mind Tommo's mother who would see about "
#     "the fire and the dinner. For the good woman loved Tessa and entered into her "
#     "little plans with all her heart."
# )

# logging.debug(f'Reference transcript with punctuation: {punct_text}')

sample_rate, signal = wave.read(args.audio)

# print('cutting...')
# signal = signal[:sample_rate * 10]

original_duration = len(signal) / sample_rate
logging.info(f'Original audio length: {original_duration}')

preds = asr_model.infer_signal(signal, sample_rate)

pred = [int(np.argmax(p)) for p in preds[0].detach().cpu()]

logging.debug(f'Pred: {pred}')

pred_text = greedy_merge(pred, labels)
logging.debug(f'Predicted text: {pred_text}')

matches = find_matches(ref_text=original_text, pred_text=pred_text)
logging.debug(f'Matches: {matches}')


# get timestamps for space and blank symbols
spots = {}
spots['space'] = []
spots['blank'] = []

state = ''
idx_state = 0

if pred[0] == 0:
    state = 'space'
elif pred[0] == len(labels):
    state = 'blank'

for idx in range(1, len(pred)):
    if state == 'space' and pred[idx] != 0:
        spots['space'].append([idx_state, idx - 1])
        state = ''
    elif state == 'blank' and pred[idx] != len(labels):
        spots['blank'].append([idx_state, idx - 1])
        state = ''
    if state == '':
        if pred[idx] == 0:
            state = 'space'
            idx_state = idx
        elif pred[idx] == len(labels):
            state = 'blank'
            idx_state = idx

if state == 'space':
    spots['space'].append([idx_state, len(pred) - 1])
elif state == 'blank':
    spots['blank'].append([idx_state, len(pred) - 1])

logging.debug(f'Spots: {spots}')

time_stride = asr_model.cfg.preprocessor['params']['window_stride']  # 0.01
# take into account strided conv
time_stride *= 2  # 0.02
# calibration offset for timestamps
offset = -0.15
# cut sentences
pred_words = pred_text.split()
pos_prev = 0
first_word_idx = 0
manifest_path = os.path.join(args.output_dir, 'manifest.json')
total_duration = 0
with open(manifest_path, 'w') as f:
    for j, (last_word_idx, ref_text) in enumerate(matches):
        # for the last segment
        if last_word_idx is None:
            # saving the last piece
            logging.debug(f'{" ".join(pred_words[first_word_idx:])}')
            audio_piece = signal[int(pos_prev * sample_rate) :]
            audio_filepath = os.path.join(args.output_dir, f'{j:03}.wav')
            duration = len(audio_piece) / sample_rate
            total_duration += duration
        else:
            # cut in the middle of the space
            space_spots = spots['space'][last_word_idx]
            pos_end = offset + (space_spots[0] + space_spots[1]) / 2 * time_stride
            audio_piece = signal[int(pos_prev * sample_rate) : int(pos_end * sample_rate)]
            audio_filepath = os.path.join(args.output_dir, f'{j:03}.wav')
            duration = len(audio_piece) / sample_rate
            total_duration += duration
            pos_prev = pos_end
            first_word_idx = last_word_idx + 1

        # save new audio file and write to manifest
        wave.write(audio_filepath, sample_rate, audio_piece)
        info = {'audio_filepath': audio_filepath, 'duration': duration, 'text': ref_text}
        json.dump(info, f)
        f.write('\n')

print(f'Original duration: {original_duration}')
print(f'All pieces duration: {total_duration}')
assert round(original_duration, 2) == round(total_duration, 2)
