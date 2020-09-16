import copy
import json
import os
from test import find_matches

import numpy as np
import scipy.io.wavfile as wave
import torch
from IPython.display import Audio, display
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import nemo.collections.asr as nemo_asr
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType

output_dir = 'output'
# sample rate, Hz
SAMPLE_RATE = 16000

asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')


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

# simple data layer to pass audio signal
class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = signal.astype(np.float32) / 32768.0
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1


# inference method for audio signal (single instance)
def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    log_probs, encoded_len, predictions = model.forward(
        input_signal=audio_signal, input_signal_length=audio_signal_len
    )
    return log_probs


def greedy_merge(pred, labels):
    blank_id = len(labels)
    prev_c = -1
    merged = ''
    for c in pred:
        if c != prev_c and c != blank_id:
            merged += labels[c]
        prev_c = c
    return merged


data_layer = AudioDataLayer(sample_rate=cfg.preprocessor.params.sample_rate)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)
labels = asr_model.cfg.decoder['params']['vocabulary']
print(labels)
# print(asr_model.cfg.preprocessor['params'])
data = open('/mnt/sdb/DATA/LibriSpeech/librivox-dev-other.json', 'r').readlines()

for line in data[999:]:
    utt = json.loads(line)
    text = utt['text']
    filename = utt['audio_filepath']
    print(filename)
    print('Reference transcript:', text)

    ref_text = (
        "as soon as her father was gone tessa flew about and put everything in nice order telling the "
        "children she was going out for the day and they were to mind tommo's mother who would see about "
        "the fire and the dinner for the good woman loved tessa and entered into her "
        "little plans with all her heart"
    )
    punct_text = (
        "As soon as her father was gone, Tessa flew about and put everything in nice order, telling the "
        "children she was going out for the day. And they were to mind Tommo's mother who would see about "
        "the fire and the dinner. For the good woman loved Tessa and entered into her "
        "little plans with all her heart."
    )

    print(f'Reference transcript with punctuation: {punct_text}')

    sample_rate, signal = wave.read(filename)

    preds = infer_signal(asr_model, signal)
    pred = [int(np.argmax(p)) for p in preds[0].detach().cpu()]

    #     print(pred)

    pred_text = greedy_merge(pred, labels)
    print('\n\nPredicted text:')
    display(pred_text)

    matches = find_matches(ref_text=punct_text, pred_text=pred_text)
    print(f'Matches: {matches}')

    wave.write(os.path.join(output_dir, 'signal.wav'), sample_rate, signal)
    # display(Audio(data=signal, rate=sample_rate))

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

    # print(spots)

    audio = signal / 32768.0
    time_stride = asr_model.cfg.preprocessor['params']['window_stride']

    hop_length = int(sample_rate * time_stride)  # 160
    n_fft = 512

    # # linear scale spectrogram
    # s = librosa.stft(y=audio,
    #                  n_fft=n_fft,
    #                  hop_length=hop_length)
    # s_db = librosa.power_to_db(np.abs(s) ** 2, ref=np.max, top_db=100)
    # figs = make_subplots(rows=2, cols=1,
    #                      subplot_titles=('Waveform', 'Spectrogram'))
    # figs.add_trace(go.Scatter(x=np.arange(audio.shape[0]) / sample_rate,
    #                           y=audio, line={'color': 'green'},
    #                           name='Waveform'),
    #                row=1, col=1)
    # figs.add_trace(go.Heatmap(z=s_db,
    #                           colorscale=[
    #                               [0, 'rgb(30,62,62)'],
    #                               [0.5, 'rgb(30,128,128)'],
    #                               [1, 'rgb(30,255,30)'],
    #                           ],
    #                           colorbar=dict(
    #                               yanchor='middle', lenmode='fraction',
    #                               y=0.2, len=0.5,
    #                               ticksuffix=' dB'
    #                           ),
    #                           dx=time_stride, dy=sample_rate / n_fft / 1000,
    #                           name='Spectrogram'),
    #                row=2, col=1)
    # figs.update_layout({'margin': dict(l=0, r=0, t=20, b=0, pad=0),
    #                     'height': 500})
    # figs.update_xaxes(title_text='Time, s', row=1, col=1)
    # figs.update_yaxes(title_text='Frequency, kHz', row=2, col=1)
    # figs.update_xaxes(title_text='Time, s', row=2, col=1)

    # take into account strided conv
    time_stride *= 2

    # calibration offset for timestamps
    offset = -0.15

    """
    # cut words
    pos_prev = 0
    for j, spot in enumerate(spots['space']):
        text_j = pred_text.split()[j]
        # display(text_j)
        pos_end = offset + (spot[0] + spot[1]) / 2 * time_stride
        audio_piece = signal[int(pos_prev*sample_rate):int(pos_end*sample_rate)]
        # display(Audio(audio_piece, rate=sample_rate))
        wave.write(f'output/{j}_{pos_prev}-{pos_end}_{text_j}.wav', sample_rate, audio_piece)
        pos_prev = pos_end
    """

    # displaying the last piece
    # display(pred_text.split()[j + 1])
    # display(Audio(signal[int(pos_prev * sample_rate):],
    #               rate=sample_rate))

    # cut sentences
    pred_words = pred_text.split()
    pos_prev = 0
    first_word_idx = 0
    manifest_path = os.path.join(base, 'manifest.json')
    with open(manifest_path, 'w') as f:
        utt_count = 0
        for utt, times in utt_times.items():
            utt_count += 1
            utt_audio_path = os.path.join(base, f"{utt_count:04}.wav")
            start_time, end_time = times
            utt_audio = wav[floor(start_time * sr) : ceil(end_time * sr)]
            wavfile.write(utt_audio_path, sr, utt_audio)

            # Write to manifest
            info = {'audio_filepath': utt_audio_path, 'duration': end_time - start_time, 'text': utt}
            json.dump(info, f)
            f.write('\n')

    for j, last_word_idx in matches.items():
        text_j = pred_words[first_word_idx : last_word_idx + 1]
        print(' '.join(text_j))
        # cut in the middle of the space
        space_spots = spots['space'][last_word_idx]
        pos_end = offset + (space_spots[0] + space_spots[1]) / 2 * time_stride
        audio_piece = signal[int(pos_prev * sample_rate) : int(pos_end * sample_rate)]
        # display(Audio(audio_piece, rate=sample_rate))
        wave.write(f'output/{j:03}.wav', sample_rate, audio_piece)
        pos_prev = pos_end
        first_word_idx = last_word_idx + 1

    # saving the last piece
    print(' '.join(pred_words[first_word_idx:]))
    audio_piece = signal[int(pos_prev * sample_rate) :]
    wave.write(f'output/{j+1:03}.wav', sample_rate, audio_piece)

    # # display alphabet
    # labels_ext = labels + ['_']
    # colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Pastel2[4:4 + 3]
    # color_legend = {'chars': labels_ext, 'dummy': [1] * len(labels_ext)}
    # fig = px.bar(color_legend, x='chars', y='dummy', color='chars', orientation='v',
    #              color_discrete_sequence=colors)
    # fig.update_layout({'height': 200, 'showlegend': False,
    #                    'yaxis': {'visible': False}
    #                    })
    # fig.show()
    #
    # # add background character shapes
    # shapes = []
    # for idx in range(len(pred)):
    #     shapes.append(
    #         dict(type='rect', xref='x1', yref='y1',
    #              x0=offset + idx * time_stride, y0=-0.5,
    #              x1=offset + (idx + 1) * time_stride, y1=0.5,
    #              fillcolor=colors[pred[idx]],
    #              opacity=1.0,
    #              layer='below',
    #              line_width=0,
    #              ),
    #     )
    # figs.update_layout(
    #     shapes=shapes
    # )
    # figs.update_layout(showlegend=False)
    # figs.show()

    break
