# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint
"""

"""
Usage:
python speech_to_label_infer.py  --asr_model="MatchboxNet-VAD-3x2" --dataset='/home/fjia/data/ava/all_ava_split_300_63.json'  --batch_size=1 --out_dir='SAD/mbn_frame/vad_63_notrim' --time_length=0.63

"""


import os
from argparse import ArgumentParser

import numpy as np
import torch
import json
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.common.metrics.classification_accuracy import TopKClassificationAccuracy, compute_topk_accuracy
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield

        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
def main():
    parser = ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="MatchboxNet-VAD-3x2", required=True, help="Pass: 'MatchboxNet-VAD-3x2'")
    parser.add_argument("--dataset", type=str, required=True, help="path of json file of evaluation data")
    parser.add_argument("--out_dir", type=str, default="vad_frame", help="dir of your vad outputs")
    parser.add_argument("--time_length", type=float, default=0.31)
    parser.add_argument("--batch_size", type=int, default=3)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecClassificationModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecClassificationModel.from_pretrained(model_name=args.asr_model)
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
        
    # setup_test_data
    asr_model.setup_test_data(
        test_data_config={
            'vad_stream': True,
            'sample_rate': 16000,
            'manifest_filepath':  args.dataset,
            'labels': ['infer',],  #asr_model._cfg.labels,
            'batch_size': args.batch_size,
            'num_workers' : 20,
            'shuffle' : False ,
            'time_length' : args.time_length,
            'shift_length' : 0.01,
            'trim': False
        }
    )
    
    asr_model = asr_model.to(device)
    asr_model.eval()
  

    data = []
    for line in open(args.dataset, 'r'):
        data.append(json.loads(line)['audio_filepath'].split("/")[-1].split(".wav")[0])
    print(f"Inference on {len(data)} audio files/json lines!")
    

    i = 0
    for test_batch in asr_model.test_dataloader():
        movie = data[i]
        print(movie)
        test_batch = [x.to(device) for x in test_batch]
        
        with autocast():
            log_probs = asr_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
            probs = torch.softmax(log_probs, dim=-1)
            to_save = probs[:, 1]
            
            outpath = os.path.join(args.out_dir, data[i] + ".frame")                
            with open(outpath, "a") as fout:
                for f in range(len(to_save)):
                    fout.write('{0:0.4f}\n'.format(to_save[f]))       
        del test_batch
        i+=1



if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
