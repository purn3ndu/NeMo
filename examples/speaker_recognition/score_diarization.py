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

import argparse
import os
import pickle as pkl

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.cluster import SpectralClustering

from nemo.utils import logging


"""
This script faciliates to get DER % based on cosine-smilarity 
for embeddings extracted using speaker_diarize.py

Args:
embeddings_file: path to embeddings pickle file
window: window used to generate speaker embeddings
shift: shift used to generate speaker embeddings
num_speakers: number of speakers present in each audio file
truth_rttm_dir: path to the directory of ground truth rttm files
write_labels: if required pass this argument to save predicted labels for each of audio file in embeddings pickle file

"""


def combine_stamps(stamps):
    combine = []
    idx, end, speaker = (0, 0, 'unknown')
    prev_start, prev_end, prev_speaker = stamps[idx].split()
    while idx < len(stamps) - 1:
        idx += 1
        start, end, speaker = stamps[idx].split()
        if speaker == prev_speaker and start <= prev_end:
            prev_end = end
        else:
            combine.append("{} {} {}".format(prev_start, prev_end, prev_speaker))
            prev_start = start
            prev_end = end
            prev_speaker = speaker

    combine.append("{} {} {}".format(prev_start, end, speaker))
    return combine


def write_label_file(labels, filename):
    with open(filename, 'w') as f:
        for line in labels:
            start, end, speaker = line.strip().split()
            f.write("{}\t{}\t{}\n".format(start, end, speaker))
    print("wrote labels to audacity type label file at ", filename)


def rttm_to_labels(rttm_filename, write=False):
    outname = rttm_filename.split('/')[-1]
    outname = outname[:-5] + '.txt'
    labels = []
    if write:
        g = open(outname, 'w')
    with open(rttm_filename, 'r') as f:
        for line in f.readlines():
            rttm = line.strip().split()
            start, end, speaker = float(rttm[3]), float(rttm[4]) + float(rttm[3]), rttm[7]
            labels.append('{} {} {}'.format(start, end, speaker))
            if write:
                g.write("{}\t{}\t{}\n".format(start, end, speaker))
    if write:
        logging.info("wrote to {}".format(outname))
        g.close()
    else:
        return labels


def labels_to_pyannote_object(labels, identifier='file1'):
    annotation = Annotation(uri=identifier)
    for label in labels:
        start, end, speaker = label.strip().split()
        start, end = float(start), float(end)
        annotation[Segment(start, end)] = speaker

    return annotation


def get_score(embeddings_file, window, shift, num_speakers, truth_rttm_dir, write_labels=True):
    window = window
    shift = shift
    embeddings = pkl.load(open(embeddings_file, 'rb'))
    embeddings_dir = os.path.dirname(embeddings_file)
    num_speakers = num_speakers
    metric = DiarizationErrorRate(collar=0.0)
    DER = 0

    for uniq_key in embeddings.keys():
        logging.info("Diarizing {}".format(uniq_key))
        identifier = uniq_key.split('@')[-1].split('.')[0]
        emb = embeddings[uniq_key]
        cluster_method = SpectralClustering(n_clusters=num_speakers, random_state=42)
        cluster_method.fit(emb)
        lines = []
        for idx, label in enumerate(cluster_method.labels_):
            start_time = idx * shift
            end_time = start_time + window
            tag = 'speaker_' + str(label)
            line = "{} {} {}".format(start_time, end_time, tag)
            lines.append(line)
        # ReSegmentation -> VAD and Segmented Results
        labels = combine_stamps(lines)
        truth_rttm = os.path.join(truth_rttm_dir, identifier + '.rttm')
        truth_labels = rttm_to_labels(truth_rttm)
        reference = labels_to_pyannote_object(truth_labels, identifier=identifier)
        hypothesis = labels_to_pyannote_object(labels, identifier=identifier)
        DER = metric(reference, hypothesis)
        if write_labels:
            filename = os.path.join(embeddings_dir, identifier + '.txt')
            write_label_file(labels, filename)
            logging.info("Wrote {} to {}".format(uniq_key, filename))

    return abs(DER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", help="path to embeddings pickle file", type=str, required=True)
    parser.add_argument("--window", help="window used to generate speaker embeddings", type=float, required=True)
    parser.add_argument("--shift", help="shift used to generate speaker embeddings", type=float, required=True)
    parser.add_argument(
        "--num_speakers", help="number of speakers present in each audio file", type=int, required=True
    )
    parser.add_argument(
        "--truth_rttm_dir", help="path to the directory of ground truth rttm files", type=str, required=True
    )
    parser.add_argument(
        "--write_labels",
        help=":save predicted labels for each of audio file in embeddings pickle file",
        required=False,
        action='store_true',
    )

    args = parser.parse_args()
    embeddings_file, window, shift, num_speakers, truth_rttm_dir, write_labels = (
        args.embeddings_file,
        args.window,
        args.shift,
        args.num_speakers,
        args.truth_rttm_dir,
        args.write_labels,
    )
    DER = get_score(embeddings_file, window, shift, num_speakers, truth_rttm_dir, write_labels=True)
    logging.info("Cumulative DER of all the files is {:.3f}".format(DER))
