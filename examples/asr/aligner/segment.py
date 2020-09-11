import json
import os
import re
import string
import sys
from math import ceil, floor

from scipy.io import wavfile

# Segmenter for aligned speech/text (using the gentle aligner).
# Outputs audio files and a manifest in NeMo-accepted format.

MIN_LENGTH = 2.0
alignment_dir = '/home/jocelynh/Desktop/jensen_keynotes/align'


def check_alignment(word, alignment):
    """
    Checks that the alignment line corresponds to the word in the transcript.
    Shows an error message and exits if not.
    :param word: Word from the transcript
    :param alignment: Line from alignment file
    """
    word = re.sub("'", '', word)
    align_word = alignment[: alignment.find(',')]
    align_word = re.sub("'", '', align_word)

    if word.lower() != align_word.lower():
        print(f"ERROR: Alignment and transcript mismatch: " f"'{word}' vs '{align_word}'")
        exit()


def get_utterance_alignments(utterances, alignfile):
    """
    Gets the timings for each utterance, based on the alignment file.
    :param utterances: List of utterances from the transcript
    :param alignfile: Alignment file path
    :return: Mapping from utterances to timings, and number of utts filtered
    """
    utt_times = {}
    remove_punc = string.punctuation
    remove_pattern = r"[{}]".format(remove_punc)

    missing_count = 0
    with open(alignfile, 'r') as f:
        prev_utt_time = None

        for utt in utterances:
            start_time = None
            end_time = None
            utt_tmp = re.sub('-', ' ', utt)
            utt_tmp = re.sub(remove_pattern, '', utt_tmp)

            # Check alignments for each word in the utterance
            for word in utt_tmp.split():
                alignment = f.readline()
                check_alignment(word, alignment)

                # Get utterance start time
                if start_time is None:
                    time_idx = re.search(r"\d", alignment)
                    if time_idx:
                        start_time = float(alignment[time_idx.start() : alignment.rfind(",")])

            # Get utterance end time
            end_time = float(alignment[alignment.rfind(',') + 1 :])

            prev_utt_time = end_time
            if not (start_time and end_time) or (end_time - start_time) < MIN_LENGTH:
                missing_count += 1
            else:
                utt_times[utt] = (start_time, end_time)

    return utt_times, missing_count


def segment_audio(utt_times, audiofile, base):
    """
    Loads, segments, and saves the audio based on utterances.
    Also creates a manifest file.
    :param utt_times: Mapping from utterances to start/stop times
    :param audiofile: Audio (.wav) file path
    :param base: Base path where the manifest should be written
    """
    sr, wav = wavfile.read(audiofile)

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


def main():
    try:
        base = sys.argv[1]
    except IndexError:
        print("You must include a base filename for segmenting," " e.g. 'python segment.py <basename>'.")

    # Construct filenames that we use for segmenting
    textfile = base + '.txt'
    audiofile = base + '.wav'
    alignfile = os.path.join(alignment_dir, f"align_{base}.csv")

    nonexistent = [fname for fname in [textfile, audiofile, alignfile] if not os.path.exists(fname)]
    if nonexistent:
        print(f"Couldn't find files: {nonexistent}")
    else:
        print("Found all necessary files.")

    # Where the audio segments & manifests will be stored
    if not os.path.exists(base):
        os.makedirs(base)

    # Read and split transcript by utterance (roughly, sentences)
    sentence_split_punc = '...', '.', '?', '!', '…', '\n', '- '
    split_pattern = '|'.join(map(re.escape, sentence_split_punc))
    with open(textfile, 'r') as f:
        utterances = f.read()
    utterances = re.split(split_pattern, utterances)
    utterances = [s.strip() for s in utterances if s]

    # Match utterances to timestamps
    utt_times, missing_count = get_utterance_alignments(utterances, alignfile)

    # Yield => percentage of utterances where we have start & end times
    num_aligned = len(utt_times)
    print(f"Yield: {num_aligned}/{num_aligned + missing_count} = " f"{num_aligned / (num_aligned + missing_count)}")

    # Match segments to audio, and split the wav file accordingly
    print("Segmenting audio and writing a manifest...")
    segment_audio(utt_times, audiofile, base)

    print("Done.")


if __name__ == '__main__':
    main()
