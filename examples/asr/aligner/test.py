import re
import string
from collections import OrderedDict
from typing import List


def normalize(transcripts: List[str]):
    """
    Normalizes text: removes punctuation and applies lower case

    Args:
        transcripts: original transcripts

    Returns:
        normalized transcripts
    """
    all_punct_marks = string.punctuation.replace("'", '')
    transcripts = [re.sub('[' + all_punct_marks + ']', '', t).lower() for t in transcripts]
    return transcripts


def find_matches(ref_text: str, pred_text: str):
    """
    Finds words matches around EOS punctuation

    Args:
        ref_text:
        pred_text:

    Returns:
        matches: Dict[int, int]: a dict where the key is the id of the splitted ref_text,
                values: position of the last word in predicted text.
    """

    # Read and split transcript by utterance (roughly, sentences)
    sentence_split_punc = '...', '.', '?', '!', '…', '\n', '- '
    split_pattern = '|'.join(map(re.escape, sentence_split_punc))

    ref_text_splitted = re.split(split_pattern, ref_text)
    # remove empty strings - left after re.split
    ref_text_splitted = [s.strip() for s in ref_text_splitted if s]

    # print(f'REF TEXT:  {ref_text_splitted}')

    pred_text = pred_text.split()
    # print(f'PRED TEXT: {pred_text}')

    # Now let's normalize the text splitted into sentences
    ref_text_splitted = normalize(ref_text_splitted)

    # and then break each sentences into words
    ref_text_words = [t.split() for t in ref_text_splitted]

    # find the last word of the ith split and the first word of the (i+1)th split
    last_first_words_pairs = []
    for i in range(len(ref_text_words) - 1):
        last_first_words_pairs.append((ref_text_words[i][-1], ref_text_words[i + 1][0]))

    print(f'Split words: {last_first_words_pairs}')

    matches = OrderedDict()
    prev_pos = 0
    for i, (first, last) in enumerate(last_first_words_pairs):
        try:
            match_idx = pred_text.index(first, prev_pos)
            if last == pred_text[match_idx + 1]:
                prev_pos = match_idx + 2

            matches[i] = match_idx
        except Exception:
            matches[i] = None
            print(f'{i} skipped - no matches found in the predicted text')

    return matches

if __name__ == '__main__':
    ref_text = 'The first sentence. And this the second one! Yay, one more. ' 'And the final one.\n New segment here.'

    pred_text = 'the first sentence and this the second one yay one more ' 'and the final one new segment here.'

    matches = find_matches(ref_text=ref_text, pred_text=pred_text)
    print(matches)
