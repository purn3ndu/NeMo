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
    split_pattern = '(' + '|'.join(map(re.escape, sentence_split_punc)) + ')'
    ref_text_splitted = re.split(split_pattern, ref_text)
    # remove empty strings - left after re.split
    ref_text_splitted = [s.strip() for s in ref_text_splitted if len(s.strip()) > 0]
    # after running the above we'll have original text and split tokens in a separate list,
    # for example: ['The first sentence', '.', 'And this the second one', '!', 'Yay, one more', '.',
    # 'And the final one', '.', '', 'New segment here', '.']

    # put the punctuation used for split back to the sentence
    ref_text_splitted_clean = []
    text = ''
    for t in ref_text_splitted:
        if t in sentence_split_punc:
            text += t.strip() + ' '
        else:
            if len(text) > 0:
                ref_text_splitted_clean.append(text.strip())
                text = ''
            text += t
    if len(text) > 0:
        ref_text_splitted_clean.append(text.strip())

    # print('REF TEXT:', '\n'.join(ref_text_splitted_clean))

    pred_text = pred_text.split()
    # print(f'PRED TEXT: {pred_text}')

    # Now let's normalize the text splitted into sentences
    ref_text_normalized = normalize(ref_text_splitted_clean)

    # and then break each sentences into words
    ref_text_words = [t.split() for t in ref_text_normalized]

    # find the last word of the ith split and the first word of the (i+1)th split in the reference text
    last_first_words_pairs = []
    for i in range(len(ref_text_words) - 1):
        last_first_words_pairs.append((ref_text_words[i][-1], ref_text_words[i + 1][0]))

    print(f'Split words: {last_first_words_pairs}')

    matches = []
    prev_pos = 0
    matched_ref_pos = -1
    for i, (last, first) in enumerate(last_first_words_pairs):
        match_found = False
        if last in pred_text[prev_pos:]:
            pred_match_idx = pred_text.index(last, prev_pos)
            try:
                if first == pred_text[pred_match_idx + 1]:
                    prev_pos = pred_match_idx + 2
                    match_found = True
                else:
                    prev_pos += 1
            except IndexError:
                import pdb

                pdb.set_trace()
                print(pred_match_idx)

        if match_found:
            matches.append((pred_match_idx, ' '.join(ref_text_splitted_clean[matched_ref_pos + 1 : i + 1])))
            matched_ref_pos = i
        else:
            print(f'{i} skipped - no matches found in the predicted text')
    matches.append((None, ' '.join(ref_text_splitted_clean[matched_ref_pos + 1 :])))
    return matches


if __name__ == '__main__':
    ref_text = (
        'The first sentence. And this the second one! Yay, one more. And the final one. ! New segment here. LAST ONE.'
    )
    pred_text = 'the first sentence and this the second one yay one more gnd the final one new segment here last one.'

    matches = find_matches(ref_text=ref_text, pred_text=pred_text)
    print(matches)
