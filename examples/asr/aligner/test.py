import re

ref_transcripts = 'Sentences 1. And this the second one! Yay, one more. ' \
                  'And the final.\n New segment here.'

# Read and split transcript by utterance (roughly, sentences)
sentence_split_punc = '...', '.', '?', '!', '…', '\n', '- '
split_pattern = '|'.join(map(re.escape, sentence_split_punc))

ref_splitted = re.split(split_pattern, ref_transcripts)
# remove empty strings - left after re.split
ref_splitted = [s.strip() for s in ref_splitted if s]

print(ref_transcripts)
print('\n', ref_splitted)

# find the last word of the ith split and the first word of the (i+1)th split
last_first_words = []
for i in range(len(ref_splitted) - 1):


import pdb; pdb.set_trace()
print()
