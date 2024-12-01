from collections import defaultdict
import pprint

words = open('names.txt', 'r').read().splitlines()

# basic information about words
# print(f"len of words: {len(words)}")
# print(f"longest word: {max(words, key=lambda x: len(x))}")

# bi-gram
b = defaultdict(int)
for w in words:
    w = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(w, w[1:]):
        b[(ch1, ch2)] += 1

# pprint.pprint(b)

