import numpy as np
from utils import _flatten
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

data = None
with open('Wikipedia/10kdata.pickle', 'rb') as f:
    data = pickle.load(f)

all_words = []
for sample in data:
    words = _flatten(sample["tokens"])
    words = [x for x in words if x not in ["[SEP]", "[CLS]"]]
    all_words.extend(words)

d = defaultdict(int)
for word in all_words:
    d[word] += 1
count = len(all_words)

words_sorted_by_occurence = dict(sorted(d.items(), key=lambda item: item[1], reverse=False))
word_count_asc = list(words_sorted_by_occurence.values())
words_sorted_by_occurence = {k:v for k,v in words_sorted_by_occurence.items()
                             #if not (
                             #
                             #        len(k)>3 and not k.startswith("##")
                             #)
                            }

word_count_cum = []
for i,c in enumerate(word_count_asc):
    number = c + word_count_cum[i-1] if i > 0 else c
    word_count_cum.append(number)

s = 0

######## CDF

cdf = False

if cdf:
    word_count_prop_cum = [x/count for x in word_count_cum]
    plt.plot(range(len(word_count_prop_cum)),
             word_count_prop_cum)
    plt.title("CDF of word distribution")
    plt.xlabel("Word number ordered ascendingly")
    plt.ylabel("Percentage of corpus covered")
    plt.show()

######## PDF
if not cdf:
    word_count_prop_cum = [x/count for x in word_count_asc]
    plt.plot(range(len(word_count_prop_cum)),
             word_count_prop_cum)
    plt.title("PDF of word distribution")
    plt.xlabel("Word number ordered ascendingly")
    plt.ylabel("Percentage of corpus covered by word no. x")
    plt.show()

s = 0