"""
BERT pre-training set up
"""

from datasets import load_dataset, load_from_disk#, save_to_disk
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from nltk.tokenize import sent_tokenize
import pickle
from tqdm import tqdm
import json

# Variables
max_seq_length = 512

# Read 10k wiki documents
corpus = None
with open('Wikipedia/processed_10k.pkl', 'rb') as f:
    corpus = pickle.load(f)

vocab = None
with open('vocab.json') as f:
    vocab = json.load(f)
    vocab = dict(zip(vocab.values(), vocab.keys()))


    """
    _sum, c = 0, 0
    for x in document["tokenized"]:
        s = len(x["input_ids"])
        if _sum + s >= 512: break
        _sum += s
        c += 1
    p = document["tokenized"][:c]

    sents_amount = len(document["sentences"])
    counter = 0
    sents, toks, specials = [], [], []
    for i in range(sents_amount):
        tok = document["tokenized"][i]
        tok_amount = len(tok)
        #counter +=

        sent = document["sentences"][i]
        #doc = {:}
        #doc.append()

    #return [lst[i:i + n] for i in range(0, len(lst), n)]
    """

def chunks(document, n=512):
    """Chunks a document into x parts of maximal size n."""

    # If document tokens is < n just return it
    amount_tokens = sum([len(x["input_ids"]) for x in document["tokenized"]])
    if amount_tokens <= 512: return document

    # Find out the thresholds for slicing into chunks of size max. n
    _sum, indices = 0, [0]
    for i, x in enumerate(document["tokenized"]):
        amount_tokens = len(x["input_ids"])
        if _sum + amount_tokens >= n:
            indices.append(i)
            _sum = 0
        _sum += amount_tokens


    for i in range(len(indices) - 1):
        new_doc = dict()
        start, end = indices[i], indices[i + 1]
        new_doc = {"sentences": document["sentences"][start:end],
                   "tokenized": document["tokenized"][start:end]}


    s = 0

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
corpus = corpus[:100]
samples = []
for document in corpus:
    length = sum([len(x["input_ids"]) for x in document["tokenized"]])
    document = chunks(document)
    s = 0



s = 0

"""

# Preprocess data
column_names = ["title", "text"]
text_column_name = column_names[1]
max_seq_length = min(max_seq_length, tokenizer.model_max_length)

def tokenize_function(article):
    return tokenizer(article, return_special_tokens_mask=True)

#tokenized_dataset = [tokenize_function(example["text"]) for example in wiki]
#
# Test

tokenized_dataset = wiki.map(
    tokenize_function,
    input_columns=[text_column_name],
    batched=True,
    num_proc=32,
    remove_columns=column_names,
    load_from_cache_file=True
)


# Create config
config = RobertaConfig.from_pretrained("roberta-base", vocab_size=50265)
tiny = {"num_hidden_layers" : 2, "hidden_size" : 256,}
config.update(tiny)





# Create model
#model = RobertaModel(config)

"""