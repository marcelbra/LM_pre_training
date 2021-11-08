"""
BERT pre-training set up
"""
import numpy as np
from datasets import load_dataset, load_from_disk#, save_to_disk
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from transformers import (
    RobertaTokenizer,
    RobertaConfig,
    RobertaModel,
    PreTrainedTokenizerBase,
    FlaxAutoModelForMaskedLM
)
from nltk.tokenize import sent_tokenize
import pickle
from tqdm import tqdm
import json
from preprocess import get_data
from dataclasses import dataclass, field
import flax
import jax
from typing import Dict, List, Optional, Tuple
from masking import FlaxDataCollatorForLanguageModeling
import jax.numpy as jnp
import optax

# Constants
mlm_probability = 0.15
seed = 42
dtype = "float32"
num_epochs = 10
train_batch_size = 128
eval_batch_size = 128
warmup_steps = 1000
learning_rate = "5e-3"
dataset_amount = 10000
num_train_steps = dataset_amount // train_batch_size * num_epochs

# Initialize tokenizer and data collator
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
data_collator = FlaxDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

# Initialize training
rng = jax.random.PRNGKey(training_args.seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())

# Initialize model
config = RobertaConfig.from_pretrained("roberta-base", vocab_size=50265)
tiny = {"num_hidden_layers" : 2, "hidden_size" : 256,}
config.update(tiny)
model = FlaxAutoModelForMaskedLM.from_config(config, seed=seed, dtype=getattr(jnp, dtype))

# Create learning rate schedule
warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps)
decay_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps - warmup_steps)
linear_decay_lr_schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])

# Store some constant




s = 0

"""
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

def reformat(document):
    #Transform ([sentences], [tokens]) into [sentence, tokens].
    sentences, tokens = document["sentences"], document["tokenized"]
    new_doc = []
    for i in range(len(sentences)):
        new_doc.append({"sentence": sentences[i],
                        "tokens": tokens[i],
                        "length": len(tokens[i]["input_ids"])
                        })
    return new_doc

def doc_sentences(document, n=512):
    #Splits a given document into chunks of maximally token size n.
    counter, new_docs ,new_doc = 0, [], []
    for example in document:
        length = example["length"]
        # If maximum length will not be reached append the sentence
        if counter + length <= n:
            new_doc.append(example)
            counter += length
        # Else finish the document and reset
        else:
            new_docs.append(new_doc)
            new_doc, counter = [], length
    new_docs.append(new_doc)
    return new_docs

def write_doc_sentences(samples):
    with open("formated.txt", "w", encoding="utf-8") as f:
        s = ""
        for sample in samples:
            for doc in sample:
                for element in doc:
                    s += element["sentence"] + "\n"
                s += "\n"
            s += "\n"
        f.write(s)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
corpus = corpus[:100]
samples = []
for document in corpus:
    document = reformat(document)
    # Apply doc_sentences strategy
    document = doc_sentences(document)
    samples.append(document)
write_doc_sentences(samples)




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