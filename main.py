"""
BERT pre-training set up
"""

from datasets import load_dataset, load_from_disk
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

# Variables
max_seq_length = 128

# Load data sets
#book_corpus = load_dataset("bookcorpus").save_to_disk("./Data")
#wiki_corpus = load_dataset("wikipedia", "20200501.en")
wiki = load_from_disk("/home/marcelbraasch/PycharmProjects/LM_pre_training/Wikipedia/raw")

# Create tokenizer (TODO: Train from scratch using corpus)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


# Preprocess data
column_names = ["title", "text"]
text_column_name = column_names[1]
max_seq_length = min(max_seq_length, tokenizer.model_max_length)

def tokenize_function(article):
    return tokenizer(article, return_special_tokens_mask=True)

#tokenized_dataset = [tokenize_function(example["text"]) for example in wiki]


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

