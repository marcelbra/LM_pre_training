import pickle
from datasets import load_from_disk, Dataset
from transformers import RobertaTokenizer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from datasets.arrow_dataset import Batch
from utils import _filter, _clean


def get_data(dataset):

    wiki = load_from_disk("/home/marcelbraasch/PycharmProjects/LM_pre_training/Wikipedia/raw")["train"]
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    wiki = Dataset.from_dict(wiki[:10])  # Only for testing

    data = []

    for example in dataset["text"]:

        # Do sentence splitting and cleaning
        sentences = [x for x in example.split("\n") if _filter(x)]
        sentences = list(map(sent_tokenize,sentences))
        sentences = [item for sentence in sentences for item in sentence]  # Flatten
        sentences = [_clean(x) for x in sentences]

        # Tokenize each sentence, extract and structure important information
        tokenized = [tokenizer(x, return_special_tokens_mask=True, return_length=True) for x in sentences]
        special_tokens = [x["special_tokens_mask"] for x in tokenized]
        lengths = [x["length"] for x in tokenized]
        tokens = [x["input_ids"] for x in tokenized]

        data.append({"sentences": sentences,
                     "tokens": tokens,
                     "special_tokens": special_tokens,
                     "lengths": lengths})

    return data

