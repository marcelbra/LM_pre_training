import pickle
from datasets import load_from_disk, Dataset
from transformers import BertTokenizer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from datasets.arrow_dataset import Batch
from utils import _filter, _clean, _flatten


def get_data(n=10000):

    wiki = load_from_disk("/home/marcelbraasch/PycharmProjects/LM_pre_training/Wikipedia/raw")["train"]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    wiki = Dataset.from_dict(wiki[:n])  # Only for testing

    data = []

    for example in tqdm(wiki["text"]):

        # Do sentence splitting and cleaning
        sentences = [x for x in example.split("\n") if _filter(x)]
        sentences = [x.replace("() ", "") for x in sentences]
        sentences = list(map(sent_tokenize,sentences))
        sentences = _flatten(sentences)
        sentences = [_clean(x) for x in sentences]

        # Tokenize each sentence, extract and structure important information
        tokenized = [tokenizer(x, return_special_tokens_mask=True, return_length=True) for x in sentences]
        special_tokens = [x["special_tokens_mask"] for x in tokenized]
        lengths = [x["length"] for x in tokenized]
        tokens_ids = [x["input_ids"] for x in tokenized]
        tokens = list(map(tokenizer.convert_ids_to_tokens, tokens_ids))

        # Tokenize sentences into words for further analysis
        words = [word_tokenize(sentence) for sentence in sentences]

        data.append({"sentences": sentences,
                     "words": words,
                     "token_ids": tokens_ids,
                     "tokens": tokens,
                     "special_tokens": special_tokens,
                     "token_lengths": lengths})
        s = 0

    return data


#with open("Wikipedia/10kdata.pickle", "wb") as f:
#    pickle.dump(get_data(), f)