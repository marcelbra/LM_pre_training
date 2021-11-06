import pickle
from datasets import load_from_disk
from transformers import RobertaTokenizer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

wiki = load_from_disk("/home/marcelbraasch/PycharmProjects/LM_pre_training/Wikipedia/raw")["train"]
new_dataset = []
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Go over each article and filter
for example in tqdm(wiki):

    sentences = [x for x in example["text"].split("\n")]#_filter(x)]
    new_sentences = {"sentences": [],
                     "tokenized": []}
    break_words = ["References", "Category", "Links"]
    skip_words = ["List of", "Category"]

    for sentence in sentences:

        # We have reached end of document
        if "References" in sentence: break
        # Filter lists, links, categories, headings
        if len(sentence.split(" ")) <= 5: continue
        if sentence.startswith("Links"): continue
        if sentence.startswith("Category"): continue
        if sentence.startswith("List of"): continue

        # Split sentences and append
        sentences = sent_tokenize(sentence)
        for sentence in sentences:
            tokenized = tokenizer(sentence,
                                  return_special_tokens_mask=True,
                                  return_attention_mask=False,
                                  return_length=True
                                  )
            new_sentences["sentences"].append(sentence)
            new_sentences["tokenized"].append(tokenized)
            s = 0

    new_dataset.append(new_sentences)

    if len(new_dataset) == 10000: break

with open('Wikipedia/processed_10k.pkl', 'wb') as f:
    pickle.dump(new_dataset, f)

print("Wrote 10k documents")