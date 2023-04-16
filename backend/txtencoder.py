import asyncio

import numpy as np
import spacy
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from textrank4zh import TextRank4Sentence
from transformers import RoFormerModel, RoFormerTokenizer

smodel = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

modelfolder = "junnyu/roformer_chinese_sim_char_ft_base"

nlp = spacy.load("zh_core_web_lg")

tokenizer = RoFormerTokenizer.from_pretrained(modelfolder)
model = RoFormerModel.from_pretrained(modelfolder)


def roformer_encoder(sentences):
    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(
        model_output, encoded_input["attention_mask"]
    ).numpy()
    return sentence_embeddings


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def sent2emb(sents):
    embls = []
    for sent in sents:
        # get summary of sent
        summarize = get_summary(sent)
        sentence_embedding = roformer_encoder(summarize)
        embls.append(sentence_embedding)
    all_embeddings = np.concatenate(embls)
    return all_embeddings


# get summary of text
def get_summary(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source="all_filters")
    sumls = []
    for item in tr4s.get_key_sentences(num=3):
        sumls.append(item.sentence)
    summary = "".join(sumls)
    return summary


# def async sent2emb(sentences):
def sent2emb_async(sentences):
    """
    run sent2emb in async mode
    """
    # create new loop
    loop = asyncio.new_event_loop()
    # run async code
    asyncio.set_event_loop(loop)
    # run code
    task = loop.run_until_complete(sent2emb(sentences))
    # close loop
    loop.close()
    return task


# get keyword list using keybert
def keybert_keywords(text, top_n=3):
    doc = " ".join(cut_sentences(text))
    bertModel = KeyBERT(model=smodel)
    # keywords = bertModel.extract_keywords(doc,keyphrase_ngram_range=(1,1),stop_words=None,top_n=top_n)
    # mmr
    keywords = bertModel.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        use_mmr=True,
        diversity=0.7,
        top_n=top_n,
    )
    keyls = []
    for (key, val) in keywords:
        keyls.append(key)
    return keyls


# cut text into words using spacy
def cut_sentences(text):
    # cut text into words
    doc = nlp(text)
    sents = [t.text for t in doc]
    return sents
