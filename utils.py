import numpy as np
import pandas as pd
import asyncio
import torch
import jieba.analyse
import spacy
from spacy_streamlit import visualize_ner

from textrank4zh import TextRank4Sentence
from transformers import RoFormerModel, RoFormerTokenizer

modelfolder = 'junnyu/roformer_chinese_sim_char_ft_base'

tokenizer = RoFormerTokenizer.from_pretrained(modelfolder)
model = RoFormerModel.from_pretrained(modelfolder)

# nlp = spacy.load('zh_core_web_lg')
nlp = spacy.load('zh_core_web_trf')


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


async def sent2emb(sents):
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
    tr4s.analyze(text=text, lower=True, source='all_filters')
    sumls = []
    for item in tr4s.get_key_sentences(num=3):
        sumls.append(item.sentence)
    summary = ''.join(sumls)
    return summary


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def roformer_encoder(sentences):
    # Tokenize sentences
    encoded_input = tokenizer(sentences,
                              max_length=512,
                              padding=True,
                              truncation=True,
                              return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask']).numpy()
    return sentence_embeddings


# replace color of word in text match in word list
def replace_color(proc_list, proc_keywords, color):
    highlight_proc = []
    for proc, proc_key in zip(proc_list, proc_keywords):
        newproc = []
        sentls = cut_sentences(proc)
        # replace sent in proc if match with keyword
        for sent in sentls:
            if sent in proc_key:
                newproc.append(
                    sent.replace(
                        sent,
                        '<span style="color:{}">{}</span>'.format(color,
                                                                  sent)))
            else:
                newproc.append(sent)
        proc = ''.join(newproc)
        highlight_proc.append(proc)
    return highlight_proc


    # find similar words in doc embedding
def find_similar_words(words, doc, threshold_key=0.5, top_n=3):
    # compute similarity
    similarities = {}
    for word in words:
        tok = nlp(word)
        similarities[tok.text] = {}
        for tok_ in doc:
            similarities[tok.text].update({tok_.text: tok.similarity(tok_)})
    # sort
    topk = lambda x: {
        k: v
        for k, v in sorted(similarities[x].items(),
                           key=lambda item: item[1],
                           reverse=True)[:top_n]
    }
    result = {word: topk(word) for word in words}
    # filter by threshold
    result_filter = {
        word: {k: v
               for k, v in result[word].items() if v >= threshold_key}
        for word in result
    }
    return result_filter


    # get ent label and text using spacy
def get_ent_words(text):
    doc = nlp(text)

    labels = []
    textls = []
    for ent in doc.ents:
        labels.append(ent.label_)
        textls.append(ent.text)

    # combine labels and text into df ordered by labels
    df = pd.DataFrame({'类型': labels, '文本': textls})
    df = df.sort_values(by='类型')
    return df


def tfidfkeyword(text, top_n=5):
    text = ' '.join(cut_sentences(text))
    tags = jieba.analyse.extract_tags(text,
                                      topK=top_n,
                                      allowPOS=('ns', 'n', 'nr', 'm', 'ns',
                                                'nt', 'nz', 't', 'q'))
    return tags


# cut text into words using spacy
def cut_sentences(text):
    # cut text into words
    doc = nlp(text)
    sents = [t.text for t in doc]
    return sents


# convert text spacy to word embedding
def text2emb(text):
    # cut text into words
    doc = nlp(text)
    return doc


# display entities using spacy
def display_entities(text, key):
    doc = nlp(text)
    visualize_ner(doc,
                  labels=nlp.get_pipe("ner").labels,
                #   labels=None,
                  key=key,
                  show_table=False,
                  title=None)