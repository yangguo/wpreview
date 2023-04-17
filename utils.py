import pandas as pd
import requests
# import spacy

# from keybert import KeyBERT
# from sentence_transformers import SentenceTransformer
# from spacy_streamlit import visualize_ner

# smodel = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# nlp = spacy.load("zh_core_web_lg")
# nlp = spacy.load('zh_core_web_trf')

# backendurl = "http://backend.docker:8000"
backendurl = "http://localhost:8000"


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
                        sent, '<span style="color:{}">{}</span>'.format(color, sent)
                    )
                )
            else:
                newproc.append(sent)
        proc = "".join(newproc)
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
        for k, v in sorted(
            similarities[x].items(), key=lambda item: item[1], reverse=True
        )[:top_n]
    }
    result = {word: topk(word) for word in words}
    # filter by threshold
    result_filter = {
        word: {k: v for k, v in result[word].items() if v >= threshold_key}
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
    df = pd.DataFrame({"Category": labels, "Text": textls})
    df = df.sort_values(by="Category")
    return df


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
# def display_entities(text, key, labels):
#     doc = nlp(text)
#     visualize_ner(
#         doc,
#         labels=labels,
#         #   labels=None,
#         key=key,
#         show_table=False,
#         title=None,
#     )


# get nlp ner labels
# def get_ner_labels():
#     return nlp.get_pipe("ner").labels


# replace each word in list based on start and end index
def highlight_word(text_list, start, end, newtxt):
    new_text_list = newtxt
    upttxt = []
    for txt in new_text_list:
        upttxt.append('<span style="color:red">{}</span>'.format(txt))
    return text_list[:start] + upttxt + text_list[end:]


# convert tuple list to string list
def tup2list(tupls):
    strls = [str(x) for x in tupls]
    text = " ".join(strls)
    return text


# get keyword list using keybert
def keybert_keywords(text, topn=3):
    try:
        url = backendurl + "/getkeywords"
        payload = {
            "text": text,
            "topn": topn,
        }
        headers = {}
        res = requests.post(url, headers=headers, params=payload)
        result = res.json()
        keyls = result["keyls"]
    except Exception as e:
        print("转换错误: " + str(e))
        keyls = []
    return keyls


def sent2emb_async(sentences):
    try:
        url = backendurl + "/txtsent2emb"
        payload = {
            "sentences": sentences,
        }
        headers = {}
        res = requests.post(url, headers=headers, params=payload)
        result = res.json()
        embedding = result["embedding"]
    except Exception as e:
        print("转换错误: " + str(e))
        embedding = []
    return embedding
