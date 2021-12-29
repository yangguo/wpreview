import pandas as pd
import scipy.spatial

from utils import tfidfkeyword, sent2emb_async, text2emb, find_similar_words, replace_color, get_ent_words
from corrector import corrector,tup2list


def wpreview(proc_list, audit_list, threshold=0.5, threshold_key=0.5, topn=5):
    """
    encode proc_list and audit_list into df

    Parameters
    ----------
    proc_list : list
        list of process names
    audit_list : list
        list of audit names

    Returns
    -------
    df : pandas.DataFrame
        dataframe with proc_list and audit_list
    """
    proc_embeddings = sent2emb_async(proc_list)
    audit_embeddings = sent2emb_async(audit_list)

    distancels = []
    for query_embedding, audit_embedding in zip(proc_embeddings,
                                                audit_embeddings):
        distance = scipy.spatial.distance.cdist([query_embedding],
                                                [audit_embedding], "cosine")[0]
        distancels.append(1 - distance[0])

    # get keyword from proc_list and audit_list
    proc_keywords = [tfidfkeyword(proc, topn) for proc in proc_list]
    # audit_keywords = [textrankkeyword(audit) for audit in audit_list]

    audit_keywords = []
    emptyls = []
    for keyls, audit in zip(proc_keywords, audit_list):
    
        doc = text2emb(audit)
        result = find_similar_words(keyls, doc, threshold_key, top_n=3)
        subls = []
        for key in keyls:
            subls.append(list(result[key].keys()))
        # flatten subls
        subls = [item for sub in subls for item in sub]
        # remove duplicates
        subls = list(set(subls))
        audit_keywords.append(subls)

        # find keys with empty items
        empty_keys = [word for word in result if len(result[word]) == 0]
        emptyls.append(empty_keys)

    # change proc_list keywords color red by proc_keywords using markdown
    highlight_proc = replace_color(proc_list, proc_keywords, 'Turquoise')
    # change audit_list keywords color red by audit_keywords using markdown
    highlight_audit = replace_color(audit_list, audit_keywords, 'red')

    result_list = corrector(audit_list)

    errorls = []
    dferrorls = []
    # list of search list and result list
    for result in result_list:
        _, details = result
        errorls.append(details)
        dferrorls.append(tup2list(details))
    

     # display proc_list, audit_list, distancels,emptyls in a table
    df = pd.DataFrame({
        'Testing Procedure': proc_list,
        'Testing Description': audit_list,
        'Review Result': distancels,
        'Missing': emptyls,
        'Error text': dferrorls
    })

    # set df style background color gradient based on distance and threshold
    dfsty = df.style.applymap(lambda x: color_range(x, threshold),
                              subset=['Review Result'])

    return dfsty, df, highlight_proc, highlight_audit, distancels, emptyls, proc_keywords, errorls


# set bankground gradient color based on val and x
def color_range(val, x):
    color = 'green' if val > x else 'red'
    return f'background-color: {color}'
