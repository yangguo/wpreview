import io
from typing import List, Union

import pandas as pd
from corrector import corrector
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from txtencoder import keybert_keywords, sent2emb

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/txtcorrector")
async def txtcorrector(audit_list: List[str] = Query(default=[])):
    result_list = corrector(audit_list)
    return {"result_list": result_list}


@app.post("/txtsent2emb")
async def txtsent2emb(sentences: List[str] = Query(default=[])):
    embedding = sent2emb(sentences)
    # convert numpy array to list
    return {"embedding": embedding.tolist()}


@app.post("/getkeywords")
async def getkeywords(
    text: str,
    topn: int = 3,
):
    keyls = keybert_keywords(text, topn)
    return {"keyls": keyls}
