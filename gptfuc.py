import json
import os

# import faiss
import pandas as pd
from langchain import LLMChain

# from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceHubEmbeddings,
    OpenAIEmbeddings,
)

# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAIChat
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS, Chroma, Pinecone, Qdrant

# import pinecone


# from qdrant_client import QdrantClient
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction",
    huggingfacehub_api_token="***REMOVED***",
)

# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# get openai api key from config.json
api_key = config["openai_api_key"]

PINECONE_API_KEY = "***REMOVED***"
PINECONE_API_ENV = "us-west1-gcp"

# initialize pinecone
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)


qdrant_host = "127.0.0.1"
# qdrant_api_key = ""


os.environ["OPENAI_API_KEY"] = api_key

uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "fileidx"
backendurl = "http://localhost:8000"

# openai_api_key = os.environ.get("OPENAI_API_KEY")
# if openai_api_key is None:
#     print("请设置OPENAI_API_KEY")
# else:
#     print("已设置OPENAI_API_KEY" + openai_api_key)

# initialize pinecone
# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_API_ENV
# )


def gpt_vectoranswer(question, chaintype="stuff", top_k=4, model_name="gpt-3.5-turbo"):
    # get faiss client
    store = FAISS.load_local(fileidxfolder, embeddings)

    # pinecone_namespace = "bank"
    # pinecone_index_name = "ruledb"

    # # Create an index object
    # index = pinecone.Index(index_name=pinecone_index_name)

    # index_stats_response = index.describe_index_stats()
    # print(index_stats_response)

    # collection_description = pinecone.describe_index('ruledb')
    # print(collection_description)

    system_template = """根据提供的背景信息，请准确和全面地回答用户的问题。
    如果您不确定或不知道答案，请直接说明您不知道，避免编造任何信息。
    {context}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name=model_name)
    # chain = VectorDBQA.from_chain_type(
    receiver = store.as_retriever()
    receiver.search_kwargs["k"] = top_k
    chain = RetrievalQA.from_chain_type(
        llm,
        chain_type=chaintype,
        # vectorstore=store,
        retriever=receiver,
        # k=top_k,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    result = chain({"query": question})

    answer = result["result"]
    # sourcedf=None
    source = result["source_documents"]
    sourcedf = docs_to_df_audit(source)

    return answer, sourcedf


def gpt_auditanswer(question, chaintype="stuff", top_k=4, model_name="gpt-3.5-turbo"):
    # get faiss client
    store = FAISS.load_local(fileidxfolder, embeddings)

    system_template = "您是一位资深的 IT 咨询顾问，专业解决问题并能有条理地分析问题。"

    human_template = """
请根据以下政策文件，检查它们整体上是否符合 {question} 的要求。请在回答中描述您的审核过程、依据和推理。
请指出不符合规定的地方，给出改进意见和具体建议。
待审核的监管要求包括：{context}

如果您无法确定答案，请直接回答“不确定”或“不符合”，切勿编造答案。
"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name=model_name)
    # chain = VectorDBQA.from_chain_type(
    receiver = store.as_retriever()
    receiver.search_kwargs["k"] = top_k
    chain = RetrievalQA.from_chain_type(
        llm,
        chain_type=chaintype,
        # vectorstore=store,
        retriever=receiver,
        # k=top_k,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    result = chain({"query": question})

    answer = result["result"]
    # sourcedf=None
    source = result["source_documents"]
    sourcedf = docs_to_df_audit(source)

    return answer, sourcedf


def gpt_wpreview(audit_requirement, audit_procedure, model_name="gpt-3.5-turbo"):
    template = """您是一位善于解决问题、按步骤思考的专业咨询顾问。您的任务是：
    1. 验证提供的审计结果是否符合审计要求，是否存在不满足审计要求的内容，请在回答中描述您的验证过程、依据和推理。输出为：回答1。
    2. 根据当前审计结果的内容，重新描述每一步执行的审计程序、查看的审计证据名称及具体细节和审计结果，同时修复任何语法或拼写错误。有经验的专业审计师应该能够根据相关描述重复执行审计程序。输出为：回答2。
    3. 回答1和回答2之间用'\\'分隔。"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = """
    审计要求：
    {audit_requirement}

    审计结果：
    {audit_procedure}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chat = ChatOpenAI(model_name=model_name, temperature=0)

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(
        audit_requirement=audit_requirement, audit_procedure=audit_procedure
    )

    output_text = response.strip()

    # Split the output text into verification result and modified audit procedure
    output_parts = output_text.split("\\")
    verification_result = output_parts[0].strip()
    modified_audit_procedure = "\n".join(output_parts[1:]).strip()

    return verification_result, modified_audit_procedure


# convert document list to pandas dataframe
def docs_to_df_audit(docs):
    """
    Converts a list of documents to a pandas dataframe.
    """
    data = []
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        plc = metadata["source"]
        row = {"内容": page_content, "来源": plc}
        data.append(row)
    df = pd.DataFrame(data)
    return df
