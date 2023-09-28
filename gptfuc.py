import json
import os

# import faiss
import pandas as pd
from dotenv import load_dotenv
from langchain import LLMChain

# from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.chains.question_answering import load_qa_chain

# from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

# from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceHubEmbeddings,
    OpenAIEmbeddings,
)
from langchain.output_parsers import (
    ResponseSchema,
    RetryWithErrorOutputParser,
    StructuredOutputParser,
)
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

load_dotenv()

AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_NAME_16K = os.environ.get("AZURE_DEPLOYMENT_NAME_16K")
AZURE_DEPLOYMENT_NAME_GPT4 = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
AZURE_DEPLOYMENT_NAME_GPT4_32K = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_32K")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")


# from qdrant_client import QdrantClient
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=HF_API_TOKEN,
)


uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "fileidx"
backendurl = "http://localhost:8000"


# import openai
# openai.api_base="https://tiny-shadow-5144.vixt.workers.dev/v1"
# openai.api_base="https://super-heart-4116.vixt.workers.dev/v1"
# openai.api_base="https://az.139105.xyz/v1"

# llm = ChatOpenAI(model_name=model_name )
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=AZURE_API_KEY )


# use azure model
#     llm = AzureChatOpenAI(
#     openai_api_base=AZURE_BASE_URL,
#     openai_api_version="2023-03-15-preview",
#     deployment_name=AZURE_DEPLOYMENT_NAME,
#     openai_api_key=AZURE_API_KEY,
#     openai_api_type = "azure",
# )
# use cohere model
# llm = Cohere(model="command-xlarge-nightly",cohere_api_key=COHERE_API_KEY,temperature=0)

# convert gpt model name to azure deployment name
gpt_to_deployment = {
    "gpt-35-turbo": AZURE_DEPLOYMENT_NAME,
    "gpt-35-turbo-16k": AZURE_DEPLOYMENT_NAME_16K,
    "gpt-4": AZURE_DEPLOYMENT_NAME_GPT4,
    "gpt-4-32k": AZURE_DEPLOYMENT_NAME_GPT4_32K,
}

# use azure llm based on model name
def get_azurellm(model_name):
    deployment_name = gpt_to_deployment[model_name]
    llm = AzureChatOpenAI(
        openai_api_base=AZURE_BASE_URL,
        openai_api_version="2023-07-01-preview",
        deployment_name=deployment_name,
        openai_api_key=AZURE_API_KEY,
        openai_api_type="azure",
    )
    return llm


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


def gpt_wpreview(audit_requirement, audit_procedure, model_name="gpt-35-turbo"):

    response_schemas = [
        ResponseSchema(name="错误检查", description="回答1的具体内容"),
        ResponseSchema(name="更新结果", description="回答2的具体内容"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    template = """
    您是一位专业咨询顾问，善于解决问题并按步骤思考。请完成以下任务：
    1. 仔细阅读并分析提供的审计要求和审计结果。
    2. 验证审计结果是否符合审计要求。如有不符合的内容，请提供您的验证过程、依据和推理。将这部分回答称为"回答1"。
    3. 重新描述审计程序的每个步骤，以清单形式列出每个步骤的审计过程和审计结果，并引用审计程序的具体内容，确保经验丰富的审计师能够根据描述重复执行审计程序并得到相同的结果。同时修复任何语法或拼写错误。将这部分回答称为"回答2"。
    
    {format_instructions}
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = """
    审计要求：
    {audit_requirement}

    审计结果：
    {audit_procedure}
    """

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate(
        messages=[system_message_prompt, human_message_prompt],
        input_variables=["audit_requirement", "audit_procedure"],
        partial_variables={"format_instructions": format_instructions},
    )

    llm = get_azurellm(model_name)

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run(
        audit_requirement=audit_requirement, audit_procedure=audit_procedure
    )
    # return response
    # print(response)

    retry_parser = RetryWithErrorOutputParser.from_llm(parser=output_parser, llm=llm)

    try:
        response_json = output_parser.parse(response)
    except Exception as e:
        print(e)
        response_json = retry_parser.parse_with_prompt(response, chat_prompt)

    # print(response_json)
    # load json response
    # response_json = json.loads(resp)

    # get verification result
    verification_result = response_json["错误检查"]
    # get modified audit procedure
    modified_audit_procedure = response_json["更新结果"]

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
