import os

# import faiss
import pandas as pd
from dotenv import load_dotenv
from langchain import hub
from langchain.schema import StrOutputParser
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

load_dotenv()

AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_NAME_16K = os.environ.get("AZURE_DEPLOYMENT_NAME_16K")
AZURE_DEPLOYMENT_NAME_GPT4 = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
AZURE_DEPLOYMENT_NAME_GPT4_32K = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_32K")
AZURE_DEPLOYMENT_NAME_GPT4_TURBO = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_TURBO")
AZURE_DEPLOYMENT_NAME_EMBEDDING = os.environ.get("AZURE_DEPLOYMENT_NAME_EMBEDDING")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")


# from qdrant_client import QdrantClient
# model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

# embeddings = HuggingFaceHubEmbeddings(
#     repo_id=model_name,
#     task="feature-extraction",
#     huggingfacehub_api_token=HF_API_TOKEN,
# )


# uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "fileidx"
# backendurl = "http://localhost:8000"


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
    "gpt-4-turbo": AZURE_DEPLOYMENT_NAME_GPT4_TURBO,
}

# choose chatllm base on model name
def get_chatllm(model_name):
    if model_name == "tongyi":
        llm = ChatTongyi(
            streaming=True,
        )
    elif (
        model_name == "ERNIE-Bot-4"
        or model_name == "ERNIE-Bot-turbo"
        or model_name == "ChatGLM2-6B-32K"
        or model_name == "Yi-34B-Chat"
    ):
        llm = QianfanChatEndpoint(
            model=model_name,
        )
    elif model_name == "gemini-pro":
        llm = ChatGoogleGenerativeAI(
            model=model_name, convert_system_message_to_human=True
        )
    else:
        llm = get_azurellm(model_name)
    return llm


# use azure llm based on model name
def get_azurellm(model_name):
    deployment_name = gpt_to_deployment[model_name]
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_BASE_URL,
        openai_api_version="2023-12-01-preview",
        azure_deployment=deployment_name,
        openai_api_key=AZURE_API_KEY,
        temperature=0.0,
    )
    return llm


def gpt_wpreview(audit_requirement, audit_procedure, model_name="gpt-35-turbo"):

    # response_schemas = [
    #     ResponseSchema(name="错误检查", description="回答1的具体内容"),
    #     ResponseSchema(name="更新结果", description="回答2的具体内容"),
    # ]
    # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # format_instructions = output_parser.get_format_instructions()

    # template = """
    # 您是一位专业咨询顾问，善于解决问题并按步骤思考。请完成以下任务：
    # 1. 仔细阅读并分析提供的审计要求和审计结果。
    # 2. 验证审计结果是否符合审计要求。如有不符合的内容，请提供您的验证过程、依据和推理。将这部分回答称为"回答1"。
    # 3. 重新描述审计程序的每个步骤，以清单形式列出每个步骤的审计过程和审计结果，并引用审计程序的具体内容，确保经验丰富的审计师能够根据描述重复执行审计程序并得到相同的结果。同时修复任何语法或拼写错误。将这部分回答称为"回答2"。

    # {format_instructions}
    # """
    # system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # human_template = """
    # 审计要求：
    # {audit_requirement}

    # 审计结果：
    # {audit_procedure}
    # """

    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # chat_prompt = ChatPromptTemplate(
    #     messages=[system_message_prompt, human_message_prompt],
    #     input_variables=["audit_requirement", "audit_procedure"],
    #     partial_variables={"format_instructions": format_instructions},
    # )

    chat_prompt = hub.pull("vyang/gpt_wpreview")

    llm = get_chatllm(model_name)
    output_parser = StrOutputParser()

    # chain = LLMChain(llm=llm, prompt=chat_prompt)
    # response = chain.run(
    #     audit_requirement=audit_requirement, audit_procedure=audit_procedure
    # )
    # return response
    # print(response)

    # retry_parser = RetryWithErrorOutputParser.from_llm(parser=output_parser, llm=llm)

    # try:
    #     response_json = output_parser.parse(response)
    # except Exception as e:
    #     print(e)
    #     response_json = retry_parser.parse_with_prompt(response, chat_prompt)

    # print(response_json)
    # load json response
    # response_json = json.loads(resp)

    # get verification result
    # verification_result = response_json["错误检查"]
    # get modified audit procedure
    # modified_audit_procedure = response_json["更新结果"]

    # return verification_result, modified_audit_procedure

    chain = chat_prompt | llm | output_parser
    response = chain.invoke(
        {"audit_requirement": audit_requirement, "audit_procedure": audit_procedure}
    )

    return response


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
