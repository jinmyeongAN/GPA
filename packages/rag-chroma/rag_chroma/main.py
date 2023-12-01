import os

from langchain.chat_models import ChatOpenAI
#from src.config.const import OPENAI_API_KEY
from langchain.llms import HuggingFaceHub
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import sys
sys.path.append('/home/jhkim980112/workspace/code/Deep_Learning_Proj/GPA/packages/rag-chroma')
sys.path.append('.')

from .src.utils.utils import get_docsSplitter, get_pdfLoader, get_prompt, get_RAG, get_vectorstorce, get_routerChain, prompt_infos, get_agent

#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = OPENAI_API_KEY

# 1. Load
#data_path_dir = os.path.dirname(
#    os.path.dirname(os.path.dirname(os.path.realpath(os.path.dirname(os.path.realpath(__file__)))))
#)
#data_name = "w5-ml.pdf"
#pdf_pages = get_pdfLoader(path=os.path.join(data_path_dir, "Data"), filename=data_name)

# 2. Split
#splits = get_docsSplitter(docs=pdf_pages)

# 3. Store
#vectorstore = get_vectorstorce(splits)

# 4. Retrieve
#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# 5. Generate
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Set up LLM to user
llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha", 
                     huggingfacehub_api_token="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")


#prompt = get_prompt(mode="cs")

# Routing
print("test")
chain = get_routerChain(llm=llm, prompt_infos=prompt_infos)

#chain = {"context": RunnablePassthrough()} | prompt | llm | StrOutputParser()

#chain = get_routerChain(llm=llm, prompt_infos=prompt_infos)

# chain.run(PDF_files)

# Agent
#agent_executor = get_agent(llm=llm)# agent_executor.run("How many people live in canada as of 2023?")
