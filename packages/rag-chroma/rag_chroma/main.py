import os

from langchain.chat_models import ChatOpenAI
from src.config.const import OPENAI_API_KEY
from src.utils.utils import get_docsSplitter, get_pdfLoader, get_prompt, get_RAG, get_vectorstorce, format_docs

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = OPENAI_API_KEY

# 1. Load
data_path_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(os.path.dirname(os.path.realpath(__file__)))))
)
data_name = "w5-ml.pdf"
pdf_pages = get_pdfLoader(path=os.path.join(data_path_dir, "Data"), filename=data_name)

# 2. Split
splits = get_docsSplitter(docs=pdf_pages)

# 3. Store
vectorstore = get_vectorstorce(splits)

# 4. Retrieve
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# 5. Generate
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt = get_prompt(mode="cs")

answer = get_RAG(
    retriever=retriever,
    llm=llm,
    prompt=prompt,
    context=format_docs(pdf_pages[:3]),
) # pdf_pages의 길이가 길면 -> 아웃풋이 다 안 나올 수도 있습니다.

print(answer)
# Routing
# chain = get_routerChain(llm=llm, prompt_infos=prompt_infos)
# chain.run(PDF_files)

# Agent
# agent_executor = get_agent(llm=llm)# agent_executor.run("How many people live in canada as of 2023?")
