import os

from langchain.chat_models import ChatOpenAI

from src.config.const import OPENAI_API_KEY
from src.utils.utils import get_docsSplitter, get_pdfLoader, get_prompt, get_RAG, get_vectorstorce

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = OPENAI_API_KEY

# 1. Load
pdf_pages = get_pdfLoader(path="/home/jinmyeong/code/GPA/Data", filename="w5-ml.pdf")

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
    question="Can you make ten True or False Quiz using the important information in context? you should give me the answer of each Quiz.",
)

print(answer)

# 6. Agent
