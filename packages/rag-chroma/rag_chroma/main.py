import os

from langchain.chat_models import ChatOpenAI
from src.config.const import OPENAI_API_KEY
from src.utils.utils import get_docsSplitter, get_pdfLoader, get_prompt, get_RAG, get_vectorstorce
from langchain.agents.agent_toolkits import create_retriever_tool

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = OPENAI_API_KEY

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

tool = create_retriever_tool(
    retriever,
    "search_state_of_union",
    "Searches and returns documents regarding the state-of-the-union.",
)
tools = [tool]



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
