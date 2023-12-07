import os

from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chains import LLMChain, SequentialChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import GrobidParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf_loader import PDFPlumberLoaderPlus


def get_pdfLoader(path: str, filename: str, mode: str = "normal"):
    if mode == "normal":
        loader = PyPDFLoader(os.path.join(path, filename), extract_images=False)
        pages = loader.load_and_split()
    elif mode == "plus":
        loader = PDFPlumberLoaderPlus(os.path.join(path, filename), extract_images=False)
        pages = loader.load()
    elif mode == "plus_i":
        loader = PDFPlumberLoaderPlus(os.path.join(path, filename), extract_images=True)
        pages = loader.load()
    elif mode == "paper":
        loader = GenericLoader.from_filesystem(
            path,
            glob="*",
            suffixes=[".pdf"],
            parser=GrobidParser(segment_sentences=False),
        )
        pages = loader.load()
    else:
        raise ValueError(f"Unsupported mode {mode}.")
    return pages


def get_docsSplitter(docs, mode: str = "normal"):
    if mode == "normal":
        # sliding window with chunk_size and chunk_overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)

    return all_splits


def get_vectorstorce(all_splits):
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    return vectorstore


def get_RAG(retriever, llm, prompt, question):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )

    answer = rag_chain.invoke(question)

    return answer


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_prompt(mode="cs"):
    if mode == "cs":
        template = """Your the computer science and machine learning professor. You have to say something about question by considering the context,
        which will be given later. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use 30 sentences maximum and keep the answer as concise as possible. 
        The answer form is 

        Quiz 1: What is the most famous method of optimization in machine learning recently
        Answer 1: Gradient descent

        Quiz 2: What should we know to find MAP from MLE?
        Answer 2: Prior

        Always say "thanks for asking!" at the end of the answer. 
        This is start of context. {context} Here is end of context.
        Question: {question}
        Helpful Answer:"""

        rag_prompt_custom = PromptTemplate.from_template(template)

        return rag_prompt_custom


def get_agent(retriever, llm, prompt):
    tool = create_retriever_tool(
        retriever,
        "search_state_of_union",
        "Searches and returns documents regarding the state-of-the-union.",
    )
    tools = [tool]
    agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
    result = agent_executor({"input": "hi, im bob"})

    print(result)

    # This is needed for both the memory and the prompt
    memory_key = "history"
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

    # system_message = SystemMessage(
    #     content=(
    #         "Do your best to answer the questions. "
    #         "Feel free to use any tools available to look up "
    #         "relevant information, only if necessary"
    #     )
    # )

    # prompt = OpenAIFunctionsAgent.create_prompt(
    #     system_message=system_message,
    #     extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    # )

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )


def get_langchain(llm, selection_prompt, expert_prompt):
    chain_seletion = LLMChain(llm=llm, prompt=selection_prompt, output_key="expert_selection")
    chain_expert = LLMChain(llm=llm, prompt=expert_prompt, output_key="answer")

    overall_chain = SequentialChain(
        chains=[chain_seletion, chain_expert],
        input_variables=["context", "question"],
        output_variables=["expert_selection", "answer"],
        verbose=True,
    )

    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise\
    and easy to understand manner. \
    When you don't know the answer to a question you admit\
    that you don't know.

    Here is a question:
    {input}"""

    math_template = """You are a very good mathematician. \
    You are great at answering math questions. \
    You are so good because you are able to break down \
    hard problems into their component parts, 
    answer the component parts, and then put them together\
    to answer the broader question.

    Here is a question:
    {input}"""

    history_template = """You are a very good historian. \
    You have an excellent knowledge of and understanding of people,\
    events and contexts from a range of historical periods. \
    You have the ability to think, reflect, debate, discuss and \
    evaluate the past. You have a respect for historical evidence\
    and the ability to make use of it to support your explanations \
    and judgements.

    Here is a question:
    {input}"""

    computerscience_template = """ You are a successful computer scientist.\
    You have a passion for creativity, collaboration,\
    forward-thinking, confidence, strong problem-solving capabilities,\
    understanding of theories and algorithms, and excellent communication \
    skills. You are great at answering coding questions. \
    You are so good because you know how to solve a problem by \
    describing the solution in imperative steps \
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""

    prompt_infos = [
        {
            "name": "physics",
            "description": "Good for answering questions about physics",
            "prompt_template": physics_template,
        },
        {"name": "math", "description": "Good for answering math questions", "prompt_template": math_template},
        {"name": "History", "description": "Good for answering history questions", "prompt_template": history_template},
        {
            "name": "computer science",
            "description": "Good for answering computer science questions",
            "prompt_template": computerscience_template,
        },
    ]

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    # destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    # destinations_str = "\n".join(destinations)

    return overall_chain
