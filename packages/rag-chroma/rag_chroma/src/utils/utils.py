import os

from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import GrobidParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from pdf_loader import PDFPlumberLoaderPlus
import re

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

    return answer  # return rag_chain?


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


question = """Can you make ten True or False Quiz based on valuable knowledge for learning in lecture notes I gave you? you alsoe should give me the answer of each Quiz.
"""  # TODO: it can be dynamic due to question form can be changed T/F, essay or something

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here are lecture notes which you can make use of and my question
Lecture notes: {input}

Question: {question}
""".format(
    input="{input}", question=question
)


math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here are lecture notes which you can make use of and my question
Lecture notes: {input}

Question: {question}
{input}""".format(
    input="{input}", question=question
)

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here are lecture notes which you can make use of and my question
Lecture notes: {input}

Question: {question}
{input}""".format(
    input="{input}", question=question
)


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

Here are lecture notes which you can make use of and my question
Lecture notes: {input}

Question: {question}
{input}""".format(
    input="{input}", question=question
)

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


def get_routerChain(llm, prompt_infos):

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising\
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use or "DEFAULT"
        "next_inputs": string \ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt \
    names specified below OR it can be "DEFAULT" if the input is not\
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiPromptChain(
        router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True
    )

    return chain

def get_agent(llm):
    tools = load_tools(["llm-math","wikipedia"], llm=llm)
    agent= initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True)
    
    # Define which tools the agent can use to answer user queries
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        )    
    ]
    # Set up the base template
    template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

    Question: {input}
    {agent_scratchpad}"""

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )
    output_parser = CustomOutputParser()

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )   

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    return agent_executor # agent_executor.run("How many people live in canada as of 2023?")

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
