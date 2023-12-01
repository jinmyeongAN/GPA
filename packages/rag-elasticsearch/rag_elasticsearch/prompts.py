from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Used to condense a question and chat history into a single question
condense_question_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. If there is no chat history, just rephrase the question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_prompt_template)

# RAG Prompt to provide the context and question for LLM to answer
# We also ask the LLM to cite the source of the passage it is answering from
llm_context_prompt_template = """
Use the following passages to answer the user's question.
Each passage has a SOURCE which is the title of the document. When answering, cite source name of the passages you are answering from below the answer in a unique bullet point list.

If you don't know the answer, just say that you don't know, don't try to make up an answer.

----
{context}
----
Question: {question}
"""  # noqa: E501

LLM_CONTEXT_PROMPT = ChatPromptTemplate.from_template(llm_context_prompt_template)

# Used to build a context window from passages retrieved
document_prompt_template = """
---
NAME: {name}
PASSAGE:
{page_content}
---
"""

DOCUMENT_PROMPT = PromptTemplate.from_template(document_prompt_template)


dl_quiz_template = """Your the computer science and machine learning professor. You have to say something about question by considering the context,
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
Question: "Can you make ten True or False Quiz using the important information in context? you should give me the answer of each Quiz."
Helpful Answer:
"""

DL_QUIZ_PROMPT = PromptTemplate.from_template(dl_quiz_template)


song_template = "Write me a song about goldfish on the moon"

SONG_PROMPT = PromptTemplate.from_template(song_template)