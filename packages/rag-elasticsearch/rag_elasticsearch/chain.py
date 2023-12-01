from typing import List, Optional, Tuple

from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseMessage, format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from pydantic import BaseModel, Field

from .connection import es_connection_details, ELASTIC_CLOUD_ID, ELASTIC_USERNAME, ELASTIC_PASSWORD
from .prompts import CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT, LLM_CONTEXT_PROMPT, DL_QUIZ_PROMPT, SONG_PROMPT
from langchain.llms import HuggingFaceHub


print("ElasticSearch RAG")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda:0"}),

# Setup connecting to Elasticsearch
elastic_search_db = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="DL Project",
    embedding=embedding,
    es_user="elastic",
    es_password=ELASTIC_PASSWORD
)


"""
For Cloud service

vectorstore = ElasticsearchStore(
    es_api_key="essu_UjJSa1ExOVpjMEoxUzI1SFJESlRjMmxLY3pVNmVubzJVRUpwVFdKUmN5MWhMV1U1WlRVMFJqVkhVUT09AAAAADEErdk=",
    **es_connection_details,
    index_name="DL-lecture-note",
)
"""

retriever = elastic_search_db.as_retriever()

# Set up LLM to user
llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha", 
                     huggingfacehub_api_token="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")


def _combine_documents(docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

''''''
class ChainInput(BaseModel):
    chat_history: Optional[List[BaseMessage]] = Field(
        description="Previous chat messages."
    )
    question: str = Field(..., description="The question to answer.")


_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

chain = {"context": RunnablePassthrough()}| DL_QUIZ_PROMPT | llm | StrOutputParser()


#chain = chain.with_types(input_type=ChainInput)
