import os

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import GrobidParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def get_pdfLoader(path: str, filename: str, mode: str = "normal"):
    if mode == "normal":
        loader = PyPDFLoader(os.path.join(path, filename), extract_images=True)
        pages = loader.load_and_split()

    if mode == "paper":
        loader = GenericLoader.from_filesystem(
            path,
            glob="*",
            suffixes=[".pdf"],
            parser=GrobidParser(segment_sentences=False),
        )
        pages = loader.load()

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
        Use ten sentences maximum and keep the answer as concise as possible. 
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
