import os
import tempfile

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.llms.openai import OpenAI
from utils.utils import get_docsSplitter, get_prompt, get_RAG, get_vectorstorce

# Streamlit app
st.subheader("🚨 Save your GPA")

# Get OpenAI API key and source document input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
    st.caption("*If you don't have an OpenAI API key, get it [here](https://platform.openai.com/account/api-keys).*")
source_doc = st.file_uploader("Source Document", label_visibility="collapsed", type="pdf")

# If the 'Summarize' button is clicked
if st.button("Make Quiz!"):
    # Validate inputs
    if not openai_api_key.strip() or not source_doc:
        st.error("Please provide the missing fields.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(source_doc.read())
                    print(tmp_file.name)
                loader = PyPDFLoader(tmp_file.name)
                pages = loader.load_and_split()
                os.remove(tmp_file.name)

                splits = get_docsSplitter(docs=pages)
                print(f"this is splits {splits}")

                vectorstore = get_vectorstorce(splits)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
                prompt = get_prompt(mode="cs")

                # Initialize the OpenAI module, load and run the summarize chain
                llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

                answer = get_RAG(
                    retriever=retriever,
                    llm=llm,
                    prompt=prompt,
                    question="Can you make ten True or False Quiz using the important information in context? you should give me the answer of each Quiz.",
                )

                print(answer)

                st.success(answer)
        except Exception as e:
            st.exception(f"An error occurred: {e}")
