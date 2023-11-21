import os
import openai
import streamlit as st 
from streamlit_extras.add_vertical_space import add_vertical_space 
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback



# Side bar contents
with st.sidebar:
    st.title('LLM PDF Chat')
    st.markdown('''
    ## About:
    This app is an LLM-Powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://www.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    ''')
    add_vertical_space(5)
    st.write('Created by [HaoES](https://github.com/HaoES)')


# api key
openai.api_key = os.environ["OPENAI_API_KEY"]



# main function
def main():

    st.header("Chat with a PDF!")

    # upload a PDF file:
    file = st.file_uploader("Upload your PDF", type='pdf')

    if file:
        pdf = PdfReader(file)

        text = ''
        for page in pdf.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # embeddings:
        embeddings = OpenAIEmbeddings()
        store_name = str(file.name[:-4]) + str(file.size)
        if os.path.exists(f"{store_name}"):
            vects = FAISS.load_local(f"{store_name}",embeddings)
            st.write("Embeddings loaded from disk")
        else:
            # save our embeddings
            vects = FAISS.from_texts(texts = chunks,embedding=embeddings)
            vects.save_local(f"{store_name}")
            st.write("Embeddings computed successfully")

        # intereact with user:
        query = st.text_input("What would you like to know about your PDF?")
        if query:
            docs = vects.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                #st.write(cb) uncomment to feel the pain of each of the requests
            st.write(response)



if __name__ == '__main__':
    main()
