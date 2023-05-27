from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Python for Engineers - AI PDF Companion", page_icon="🐍", layout="wide")
    st.header("Python for Engineers - AI PDF Companion 🐍")
    
    # Add banner image
    image_url = "https://github.com/joreilly86/langchain-ask-pdf/blob/main/images/Banner.png?raw=true"
    st.image(image_url, use_column_width=True)
    
    # Add heading 2
    st.header("Welcome to Python for Engineers! 👋")

    # Add text
    st.title("Contextualizing Design Guidance Documents")

    st.write("This chatbot serves as a small example of the kinds of projects we will develop throughout our course. It contextualizes design guidance documents and reports, providing critical design information quickly and accurately. Engineers can upload their documents, ask questions, and gain a comprehensive understanding of design principles, improving decision-making and ensuring compliance with industry standards.")

    st.write("Benefits of using our chatbot:")
    st.markdown("- Quick and accurate access to critical design information")
    st.markdown("- Improved decision-making process")
    st.markdown("- Enhanced understanding of design principles")
    st.markdown("- Streamlined engineering workflow")
    st.markdown("- Ensured compliance with industry standards")

    st.write("Make informed design choices and propel your engineering projects forward with our powerful chatbot.")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    
if __name__ == '__main__':
    main()
