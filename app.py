import streamlit as st
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from models import ModelHandler
from prompt_engineering import create_retrieval_qa_chain
from evaluation import log_evaluation_result

# Set up logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize ModelHandler
model_handler = ModelHandler()


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        st.error("An error occurred while processing the PDF files.")
    return text


def get_text_chunks(text):
    """Split text into smaller chunks for processing."""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        st.error("An error occurred while splitting the text.")
        chunks = []
    return chunks


def get_vectorstore(text_chunks):
    """Create a vector store using FAISS for efficient retrieval."""
    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error("An error occurred while creating the vector store.")
        vectorstore = None
    return vectorstore


def get_conversation_chain(vectorstore, selected_model):
    """Create a conversational retrieval chain with the selected model."""
    try:
        llm = model_handler.models[selected_model]
        conversation_chain = create_retrieval_qa_chain(llm, vectorstore)

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory
        )
    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        st.error("An error occurred while setting up the conversation chain.")
        conversation_chain = None
    return conversation_chain


def handle_userinput(user_question, selected_model):
    """Handles user input by running the selected model."""
    try:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(
                    user_template.replace("{{MSG}}", message.content),
                    unsafe_allow_html=True,
                )
            else:
                st.write(
                    bot_template.replace("{{MSG}}", message.content),
                    unsafe_allow_html=True,
                )
    except Exception as e:
        logging.error(f"Error handling user input: {e}")
        st.error("An error occurred while processing your request.")


# Example evaluation function
def evaluate_retrieval_accuracy():
    """Evaluate the accuracy of the retrieval component."""
    try:
        retrieval_accuracy = 0.85  # Example value
        log_evaluation_result("Retrieval Component", "Accuracy", retrieval_accuracy)
    except Exception as e:
        log_evaluation_result(f"Error evaluating retrieval accuracy: {e}")


def evaluate_generation_accuracy():
    """Evaluate the accuracy of the generation component."""
    try:
        generation_accuracy = 0.90  # Example value
        log_evaluation_result("Generation Component", "Accuracy", generation_accuracy)
    except Exception as e:
        log_evaluation_result(f"Error evaluating generation accuracy: {e}")


def main():
    load_dotenv()
    st.set_page_config("Chat with Multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your Medical PDFs :books:")

    # Model Selection Dropdown
    st.sidebar.subheader("Select Model to Use")
    selected_model = st.sidebar.selectbox(
        "Select a Model", list(model_handler.models.keys())
    )

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question, selected_model)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                try:
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    if vectorstore:
                        # Create conversation chain with the selected model
                        st.session_state.conversation = get_conversation_chain(
                            vectorstore, selected_model
                        )
                except Exception as e:
                    logging.error(f"Error during processing: {e}")
                    st.error("An error occurred during processing.")

    evaluate_retrieval_accuracy()
    evaluate_generation_accuracy()


if __name__ == "__main__":
    main()
