# Chat with Your Medical PDFs

## Documentation

This project is a Streamlit-based web application designed to assist medical students in their research by providing an interactive way to query and extract information from multiple PDF documents. The application simplifies the process of reading through large volumes of medical PDFs by allowing users to ask questions and get responses based on the content of the uploaded documents.

## Tech Stack

- **Streamlit**: For creating the interactive web application.
- **LangChain**: For handling language models and conversational chains.
- **PyPDF2**: For extracting text from PDF files.
- **FAISS**: For efficient vector store creation and retrieval.
- **HuggingFace**: For embedding models used in the vector store.
- **Logging**: To handle and record errors and evaluation results.
- **dotenv**: To manage environment variables.

## Features

- **PDF Text Extraction**: Extracts and processes text from multiple PDF documents.
- **Text Chunking**: Splits the extracted text into manageable chunks.
- **Vector Store**: Uses FAISS for efficient text retrieval.
- **Conversational Retrieval**: Allows users to ask questions about the content of the PDFs.
- **Evaluation Logging**: Records evaluation results for the retrieval and generation components.

## Purpose

The application is specifically developed for medical students to streamline their research process. It helps them efficiently manage and query through multiple medical PDFs, making it easier to find and utilize relevant information.

## Technical Walkthrough

### File Structure

- `app.py`: The main Streamlit application file that handles PDF processing, text extraction, and user interactions.
- `prompt_engineering.py`: Defines the prompt template and creates a RetrievalQA chain using LangChain.
- `requirements.txt`: Lists the necessary Python packages for the project.
- `models.py`: Contains the `ModelHandler` class for managing and running various language models.
- `htmlTemplates.py`: Contains HTML templates for displaying chat messages.
- `evaluation.py`: Manages logging of evaluation results to `evals.log`.

### Running the Application Locally

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Create a Virtual Environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   # On Windows use `venv\Scripts\activate`


3. **Install Dependencies:**:
   ```bash
   pip install -r requirements.txt


4. **Run the Streamlit Application:**:
   ```bash
   streamlit run app.py



5. **Open the Application: Navigate to http://localhost:8501 in your web browser to interact with the application.**:


## Live Application
The application is deployed and live at Streamlit Live URL. You can access it to interact with the application and see it in action

## Future Improvements
- **Model Optimization:** Fine-tune or update models for better performance.
- **Enhanced Error Handling:** Improve error messages and handling for edge cases.
- **User Interface:** Refine UI/UX for a more intuitive user experience.
- **Additional Features:** Add support for more document formats and advanced query capabilities.

## Contributing
Feel free to fork the repository, make improvements, and submit pull requests. For any issues or feature requests, please open an issue on GitHub

