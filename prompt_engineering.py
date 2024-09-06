# prompt_engineering.py
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Define the prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

Helpful Answer:
"""

# Create a PromptTemplate object
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def create_retrieval_qa_chain(llm, vectorstore):
    """Create and return a RetrievalQA chain using the custom prompt."""
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return retrieval_qa
