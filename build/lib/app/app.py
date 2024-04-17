import os
import sys
sys.path.insert(0,'.')

import openai

# from app.utils import get_openai_api_key
from app.utils import *

# Import dependencies
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

openai.api_key = get_openai_api_key()
chroma_persist_directory = get_chroma_persist_directory()


class RAG:

    def main(self):
        """
        Executes the full Retrieval-Augmented Generation (RAG) pipeline to process a list of test questions from (test_questions.txt)
        generate answers for each, and extract relevant contexts from the original the pdf text sources(eBook.pdf).

        This method orchestrates the workflow of the RAG pipeline, starting from question processing,
        through context retrieval, answer generation, and finally compiling the results into a structured JSON format.
        The JSON object includes the original questions, their corresponding answers, and the contexts used to generate these answers.

        Note: The results will be assessed using the 'ragas' library, focusing on the following metrics:
            - Faithfulness: The degree to which the generated answers accurately reflect the information in the context.
            - Answer Relevancy: How relevant the generated answers are to the input questions.
            - Context Precision: The accuracy and relevance of the contexts retrieved to inform the answer generation.

        Returns:
            dict: A JSON-structured dictionary containing the following keys:
                - "question": A list of the input test questions.(the 3 questions from the test_questions.txt)
                - "answer": A list of generated answers corresponding to each question.
                - "contexts": The extracted contexts from original text sources that were used to inform the generation of each answer.

        """
        # Implementation of the RAG pipeline goes here
        # Initialize the OpenAI Embeddings model
        embedding_function = OpenAIEmbeddings()
        # Create (or get if existing) a vector store
        vector_store = self.get_or_create_vector_store(chroma_persist_directory, embedding_function, "./app/eBook.pdf")

        # Initialize a retriever using the vector store
        retriever = vector_store.as_retriever()
        # Initialize the OpenAI based Chat model
        model = ChatOpenAI(temperature=0)

        # Read a template for the prompt
        template = read_file_into_string("./app/prompts/chat_prompt.txt")
        # Create a Chat prompt template from the template
        prompt = ChatPromptTemplate.from_template(template)

        # Setup Langchain chain to get the answers with resources.
        chain = self.setup_chain(prompt, model, retriever)

        # Read the questions from a file
        questions = read_non_empty_lines("./app/test_questions.txt")

        # Initialize a dictionary to store the results
        results = {
            "question":[],
            "answer":[],
            "contexts":[],
        }
        # Iterate through each question
        # NOTE: we could use parallel processing to get it faster
        # since the questions are not sequential
        for question in questions:
            # Invoke the pipeline with the question
            answer = chain.invoke(question)
            # Append the question, answer, and context to the results dictionary
            results["question"].append(answer["question"])
            results["answer"].append(answer["answer"])
            results["contexts"].append(format_docs(answer["context"]))

        # Return the results
        return results

    def parse_doc(self, doc_path, chunk_size=200, chunk_overlap=50):
        """
        Parse a document and split it into chunks.

        Args:
            doc_path (str): Path to the document file.
            chunk_size (int, optional): Size of each chunk. Defaults to 200.
            chunk_overlap (int, optional): Overlap between chunks. Defaults to 50.

        Returns:
            list: List of text chunks.
        """
        # Prepare the loaders
        loader = PyPDFLoader(doc_path)
        data = loader.load()

        # Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
        docs = splitter.split_documents(data)

        return format_docs(docs)

    def get_or_create_vector_store(self, vector_store_path, embedding_function, doc_path):
        """
        Get or create a vector store for the documents.

        Args:
            vector_store_path (str): Path to the vector store directory.
            embedding_function (function): Function to get embeddings.
            doc_path (str): Path to the document file.

        Returns:
            Chroma: Chroma instance where instances are located.
        """
        if os.path.isdir(vector_store_path):
            print("Existing_chroma")
            chroma = self.read_chroma(vector_store_path, embedding_function)
        else:
            print("New_chroma")
            # Create new chroma instance
            chroma = self.setup_chorma(vector_store_path, embedding_function)
            # Upsert The data into it
            texts = self.parse_doc(doc_path)
            self.upsert_documents(chroma, texts)

        return chroma

    def setup_chorma(self, persist_directory, embedding_function):
        """
        Set up a new Chroma instance.

        Args:
            persist_directory (str): Directory to persist the Chroma instance.
            embedding_function (function): Function to compute embeddings.

        Returns:
            Chroma: Newly created Chroma instance.
        """
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        vectordb.persist()
        return vectordb

    def upsert_documents(self, chroma, texts):
        """
        Add documents in the Chroma instance.

        Args:
            chroma (Chroma): Chroma instance.
            texts (list): List of document texts to add or update.
        """
        chroma.add_texts(texts)

    def read_chroma(self, persist_directory, embedding_function):
        """
        Read an existing Chroma instance.

        Args:
            persist_directory (str): Directory where Chroma instance is persisted.
            embedding_function (function): Function to get embeddings.

        Returns:
            Chroma: Chroma instance.
        """
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    def setup_chain(self, prompt, model, retriever):
            """
            Set up the chain for retrieval and generation.

            Args:
                prompt: The prompt for answering questions.
                model: The model used for generation.
                retriever: The retriever used for context retrieval (from chroma instance).

            Returns:
                The configured chain for retrieval and generation.
            """
            # Define the basic RAG chain
            basic_rag_chain = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                    | prompt
                    | model
                    | StrOutputParser()
            )

            # Extend the chain to include source retrieval
            rag_chain_with_source = RunnableParallel(
                {"context": retriever, "question": RunnablePassthrough()}
            ).assign(answer=basic_rag_chain)

            return rag_chain_with_source
# print("hi")
# Create an instance of the RAG class
# rag_instance = RAG()
#
# # Call the main method
# print(rag_instance.main())