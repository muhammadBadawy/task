import os
from dotenv import load_dotenv, find_dotenv

def get_openai_api_key():
    """
    Retrieve the OpenAI API key from the environment variables.

    This function attempts to find and load the .env file containing environment variables using `find_dotenv()` and `load_dotenv()`.
    It searches for the .env file in the project directory or its parent directories. If a .env file is successfully found and loaded,
    the function then tries to retrieve the "OPENAI_API_KEY" from the loaded environment variables.

    Returns:
        str or None: If the .env file is found and loaded, and the "OPENAI_API_KEY" environment variable is set, the API key is returned.
                     If either the .env file is not found, not loaded, or the "OPENAI_API_KEY" variable is not set, `None` is returned.

    Note:
        It's important to ensure that the .env file containing the "OPENAI_API_KEY" is properly placed within the project directory or its parent directories.
        If the .env file or the "OPENAI_API_KEY" are missing, the function will return `None`, indicating the absence of the API key.
    """
    return _get_env_val("OPENAI_API_KEY")


def get_chroma_persist_directory():
    """
    Get the directory path where chroma persist files are stored.

    Returns:
        str: The directory path if found, otherwise "None".
    """
    return _get_env_val("CHROMA_PERSIST_DIR")

def _get_env_val(key):
    """
    Get the value of an environment variable.

    Args:
        key (str): The key of the environment variable to retrieve.

    Returns:
        str: The value of the environment variable if found, otherwise "None".
    """
    if load_dotenv(find_dotenv()):
        return os.getenv(key)
    else:
        return "None"

def read_non_empty_lines(file_path):
    """
    Read non-empty lines from a file.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        list: A list of non-empty lines stripped of leading and trailing whitespaces.
    """
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:  # Check if the line is not empty
                lines.append(stripped_line)
    return lines

def langchain_doc_to_text(doc):
    """
    Extract the text content from a document object.

    Args:
        doc: The document object.

    Returns:
        str: The text content of the document.
    """
    return doc.page_content

def format_docs(docs):
    """
    Format a list of document objects into a list of text contents.

    Args:
        docs (list): A list of document objects.

    Returns:
        list: A list of text contents extracted from the document objects.
    """
    return [langchain_doc_to_text(doc) for doc in docs]

def read_file_into_string(file_path):
    """
    Read the content of a file into a string.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        str: The content of the file as a string, or None if the file is not found.
    """
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            return file_content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
