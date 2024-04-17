import os
from app.utils import get_openai_api_key

os.environ["OPENAI_API_KEY"] = get_openai_api_key()