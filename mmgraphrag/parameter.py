from common_logger import get_logger
logger = get_logger(__name__)
from dataclasses import dataclass

logger.debug("Importing SentenceTransformer...")
from sentence_transformers import SentenceTransformer
logger.debug("SentenceTransformer imported.")

from dotenv import load_dotenv

load_dotenv()

logger.debug("Environment variables loaded.")

@dataclass
class QueryParam:
    response_type: str = "Keep the responses as brief and accurate as possible. If you need to present information in a list format, use (1), (2), (3), etc., instead of numbered bullets like 1., 2., 3. "
    top_k: int = 10
    local_max_token_for_text_unit: int = 4000
    local_max_token_for_local_context: int = 6000
    # alpha: int = 0.5
    number_of_mmentities: int = 3


cache_path = './cache'

logger.debug("Setting up embedding model...")

embedding_model_dir = 'cache/all-MiniLM-L6-v2/sentence-transformers/all-MiniLM-L6-v2'
logger.debug("Loading embedding model...")
EMBED_MODEL = SentenceTransformer(embedding_model_dir, device="cpu")
# EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
logger.debug("Embedding model loaded.")
# EMBED_MODEL = SentenceTransformer(embedding_model_dir, trust_remote_code=True, device="cuda:0")

def encode(content):
    return EMBED_MODEL.encode(content)
"""
def encode(content):
    return EMBED_MODEL.encode(content, prompt_name="s2p_query", convert_to_tensor=True).cpu()
"""

mineru_dir = "./example_input/mineru_result"
import os

# Read environment variables safely
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL", "moonshot-v1-32k")  # default value if not set
URL = os.getenv("URL", "https://api.moonshot.cn/v1")

MM_API_KEY = os.getenv("MM_API_KEY")
MM_MODEL = os.getenv("MM_MODEL", "gpt-5-nano-2025-08-07")
MM_URL = os.getenv("MM_URL", "https://api.openai.com/v1")

# Optionally, check for required environment variables
if not API_KEY:
    raise EnvironmentError("Missing required environment variable: API_KEY")

if not MM_API_KEY:
    raise EnvironmentError("Missing required environment variable: MM_API_KEY")

logger.info("Configuration loaded successfully.")
