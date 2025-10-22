import warnings
from common_logger import get_logger

logger = get_logger(__name__)
logger.info("starting!d(^_^o)")
# Ignore FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

logger.debug("Loading mmgraphrag module...")
from mmgraphrag import MMGraphRAG
logger.debug("MMGraphRAG module loaded successfully")
from time import time

pdf_path = "./example_input/2020.acl-main.45.pdf"
WORKING_DIR = "./example_output"
question = "How does the paper propose to calculate the coefficient \u03b1 for the Weighted Cross Entropy Loss?"


def index():
    logger.info("Creating MMGraph class...")
    rag = MMGraphRAG(working_dir=WORKING_DIR, input_mode=2)
    start = time()
    logger.info("Indexing PDF:" + pdf_path)
    rag.index(pdf_path)
    logger.info("Indexing completed successfully.")
    logger.info("Indexing time: %s seconds", time() - start)


def query():
    logger.info("Creating MMGraph class for querying...")
    rag = MMGraphRAG(
        working_dir=WORKING_DIR,
        query_mode=True,
    )
    logger.info("Querying question: %s", question)
    result = rag.query(question)
    logger.info("Query result: %s", result)
    logger.info("Querying completed successfully.")


if __name__ == "__main__":
    index()
    query()
