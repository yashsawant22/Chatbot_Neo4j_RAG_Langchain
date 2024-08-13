from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(reviews, OpenAIEmbeddings(api_key='sk-proj-wsv4lKXPsReT3ghKrRTPT3BlbkFJnn2UGSQhQ2uALaJX3nhQ'),persist_directory = REVIEWS_CHROMA_PATH )

#reviews_vector_db = Chroma(persist_directory = REVIEWS_CHROMA_PATH,embedding_function = OpenAIEmbeddings(api_key='sk-proj-wsv4lKXPsReT3ghKrRTPT3BlbkFJnn2UGSQhQ2uALaJX3nhQ'))


question = """Has anyone complained about
           communication with the hospital staff?"""

relevant_docs = reviews_vector_db.similarity_search(question, k=3)

print(relevant_docs)