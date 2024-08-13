import os
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import (SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate,PromptTemplate)
from dotenv import load_dotenv, dotenv_values 
from neo4j import GraphDatabase
import logging
import sys

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logging.getLogger("neo4j").addHandler(handler)
logging.getLogger("neo4j").setLevel(logging.DEBUG)

# loading variables from .env file
load_dotenv() 

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(HOSPITAL_QA_MODEL)
#chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125",api_key=OPENAI_API_KEY)

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = os.getenv("NEO4J_URI")
URI = "neo4j+ssc://a131a7f2.databases.neo4j.io:7687"
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

with GraphDatabase.driver( uri =URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    




neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding= OpenAIEmbeddings(),
    url = 'neo4j+ssc://a131a7f2.databases.neo4j.io:7687',
    username = os.getenv("NEO4J_USERNAME"),
    password = os.getenv("NEO4J_PASSWORD"),
    index_name="reviews",
    node_label="Review",
    text_node_properties= [
        "physician_name",
        "patient_name",
        "text",
        "hospital_name"
    ],
    embedding_node_property= "embedding",
    )