import dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
#from langchain_intro.chatbot import chat_model
#from langchain_mistralai import ChatMistralAI
from langchain.prompts import (ChatPromptTemplate, PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate )
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
import os


#openai_api_key = os.getenv("OPENAI_API_KEY")
#dotenv.load_dotenv()

#chat_model = ChatMistralAI(model="mistral-large-latest",api_key='ap9EMFnewJff7rkHV5QjcwjwT122RQ16')
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0,api_key='sk-proj-wsv4lKXPsReT3ghKrRTPT3BlbkFJnn2UGSQhQ2uALaJX3nhQ')

message = [SystemMessage(content="""You're an assistant knowledgeable about
             healthcare. Only answer healthcare-related questions."""
            ),
            HumanMessage( content='What is Medicaid managed care?')
]

review_template_str = """Your job is to use patient
reviews to answer questions about their experience at a hospital.
Use the following context to answer questions. Be as detailed
as possible, but don't make up any information that's not
from the context. If you don't know an answer, say you don't know.

{context}


"""
review_system_prompt = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["context"],template=review_template_str))

review_human_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["question"],template ="{question}"))
                                                 
messages = [review_system_prompt,review_human_prompt] 

                                                
review_prompt_template = ChatPromptTemplate(input_variables=["context","question"],
                                     messages = messages,)

context = 'i had a great stay '
question = 'Did anyone have a positive experience?'

#review_prompt_template.format(context = context, question = question))
output_parser = StrOutputParser()

REVIEWS_CHROMA_PATH = "chroma_data/"
reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings(api_key='sk-proj-wsv4lKXPsReT3ghKrRTPT3BlbkFJnn2UGSQhQ2uALaJX3nhQ')
)

reviews_retriever = reviews_vector_db.as_retriever(k=1)

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)
question = """Has anyone complained about
           communication with the hospital staff?"""
print(review_chain.invoke(question))

#from langchain_community.vectorstores.neo4j_vector import Neo4jVector