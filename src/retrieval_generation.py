import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from src.data_ingestion import ingest_data

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs = {"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    your ecommercebot is expert in product recomendation and customer queries.
    it anyalyzes product titles and reviews to provide accurate and helpful reponses.
    Ensure your answers are relevent to the product context and refrain from straying off-topic.
    Your responses should be concise and infomative

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:

    """

    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    llm = ChatGroq(model = "openai/gpt-oss-120b" , api_key = GROQ_API_KEY)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm 
        | StrOutputParser() 
    )
    
    return chain



if __name__ == "__main__":
    pass
    #vstore = ingest_data("connect")
    #chain = generation(vstore)

    #print(chain.invoke("can you tell me low budget sound bass headset"))
