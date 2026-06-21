from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=700
    )
)

chat_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a helpful AI assistant.

    User Question: {question}

    Answer:
    """
)

chat_chain = chat_prompt | llm | StrOutputParser()

while True:
    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = chat_chain.invoke({
        "question": question
    })

    print("\nBot:", response)