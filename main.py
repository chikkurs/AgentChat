from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")

app = FastAPI(
    title="LLM Chat API",
    version="1.0.0"
)

# Initialize LLM
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=700
    )
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant.

User: {question}

Assistant:
"""
)

chain = prompt | llm | StrOutputParser()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "LLM Chat API"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = chain.invoke({
        "question": request.question
    })

    return ChatResponse(answer=response)