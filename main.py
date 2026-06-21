from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse
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




VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")


@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return hub_challenge

    return "Verification failed"


@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    print(body)
    return {"status": "ok"}