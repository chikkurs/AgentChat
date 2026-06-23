import os
import base64
import shutil
import uvicorn

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException
)
import requests
from fastapi import Request
from fastapi.responses import JSONResponse

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from groq import Groq

# =====================================================
# ENV
# =====================================================

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not HF_TOKEN:
    raise Exception("HUGGINGFACE_TOKEN not found")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY not found")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    raise Exception("TELEGRAM_BOT_TOKEN not found")
# =====================================================
# APP
# =====================================================

app = FastAPI(
    title="Multimodal AI Agent",
    version="1.0"
)

# =====================================================
# UPLOAD DIRECTORY
# =====================================================

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

# =====================================================
# TEXT MODEL
# =====================================================

text_model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=700
    )
)

chat_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant.

Question:
{question}
"""
)

text_chain = (
    chat_prompt
    | text_model
    | StrOutputParser()
)

# =====================================================
# VISION MODEL (Groq)
# =====================================================

vision_client = Groq(api_key=GROQ_API_KEY)

# =====================================================
# TEXT CHAT
# =====================================================

def ask_text_model(question: str):

    response = text_chain.invoke({
        "question": question
    })

    return response

# =====================================================
# IMAGE CHAT
# =====================================================

def ask_vision_model(
    image_path: str,
    question: str
):

    try:

        with open(image_path, "rb") as img:

            encoded_image = base64.b64encode(
                img.read()
            ).decode("utf-8")

        response = vision_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Vision Model Error: {str(e)}"

# =====================================================
# ROOT
# =====================================================

@app.get("/")
def home():

    return {
        "message": "AI Agent Running"
    }

# =====================================================
# CHAT ENDPOINT
# =====================================================

@app.post("/chat")
async def chat(
    question: str = Form(None),
    image: UploadFile = File(None)
):

    if not question and not image:

        raise HTTPException(
            status_code=400,
            detail="Provide text or image"
        )

    # ==========================================
    # IMAGE FLOW
    # ==========================================

    if image:

        image_path = os.path.join(
            UPLOAD_DIR,
            image.filename
        )

        with open(
            image_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                image.file,
                buffer
            )

        answer = ask_vision_model(
            image_path=image_path,
            question=question or """
Extract all text from this image.
If it is a document, summarize it.
If it contains tables, preserve them.
"""
        )

        return JSONResponse(
            content={
                "type": "vision",
                "answer": answer
            }
        )

    # ==========================================
    # TEXT FLOW
    # ==========================================

    answer = ask_text_model(
        question
    )

    return JSONResponse(
        content={
            "type": "text",
            "answer": answer
        }
    )
@app.get("/health")
def health():
    return {"status": "deployment-test"}
def send_telegram_message(chat_id: int, text: str):

    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text
        },
        timeout=30
    )


def download_telegram_file(file_id: str):

    file_info = requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile",
        params={"file_id": file_id},
        timeout=30
    ).json()

    file_path = file_info["result"]["file_path"]

    download_url = (
        f"https://api.telegram.org/file/bot"
        f"{TELEGRAM_BOT_TOKEN}/{file_path}"
    )

    local_path = os.path.join(
        UPLOAD_DIR,
        os.path.basename(file_path)
    )

    response = requests.get(
        download_url,
        timeout=60
    )

    with open(local_path, "wb") as f:
        f.write(response.content)

    return local_path


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):

    try:

        data = await request.json()

        if "message" not in data:
            return {"status": "ignored"}

        message = data["message"]

        chat_id = message["chat"]["id"]

        # ===============================
        # TEXT MESSAGE
        # ===============================

        if "text" in message:

            user_text = message["text"]

            answer = ask_text_model(
                user_text
            )

            send_telegram_message(
                chat_id,
                answer
            )

            return {"status": "success"}

        # ===============================
        # IMAGE MESSAGE
        # ===============================

        if "photo" in message:

            photo_list = message["photo"]

            largest_photo = photo_list[-1]

            file_id = largest_photo["file_id"]

            image_path = download_telegram_file(
                file_id
            )

            caption = message.get(
                "caption",
                "Extract all text and explain the image."
            )

            answer = ask_vision_model(
                image_path=image_path,
                question=caption
            )

            send_telegram_message(
                chat_id,
                answer
            )

            return {"status": "success"}

        send_telegram_message(
            chat_id,
            "Unsupported message type."
        )

        return {"status": "ignored"}
        

    except Exception as e:

        print("Telegram Error:", str(e))

        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )