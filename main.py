import os
import re
import base64
import shutil
import uuid
import uvicorn
import requests

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from groq import Groq


# =====================================================
# ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_TOKEN not found in environment variables")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment variables")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not found in environment variables")


# =====================================================
# APPLICATION
# =====================================================

app = FastAPI(
    title="Multimodal AI Agent",
    description=(
        "Stateless text and image chatbot with Telegram integration. "
        "No conversation memory — each message is handled independently."
    ),
    version="4.0.0",
)


# =====================================================
# CONFIGURATION
# =====================================================

UPLOAD_DIR = "uploads"
TELEGRAM_MESSAGE_LIMIT = 4000

os.makedirs(UPLOAD_DIR, exist_ok=True)


# =====================================================
# TEXT MODEL (HuggingFace)
# =====================================================
#
# stop_sequences prevents the model from generating a hallucinated
# continuation of the conversation (a fake new turn, timestamp, or
# role label) once it has finished its actual answer.

STOP_SEQUENCES = [
    "\nUser:",
    "\nUser ",
    "\nHuman:",
    "\nAssistant:",
    "\n[",
]

text_model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=700,
        temperature=0.3,
        stop_sequences=STOP_SEQUENCES,
    )
)

chat_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful AI assistant.

Reply naturally and directly to the user's message. Do not invent
further turns, fake timestamps, or fake usernames of your own. Answer
once, then stop.

User message:
{question}

Assistant reply:""",
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
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# =====================================================
# GENERAL HELPERS
# =====================================================

def remove_file(file_path: str | None) -> None:
    """
    Delete a temporary file safely.
    """

    if not file_path:
        return

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as exc:
        print(f"Unable to delete temporary file {file_path}: {exc}")


def get_safe_upload_path(filename: str | None) -> str:
    """
    Generate a unique and safe path for an uploaded file.
    """

    original_name = os.path.basename(filename or "uploaded_image.jpg")
    extension = os.path.splitext(original_name)[1].lower()

    if not extension:
        extension = ".jpg"

    unique_filename = f"{uuid.uuid4().hex}{extension}"

    return os.path.join(
        UPLOAD_DIR,
        unique_filename,
    )


def get_image_mime_type(image_path: str) -> str:
    """
    Determine the MIME type from the image extension.
    """

    extension = os.path.splitext(image_path)[1].lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    return mime_types.get(extension, "image/jpeg")


def clean_model_output(text: str) -> str:
    """
    Defensive cleanup in case the model still generates a
    hallucinated continuation (fake new turn/timestamp/role label)
    despite the stop sequences / system prompt.
    """

    if not text:
        return text

    cut_patterns = [
        r"\n\[\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",  # "[23-08-2026 ..." log style
        r"\n\s*User\s*[:\-]",
        r"\n\s*Human\s*[:\-]",
        r"\n\s*Assistant\s*[:\-]",
        r"\n\s*AgentChat\s*[:\-]",
        r"\n\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\s*:",  # fake IP-style sender
    ]

    earliest_cut = len(text)

    for pattern in cut_patterns:
        match = re.search(pattern, text)
        if match and match.start() < earliest_cut:
            earliest_cut = match.start()

    return text[:earliest_cut].strip()


# =====================================================
# TEXT CHAT (stateless, no memory)
# =====================================================

def ask_text_model(question: str) -> str:
    """
    Call the text model with a single message. No history is loaded
    or stored — every call is independent.
    """

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty")

    response = text_chain.invoke(
        {
            "question": question,
        }
    )

    response = clean_model_output(str(response).strip())

    if not response:
        response = "I could not generate a response."

    return response


# =====================================================
# IMAGE CHAT (stateless, no memory)
# =====================================================

def ask_vision_model(
    image_path: str,
    question: str,
) -> str:
    """
    Send an image and question to the Groq vision model.
    """

    question = question.strip()

    if not question:
        question = "Explain this image."

    mime_type = get_image_mime_type(image_path)

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    response = vision_client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};"
                                f"base64,{encoded_image}"
                            )
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        return "I could not analyze the image."

    return clean_model_output(str(answer).strip())


# =====================================================
# ROOT AND HEALTH
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Multimodal AI Agent is running",
        "version": "4.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# =====================================================
# CHAT API
# =====================================================

@app.post("/chat")
async def chat(
    question: str | None = Form(None),
    image: UploadFile | None = File(None),
):
    """
    Handle text-only and image-based chat requests.

    Stateless: no conversation memory is loaded or stored.
    """

    question = question.strip() if question else None

    if not question and not image:
        raise HTTPException(
            status_code=400,
            detail="Provide text or an image",
        )

    image_path = None

    try:
        if image:
            if not image.content_type or not image.content_type.startswith(
                "image/"
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file must be an image",
                )

            image_path = get_safe_upload_path(
                image.filename
            )

            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(
                    image.file,
                    buffer,
                )

            vision_question = question or (
                "Extract all visible text from this image. "
                "If it is a document, summarize it. "
                "If it contains a table, preserve the table structure."
            )

            answer = ask_vision_model(
                image_path=image_path,
                question=vision_question,
            )

            return JSONResponse(
                content={
                    "status": "success",
                    "type": "vision",
                    "answer": answer,
                }
            )

        answer = ask_text_model(question or "")

        return JSONResponse(
            content={
                "status": "success",
                "type": "text",
                "answer": answer,
            }
        )

    except HTTPException:
        raise

    except Exception as exc:
        print("Chat Error:", str(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Unable to process the request: {str(exc)}",
        ) from exc

    finally:
        if image:
            await image.close()

        remove_file(image_path)


# =====================================================
# TELEGRAM HELPERS
# =====================================================

def send_telegram_message(
    chat_id: int,
    text: str,
) -> None:
    """
    Send a message to Telegram.

    Telegram messages are split to avoid exceeding the message limit.
    """

    text = str(text).strip()

    if not text:
        text = "I could not generate a response."

    chunks = [
        text[index:index + TELEGRAM_MESSAGE_LIMIT]
        for index in range(
            0,
            len(text),
            TELEGRAM_MESSAGE_LIMIT,
        )
    ]

    for chunk in chunks:
        response = requests.post(
            (
                f"https://api.telegram.org/"
                f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            ),
            json={
                "chat_id": chat_id,
                "text": chunk,
            },
            timeout=30,
        )

        response.raise_for_status()


def download_telegram_file(file_id: str) -> str:
    """
    Download a Telegram image to the local uploads directory.
    """

    file_info_response = requests.get(
        (
            f"https://api.telegram.org/"
            f"bot{TELEGRAM_BOT_TOKEN}/getFile"
        ),
        params={
            "file_id": file_id,
        },
        timeout=30,
    )

    file_info_response.raise_for_status()
    file_info = file_info_response.json()

    if not file_info.get("ok"):
        raise RuntimeError(
            file_info.get(
                "description",
                "Unable to get Telegram file details",
            )
        )

    telegram_file_path = file_info["result"]["file_path"]

    download_url = (
        f"https://api.telegram.org/file/"
        f"bot{TELEGRAM_BOT_TOKEN}/"
        f"{telegram_file_path}"
    )

    download_response = requests.get(
        download_url,
        timeout=60,
    )

    download_response.raise_for_status()

    local_path = get_safe_upload_path(
        os.path.basename(telegram_file_path)
    )

    with open(local_path, "wb") as file:
        file.write(download_response.content)

    return local_path


# =====================================================
# TELEGRAM WEBHOOK
# =====================================================

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """
    Handle Telegram text messages and image messages.

    Stateless: no conversation memory is loaded or stored.
    """

    image_path = None

    try:
        data = await request.json()

        message = data.get("message")

        if not message:
            return {
                "status": "ignored",
                "reason": "No message found",
            }

        chat = message.get("chat", {})
        chat_id = chat.get("id")

        if chat_id is None:
            return {
                "status": "ignored",
                "reason": "No chat ID found",
            }

        # =============================================
        # TEXT MESSAGE
        # =============================================

        if "text" in message:
            user_text = message["text"].strip()

            if not user_text:
                send_telegram_message(
                    chat_id,
                    "Please enter a message.",
                )

                return {
                    "status": "ignored",
                }

            if user_text.lower() in {
                "/start",
                "/help",
            }:
                send_telegram_message(
                    chat_id,
                    (
                        "Hello! Send me a text message or an image.\n\n"
                        "Each message is answered independently — "
                        "I don't remember previous messages."
                    ),
                )

                return {
                    "status": "success",
                }

            answer = ask_text_model(user_text)

            send_telegram_message(
                chat_id,
                answer,
            )

            return {
                "status": "success",
                "type": "text",
            }

        # =============================================
        # IMAGE MESSAGE
        # =============================================

        if "photo" in message:
            photo_list = message["photo"]

            if not photo_list:
                send_telegram_message(
                    chat_id,
                    "The image could not be read.",
                )

                return {
                    "status": "error",
                }

            largest_photo = photo_list[-1]
            file_id = largest_photo["file_id"]

            image_path = download_telegram_file(
                file_id
            )

            caption = message.get(
                "caption",
                (
                    "Extract all visible text from this image "
                    "and explain the image."
                ),
            ).strip()

            answer = ask_vision_model(
                image_path=image_path,
                question=caption,
            )

            send_telegram_message(
                chat_id,
                answer,
            )

            return {
                "status": "success",
                "type": "vision",
            }

        send_telegram_message(
            chat_id,
            (
                "Unsupported message type. "
                "Please send text or an image."
            ),
        )

        return {
            "status": "ignored",
        }

    except requests.RequestException as exc:
        print("Telegram API Error:", str(exc))

        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "message": "Telegram API request failed",
            },
        )

    except Exception as exc:
        print("Telegram Webhook Error:", str(exc))

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(exc),
            },
        )

    finally:
        remove_file(image_path)


# =====================================================
# LOCAL DEVELOPMENT
# =====================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )