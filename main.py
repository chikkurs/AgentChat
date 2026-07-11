import os
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

from memory import (
    clear_chat_history,
    format_chat_history,
    get_chat_history,
    save_message,
)


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
        "Text and image chatbot with PostgreSQL conversation memory "
        "and Telegram integration."
    ),
    version="2.0.0",
)


# =====================================================
# CONFIGURATION
# =====================================================

UPLOAD_DIR = "uploads"
CHAT_HISTORY_LIMIT = 10
TELEGRAM_MESSAGE_LIMIT = 4000

os.makedirs(UPLOAD_DIR, exist_ok=True)


# =====================================================
# TEXT MODEL
# =====================================================

text_model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=700,
        temperature=0.3,
    )
)

chat_prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""
You are a helpful AI assistant.

Use the previous conversation only when it is relevant to the current
question.

Do not claim to remember information that is not present in the
conversation history.

Previous conversation:
{history}

Current user question:
{question}

Assistant:
""",
)

text_chain = (
    chat_prompt
    | text_model
    | StrOutputParser()
)


# =====================================================
# VISION MODEL
# =====================================================

vision_client = Groq(api_key=GROQ_API_KEY)


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


# =====================================================
# TEXT CHAT WITH MEMORY
# =====================================================

def ask_text_model(
    user_id: str,
    question: str,
) -> str:
    """
    Load chat history, call the text model and save the conversation.
    """

    user_id = str(user_id).strip()
    question = question.strip()

    if not user_id:
        raise ValueError("user_id cannot be empty")

    if not question:
        raise ValueError("Question cannot be empty")

    history_rows = get_chat_history(
        user_id=user_id,
        limit=CHAT_HISTORY_LIMIT,
    )

    history_text = format_chat_history(
        history_rows
    )

    response = text_chain.invoke(
        {
            "history": history_text,
            "question": question,
        }
    )

    response = str(response).strip()

    if not response:
        response = "I could not generate a response."

    save_message(
        user_id=user_id,
        role="user",
        message=question,
    )

    save_message(
        user_id=user_id,
        role="assistant",
        message=response,
    )

    return response


# =====================================================
# IMAGE CHAT
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
        model="meta-llama/llama-4-scout-17b-16e-instruct",
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

    return str(answer).strip()


def ask_vision_model_with_memory(
    user_id: str,
    image_path: str,
    question: str,
) -> str:
    """
    Analyze an image and store the caption and response in memory.
    """

    answer = ask_vision_model(
        image_path=image_path,
        question=question,
    )

    memory_message = (
        "[User uploaded an image]\n"
        f"Question or caption: {question}"
    )

    save_message(
        user_id=str(user_id),
        role="user",
        message=memory_message,
    )

    save_message(
        user_id=str(user_id),
        role="assistant",
        message=answer,
    )

    return answer


# =====================================================
# ROOT AND HEALTH
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Multimodal AI Agent is running",
        "version": "2.0.0",
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
    user_id: str = Form(...),
    question: str | None = Form(None),
    image: UploadFile | None = File(None),
):
    """
    Handle text-only and image-based chat requests.

    Every API client must provide a user_id so conversation memory
    remains separate for each user.
    """

    user_id = user_id.strip()
    question = question.strip() if question else None

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id cannot be empty",
        )

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

            answer = ask_vision_model_with_memory(
                user_id=user_id,
                image_path=image_path,
                question=vision_question,
            )

            return JSONResponse(
                content={
                    "status": "success",
                    "type": "vision",
                    "user_id": user_id,
                    "answer": answer,
                }
            )

        answer = ask_text_model(
            user_id=user_id,
            question=question or "",
        )

        return JSONResponse(
            content={
                "status": "success",
                "type": "text",
                "user_id": user_id,
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
# MEMORY API
# =====================================================

@app.delete("/memory/{user_id}")
def delete_memory(user_id: str):
    """
    Delete all stored conversation messages for one user.
    """

    user_id = user_id.strip()

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id cannot be empty",
        )

    try:
        deleted_count = clear_chat_history(
            user_id=user_id
        )

        return {
            "status": "success",
            "user_id": user_id,
            "deleted_messages": deleted_count,
        }

    except Exception as exc:
        print("Clear Memory Error:", str(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Unable to clear memory: {str(exc)}",
        ) from exc


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

        user_id = str(chat_id)

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
                        "Commands:\n"
                        "/clear - clear your conversation memory\n"
                        "/help - show this help message"
                    ),
                )

                return {
                    "status": "success",
                }

            if user_text.lower() == "/clear":
                deleted_count = clear_chat_history(
                    user_id=user_id
                )

                send_telegram_message(
                    chat_id,
                    (
                        "Conversation memory cleared successfully.\n"
                        f"Deleted messages: {deleted_count}"
                    ),
                )

                return {
                    "status": "success",
                    "deleted_messages": deleted_count,
                }

            answer = ask_text_model(
                user_id=user_id,
                question=user_text,
            )

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

            answer = ask_vision_model_with_memory(
                user_id=user_id,
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