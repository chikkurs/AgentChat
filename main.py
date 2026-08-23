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
    version="2.0.1",
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
#
# IMPORTANT: stop_sequences is the key fix here. Without it, the model
# has no signal to stop after answering, and — because it has seen huge
# volumes of chat-log-style text in training (WhatsApp exports, support
# transcripts, etc.) — it will keep generating tokens up to
# max_new_tokens and often hallucinates a *fake continuation* of the
# conversation: a new timestamp, a fake user id, a fake reply from
# itself. That hallucinated block was previously getting saved into
# memory too, which made the problem compound over time.

STOP_SEQUENCES = [
    "\nUser:",
    "\nUser ",
    "\nHuman:",
    "\nAssistant:",
    "\nCurrent user question:",
    "\nPrevious conversation:",
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

CHAT_TEMPLATE_WITH_HISTORY = """You are a helpful AI assistant.

Below is the recent conversation history. Use it only if it is
relevant to the current question; otherwise ignore it.

Conversation history:
{history}

Current user question:
{question}

Assistant reply:"""

CHAT_TEMPLATE_NO_HISTORY = """You are a helpful AI assistant.

Answer the user's message naturally and directly.

User message:
{question}

Assistant reply:"""

with_history_prompt = PromptTemplate(
    input_variables=["history", "question"],
    template=CHAT_TEMPLATE_WITH_HISTORY,
)

no_history_prompt = PromptTemplate(
    input_variables=["question"],
    template=CHAT_TEMPLATE_NO_HISTORY,
)

with_history_chain = (
    with_history_prompt
    | text_model
    | StrOutputParser()
)

no_history_chain = (
    no_history_prompt
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


def clean_model_output(text: str) -> str:
    """
    Defensive cleanup in case the model still slips a hallucinated
    continuation past the stop sequences (e.g. if it generates a
    variant spelling/spacing the stop list doesn't cover).

    This trims the response at the first sign of a fabricated new
    turn, timestamp, or role label, so garbage never reaches the user
    or gets written into memory.
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

    text = text[:earliest_cut].strip()

    # Strip stray meta-commentary sentences about memory/context if the
    # model ignores the instruction not to produce them. This is a
    # best-effort net, not a guarantee — the prompt instruction is the
    # primary defense.
    meta_commentary_patterns = [
        r"(?i)^it seems like you'?re (starting a new conversation|referring to a previous message)[^.]*\.\s*",
        r"(?i)^i don'?t have (any )?(context|previous conversation)[^.]*\.\s*",
        r"(?i)^there'?s no previous conversation[^.]*\.\s*",
    ]

    for pattern in meta_commentary_patterns:
        text = re.sub(pattern, "", text).strip()

    return text


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

    # Only include a "history" section in the prompt when real prior
    # messages actually exist. A small 8B instruct model cannot be
    # reliably told (via instructions alone) to stay silent about an
    # empty/placeholder history — it tends to narrate it anyway
    # ("I don't have any context..."). Structurally omitting the
    # section on the first message removes the cue entirely, which is
    # far more reliable than prompting around it.
    if history_rows:
        history_text = format_chat_history(history_rows)

        response = with_history_chain.invoke(
            {
                "history": history_text,
                "question": question,
            }
        )
    else:
        response = no_history_chain.invoke(
            {
                "question": question,
            }
        )

    response = clean_model_output(str(response).strip())

    if not response:
        response = "I could not generate a response."

    # Only ever save the sanitized response — never the raw model
    # output — so hallucinated turns can't leak into future context.
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

    return clean_model_output(str(answer).strip())


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
        "version": "2.0.1",
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
