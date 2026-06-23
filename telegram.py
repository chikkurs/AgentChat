import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise Exception("TELEGRAM_BOT_TOKEN not found")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
UPLOAD_DIR = "telegram_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

offset = 0


def send_message(chat_id, text):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text[:4000]
        },
        timeout=30
    )


def download_telegram_file(file_id):
    file_info = requests.get(
        f"{TELEGRAM_API}/getFile",
        params={"file_id": file_id},
        timeout=30
    ).json()

    file_path = file_info["result"]["file_path"]
    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

    local_path = os.path.join(
        UPLOAD_DIR,
        os.path.basename(file_path)
    )

    file_response = requests.get(file_url, timeout=60)

    with open(local_path, "wb") as f:
        f.write(file_response.content)

    return local_path


while True:
    try:
        updates = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params={"offset": offset + 1},
            timeout=30
        ).json()

        for update in updates.get("result", []):
            offset = update["update_id"]

            if "message" not in update:
                continue

            message = update["message"]
            chat_id = message["chat"]["id"]

            # TEXT
            if "text" in message:
                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    data={"question": message["text"]},
                    timeout=180
                )

            # IMAGE
            elif "photo" in message:
                send_message(chat_id, "Image received. Processing...")

                photo = message["photo"][-1]
                image_path = download_telegram_file(photo["file_id"])

                question = message.get(
                    "caption",
                    "Extract all text from this image and explain it."
                )

                with open(image_path, "rb") as img:
                    response = requests.post(
                        "http://127.0.0.1:8000/chat",
                        data={"question": question},
                        files={"image": img},
                        timeout=180
                    )

            else:
                send_message(chat_id, "Unsupported message type.")
                continue

            print("API status:", response.status_code)
            print("API response:", response.text)

            answer = response.json().get("answer", "No answer found.")
            send_message(chat_id, answer)

    except Exception as e:
        print("Bot error:", str(e))

    time.sleep(1)