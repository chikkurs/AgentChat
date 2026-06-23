import requests
import time
import os 
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

offset = 0

while True:

    updates = requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
        params={"offset": offset + 1},
        timeout=30
    ).json()

    for update in updates.get("result", []):

        offset = update["update_id"]

        if "message" not in update:
            continue

        message = update["message"]

        chat_id = message["chat"]["id"]

        text = message.get("text")

        if not text:
            continue

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            data={
                "question": text
            }
        )

        answer = response.json()["answer"]

        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": answer[:4000]
            }
        )

    time.sleep(1)