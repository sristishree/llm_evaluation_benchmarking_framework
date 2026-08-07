import requests
import os

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"
    },
)

models = response.json()["data"]
model_ids = sorted(set([model["id"] for model in models]))
print(*model_ids, sep="\n")