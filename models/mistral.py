from huggingface_hub import login
login()

import os
os.environ["HUGGINGFACE_TOKEN"] = "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"

from transformers import pipeline

messages = [
    {"role": "system", "content": "tu eres un asesor experto en tecnologia, siempre habla español"},
    {"role": "user", "content": "tu quien eres?"},
]
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
chatbot(messages)
