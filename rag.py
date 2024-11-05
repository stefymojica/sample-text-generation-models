import ollama

response = ollama.chat(
    model="phi3.5:latest",
    messages=[
        {
            "role": "user",
            "content": "explicame que es javascript",
        },
    ],
)
print(response["message"]["content"])
