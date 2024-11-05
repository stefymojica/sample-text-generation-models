import ollama

stream = ollama.chat(
    model="phi3.5:latest",
    messages=[{"role": "user", "content": "Explicame que es javascript"}],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
