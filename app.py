from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ollama

class SimpleRag:
    def __init__(self):
        model_name = 'all-MiniLM-L6-v2'
        self.model_embedding = SentenceTransformer(model_name)
        self.documents = []
        self.embedding = None
    
    def add_documents(self,documents):
        self.documents = documents  
        self.embedding = self.model_embedding.encode(documents)  
    
        self.documents.extend(self.embedding)
        
    def query(self,question,k=2):
        question_embedding = self.model_embedding.encode(question)
        similarities = cosine_similarity([question_embedding], self.embedding)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        result = []
        for i in top_k_indices:
            result.append({
                'document': self.documents[i], 
                'score':similarities[i]})
        return result

    def text_generation(self,question):
        retrieved_docs = self.query(question)
        context = "\n".join([doc['document'] for doc in retrieved_docs])
        response = ""
        stream = ollama.chat(
                model="phi3.5:latest",
                messages=[{
                    "role": "system",
                    "content": "Eres un experto vendedor que proporciona recomendaciones precisas y detalladas, con un maximo de 100 palabras"
                },
                {
                    "role": "user",
                    "content": context
                }],
                stream=True)

        for chunk in stream:
            chunk_content = chunk["message"]["content"]
            response += chunk_content
            print(chunk_content,end="",flush=True)
        return response
    
def main():
    documents = [
        "iPhone 14 Pro: Smartphone de última generación con pantalla de 6.1 pulgadas, cámara de 48MP y chip A16 Bionic. Precio: $999",
        "Samsung Galaxy S23: Teléfono Android con pantalla AMOLED de 6.1 pulgadas, cámara de 50MP y procesador Snapdragon 8. Precio: $799",
        "MacBook Air M2: computador ultradelgada con chip M2, 8GB RAM, 256GB SSD y pantalla Retina de 13.6 pulgadas. Precio: $1199",
        "Sony WH-1000XM5: Auriculares inalámbricos con cancelación de ruido, 30 horas de batería y audio de alta resolución. Precio: $399",
        "Nintendo Switch OLED: Consola de videojuegos con pantalla OLED de 7 pulgadas, 64GB de almacenamiento y modo portátil. Precio: $349",
        "iPad Air 2022: Tablet con chip M1, pantalla de 10.9 pulgadas, compatible con Apple Pencil y Magic Keyboard. Precio: $599",
        "Dell XPS 15: computador para profesionales con Intel i7, 16GB RAM, 512GB SSD y pantalla 4K de 15.6 pulgadas. Precio: $1799",
        "AirPods Pro 2: Auriculares TWS con cancelación activa de ruido, audio espacial y resistencia al agua IPX4. Precio: $249",
        "PlayStation 5: Consola de nueva generación con SSD ultrarrápido, ray tracing y control DualSense. Precio: $499",
        "Samsung QN90B: Smart TV QLED 4K de 65 pulgadas con tasa de refresco de 120Hz y HDR. Precio: $1999"
    ]
    
    print("Inicializar Rag ...")
    rag = SimpleRag()
    
    print("Agregando documentos")
    rag.add_documents(documents)
    
    questions = [
        "recomiendame un computador con buenas especificaciones"
    ]

    for question in questions:
        print(f"\nPregunta: {question}")

        retrieved_docs = rag.query(question)
        print("\nDocumentos relevantes:")
        for doc in retrieved_docs:
            print(f"'documents':{doc}")

            result = rag.text_generation(question)
            print(f"\nRespuesta generada: {result}")

if __name__ == "__main__":
    main()
