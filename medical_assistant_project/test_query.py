import sys
import os

# Ensure django setup or just python path is correct
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_engine.ollama_client import generate_stream


def test():
    question = "What are the common symptoms of a migraine?"
    context = "Migraines are characterized by severe throbbing pain, usually on one side of the head, accompanied by nausea, vomiting, and extreme sensitivity to light and sound."

    print(f"Question: {question}\n")
    print("AI Response: ", end="", flush=True)

    for chunk in generate_stream(question, context):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    test()
