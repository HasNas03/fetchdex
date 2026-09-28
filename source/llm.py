from ollama import chat


MODEL_NAME = "qwen3:1.7b"


def generate_response(prompt: str) -> str:
    """
    Send a prompt to our local Ollama model
    and return its response.
    """

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


if __name__ == "__main__":

    prompt = (
        "Explain FastAPI in one short sentence."
    )

    print("PROMPT:")
    print(prompt)

    print("\nRESPONSE:")

    response = generate_response(prompt)

    print(response)