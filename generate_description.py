import ollama


def generate_description(image_path):
    """Ollamaにマスクを無視して画像の説明をさせる"""
    prompt = "Please describe the input image, ignoring the black fill in the center."
    response = ollama.chat(
        model="llama3.2-vision",
        messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
    )
    return response["message"]["content"]
