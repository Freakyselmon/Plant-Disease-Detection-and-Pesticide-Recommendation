# app/llama_utils.py

import ollama

def get_disease_info_llama_local(disease_name):
    """
    Uses LLaMA 3 8B (via Ollama) to get a brief summary and treatment for a given plant disease.

    Args:
        disease_name (str): The name of the disease.
    
    Returns:
        str: Description and treatment from LLaMA 3.
    """
    prompt = f"Give a short description and treatment for the plant disease: {disease_name}."
    
    try:
        response = ollama.chat(model='llama3', messages=[
            {"role": "user", "content": prompt}
        ])
        return response['message']['content']
    except Exception as e:
        return f"❌ Failed to get response from LLaMA 3: {e}"
