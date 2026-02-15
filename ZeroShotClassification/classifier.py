import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
# client initialization
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_gemini_2_5_pro(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    return response.text

def call_gemini_3_pro(prompt):
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt
    )
    return response.text

def call_gemini_3_flash(prompt):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return response.text

def call_gemini_2_5_flash(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt
    )
    return response.text

import time
import random

def call_llm(promt, api, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            if api == "gemini-2.5-pro":
                return call_gemini_2_5_pro(promt)
            elif api == "gemini-2.5-flash":
                return call_gemini_2_5_flash(promt)
            elif api == "gemini-3-pro-preview":
                return call_gemini_3_pro(promt)
            elif api == "gemini-3-flash":
                return call_gemini_3_flash(promt)
            else:
                 raise ValueError(f"Unknown API model: {api}")
        except Exception as e:
            retries += 1
            print(f"Error calling LLM (Attempt {retries}/{max_retries}): {e}")
            if retries >= max_retries:
                raise e
            time.sleep(1 + random.random()) # Simple backoff

