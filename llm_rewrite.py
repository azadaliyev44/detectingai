import requests
import pandas as pd
import json
import time
from tqdm import tqdm

API_KEY = "AIzaSyBy1DHdAhq4qJfSUOED3oT-GInwKw9Her4"

MODEL = "gemini-1.5-flash-002"

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"


input_file = "xsum-gpt3-5.json" 
output_json = "revised_texts.json"  
output_human_txt = "revised_human_texts.txt"  
output_llm_txt = "revised_llm_texts.txt" 

print("Loading dataset...")

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

if isinstance(data, dict):
    data = [data]

df = pd.DataFrame(data)

# Check text, label coulmns
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must have text and label columns.")

print(f"Dataset loaded, total items: {len(df)}")

# Function for asking LLm to rewrite
def revise_text(text, index):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": f"Revise the following text and provide only the revised text, no additional commentary or introductions:\n\n{text}"}]
        }]
    }
    # To retry until valid response, against quota limit
    while True: 
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            response_data = response.json()

            # To check the api response format to make sure
            if index < 3:
                print(f"API Response index {index}:\n{json.dumps(response_data, indent=4)}")

            # Check for errors in response
            if "error" in response_data:
                error_message = response_data["error"].get("message", "Unknown error")
                print(f"API Error: {error_message}. Retrying in 1 second")
                time.sleep(1)
                continue  

            # Extracting text from response
            candidates = response_data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"]

            print(f"No text found in {index}. Retrying in 1 second")
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}. Retrying in 1 second")
            time.sleep(1)

# Send query
print("Processing texts")
df["revised_text"] = [revise_text(text, idx) for idx, text in tqdm(enumerate(df["text"]), total=len(df), desc="Rewriting texts")]

# Save as JSON
df.to_json(output_json, orient="records", indent=4, force_ascii=False)
print(f"texts saved to: {output_json}")

# Separate according to labels
print("SAving txt files")

with open(output_human_txt, "w", encoding="utf-8") as human_file, open(output_llm_txt, "w", encoding="utf-8") as llm_file:
    for idx, row in df.iterrows():
        formatted_text = json.dumps({str(idx): row["revised_text"]}, ensure_ascii=False)
        
        if row["label"] == "human":
            human_file.write(formatted_text + "\n")
        elif row["label"] == "llm":
            llm_file.write(formatted_text + "\n")

print(f"Modified Human texts saved to: {output_human_txt}")
print(f"Modified LLM texts saved to: {output_llm_txt}")
