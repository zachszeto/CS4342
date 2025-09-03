import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
try:
    response = requests.post(
        'https://api.perplexity.ai/chat/completions',
        headers={
            'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'sonar-pro',
            'messages': [
                {
                    'role': 'user',
                    'content': "What are the major AI developments and announcements from today across the tech industry?"
                }
            ]
        }
    )
except Exception as e:
    print(f"An unexpected error occurred: {e}")

else:   
    data = json.dumps(response.json(), indent=4)
    print(data)