import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_api():
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("ERROR: No Groq API key found in environment variables")
        return False
    
    # Groq API endpoint for model listing
    verify_url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Make a test request
        response = requests.get(verify_url, headers=headers, timeout=10)
        
        # Check response
        if response.status_code == 200:
            print("✅ Groq API connection successful!")
            print("Available Models:")
            models = response.json().get('data', [])
            for model in models:
                print(f"- {model.get('id', 'Unknown Model')}")
            return True
        else:
            print(f"❌ API key verification failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_groq_api()
