import requests
import json
import argparse

def read_file(filename):
    """Read content from a file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise Exception(f"Error: {filename} not found")
    except IOError as e:
        raise Exception(f"Error reading {filename}: {str(e)}")

def save_output(content, filename):
    """Save content to a file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    except IOError as e:
        raise Exception(f"Error saving to {filename}: {str(e)}")

def process_with_llm(api_url, api_key, user_prompt, user_text):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "user",
            "content": f"{user_prompt}\n\n{user_text}"
        }],
        "temperature": 0.7,
        "max_tokens": 500,
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    try:
        # Read input files
        prompt = read_file('prompt.md')  # Note: Fixed typo from 'prompt.md' to 'prompt.md'
        content = read_file('content.md')
        
        # Process with LLM
        result = process_with_llm(
            config['api_url'],
            config['api_key'],
            prompt,
            content
        )
        
        # Save and display results
        save_output(result, 'output.md')
        print("AI Response:\n", result)
        print("\nOutput successfully saved to output.md")
        
    except Exception as e:
        print(str(e))