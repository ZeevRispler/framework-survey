"""
Simple LLM Prompting Example
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()

def prompt_example():
    # Initialize OpenAI client
    client = OpenAI()

    # Basic prompting
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Explain Python in one sentence"}
        ]
    )

    print("Basic prompt response:")
    print(response.choices[0].message.content)
    print()

    # With system prompt
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant"},
            {"role": "user", "content": "How do I create a for loop in Python?"}
        ]
    )

    print("With system prompt:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    prompt_example()