import os
from openai import OpenAI

client = OpenAI(
  base_url="https://api.modular.com/v1",
  api_key=os.environ["MODULAR_API_KEY"],
)

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {
            "role": "user",
            "content": "Tell me a joke",
        },
    ],
    stream=False,
)

print(response.choices[0].message.content)
