import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utilities import get_device


def main():
    model_id = "openai-community/gpt2"
    device = get_device()
    print(f'Using device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    inputs = tokenizer(
        "The process of making tea begins with", return_tensors="pt"
    ).to(device)
    outputs = model.generate(**inputs, max_new_tokens=21)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
