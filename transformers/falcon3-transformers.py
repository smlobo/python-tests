import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utilities import get_device


def main():
    model_id = "tiiuae/Falcon3-1B-Base"
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Falcon3's checkpoint is stored in bfloat16. Use float16 on MPS for wider
    # Apple GPU support; elsewhere let Transformers use the checkpoint dtype.
    model_kwargs = {"dtype": torch.float16} if device == "mps" else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, **model_kwargs
    ).to(device)

    inputs = tokenizer(
        "The process of making tea begins with", return_tensors="pt"
    ).to(device)
    outputs = model.generate(**inputs, max_new_tokens=32)
    print(tokenizer.decode(outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ))


if __name__ == "__main__":
    main()
