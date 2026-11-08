import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utilities import get_device


def main():
    model_id = "Qwen/Qwen3-0.6B-Base"
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Qwen's checkpoint is stored in bfloat16. Use float16 on MPS for broader
    # Apple GPU support; on other devices, let Transformers use the checkpoint dtype.
    model_kwargs = {"torch_dtype": torch.float16} if device == "mps" else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, **model_kwargs
    ).to(device)

    inputs = tokenizer(
        "The process of making tea begins with", return_tensors="pt"
    ).to(device)
    outputs = model.generate(**inputs, max_new_tokens=25)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
