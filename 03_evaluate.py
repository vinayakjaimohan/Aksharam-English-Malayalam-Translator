import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

def generate_translation(model, tokenizer, text, device, max_length=64):
    input_text = "translate English to Malayalam: " + text

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    model_path = "./aksharam_mt5/checkpoint-7500"  # change if needed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained(model_path)

    model.to(device)
    model.eval()

    print("\n" + "="*50)
    print("Aksharam mT5 Translator Ready 🚀")
    print("Type 'exit' to quit")
    print("="*50 + "\n")

    while True:
        try:
            text = input("Enter English sentence: ")

            if text.lower() in ["exit", "quit", "q"]:
                break

            if not text.strip():
                continue

            print("Translating...")
            output = generate_translation(model, tokenizer, text, device)

            print("Malayalam:", output)

            # 🔥 SAVE TO FILE
            with open("translations.txt", "a", encoding="utf-8") as f:
                f.write(text + " -> " + output + "\n")

        except KeyboardInterrupt:
            break

    print("\nExited.")


if __name__ == "__main__":
    main()