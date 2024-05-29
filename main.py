from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"

def main():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    text = "<s>[INST] What is your favourite condiment? [/INST]"

    model_inputs = tokenizer.encode(text, return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=1000,
        do_sample=True,
    ).to(device)

    decoded = tokenizer.batch_decode(generated_id)
    print(decoded)

if __name__ == "__main__":
    main()