import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model

def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

def sample(logits, temperature, k):
    if temperature != 0:
        logits /= temperature

        top_k = torch.topk(logits, k=k, dim=-1)
        top_k_probs = torch.softmax(top_k.values, dim=-1)
        next_token = top_k.indices[torch.multinomial(top_k_probs, num_samples=1).item()]
        
        return next_token.item()
    else:
        return torch.argmax(logits, dim=-1).item()

def generate(model, tokenizer, prompt, temperature, k, max_tokens, use_cache=True):
    input = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        if use_cache:
            current_input = {}
            for i in range(max_tokens):
                if i == 0:
                    output = model(**input, use_cache=True)
                else:
                    output = model(**current_input, past_key_values=output.past_key_values, use_cache=True)

                logits = output.logits[:, -1, :].squeeze(0)
                next_token = sample(logits, temperature, k)

                if next_token == tokenizer.eos_token_id:
                    break

                current_input['input_ids'] = torch.tensor([[next_token]])
                current_input['attention_mask'] = torch.ones((1, 1), dtype=input['attention_mask'].dtype)

                input['input_ids'] = torch.cat([input['input_ids'], current_input['input_ids']], dim=1)
                input['attention_mask'] = torch.cat([input['attention_mask'], current_input['attention_mask']], dim=1)
        else:
            for _ in range(max_tokens):
                output = model(**input)

                logits = output.logits[:, -1, :].squeeze(0)
                next_token = sample(logits, temperature, k)

                if next_token == tokenizer.eos_token_id:
                    break

                input['input_ids'] = torch.cat([input['input_ids'], torch.tensor([[next_token]])], dim=1)
                input['attention_mask'] = torch.cat([input['attention_mask'], torch.ones((1, 1), dtype=input['attention_mask'].dtype)], dim=1)

    response = tokenizer.decode(input['input_ids'][0], skip_special_tokens=True)
    return response

parser = argparse.ArgumentParser(description="simple parser")

parser.add_argument("prompt")
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--max_tokens", type=int, default=50)

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen3.5-0.8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, device)

    args = parser.parse_args()

    prompt = args.prompt
    temperature = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens

    print(generate(model, tokenizer, prompt, temperature, top_k, max_tokens))
