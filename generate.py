import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model

def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

def sample(batch_logits, temperature, k):
    """
    Samples next tokens from a batch of logits [N, vocab_size].
    Returns next_tokens [N].
    """
    if temperature != 0:
        batch_logits /= temperature
        top_k = torch.topk(batch_logits, k=k, dim=-1)
        top_k_probs = torch.softmax(top_k.values, dim=-1)
        samples = torch.multinomial(top_k_probs, num_samples=1)
        # Shape [N]
        next_tokens = top_k.indices.gather(dim=1, index=samples).squeeze(-1)
        return next_tokens
    else:
        # Greedy decoding
        return torch.argmax(batch_logits, dim=-1)

def generate(model, tokenizer, prompts: list[str], temperature: float, k: int, max_tokens: int, use_cache=True) -> list[str]:
    """
    Batched autoregressive generation, handling single or multiple prompts.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)

    batch_size = len(prompts)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Track only the latest inputs when using cache
    current_inputs = inputs if use_cache else None

    with torch.no_grad():
        for step in range(max_tokens):
            if use_cache:
                # If cached, pass only the latest token (except on step 0)
                if step == 0:
                    output = model(**inputs, use_cache=True)
                else:
                    output = model(**current_inputs, past_key_values=output.past_key_values, use_cache=True)
            else:
                # Without cache, process the full growing sequence every step
                output = model(**inputs)

            # Get logits for the last token position
            batch_logits = output.logits[:, -1, :]
            next_tokens = sample(batch_logits, temperature, k)

            # Check which sequences just finished
            finished |= (next_tokens == tokenizer.eos_token_id)
            
            # If a sequence is finished, emit PAD token
            next_tokens = torch.where(finished, tokenizer.pad_token_id, next_tokens)

            # Update cache inputs (only the new token)
            if use_cache:
                new_attention = (~finished).to(inputs['attention_mask'].dtype).unsqueeze(-1)
                current_inputs = {
                    'input_ids': next_tokens.unsqueeze(-1),
                    'attention_mask': new_attention
                }

            # Append the new token/mask to the overall sequence
            inputs['input_ids'] = torch.cat([inputs['input_ids'], next_tokens.unsqueeze(-1)], dim=1)
            
            new_attention = (~finished).to(inputs['attention_mask'].dtype).unsqueeze(-1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], new_attention], dim=1)

            # Early stopping if all sequences are finished
            if finished.all():
                break

    return tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)


parser = argparse.ArgumentParser(description="simple parser")

parser.add_argument("prompt", nargs="+", help="One or more prompts separated by space")
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--max_tokens", type=int, default=50)

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen3.5-0.8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, device)

    args = parser.parse_args()

    temperature = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens

    prompts = args.prompt
    responses = generate(model, tokenizer, prompts, temperature, top_k, max_tokens, use_cache=True)
    
    for i, res in enumerate(responses):
        print(f"\n--- Output {i+1} ---")
        print(res)