
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

model_id = "CohereForAI/aya-23-35B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')


def get_translations(all_prompts, batch_size=32):
    results = []
    for i in tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[i:i + batch_size]
        messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True)

        gen_tokens = model.generate(
            input_ids.to('cuda'),
            max_new_tokens=512,  # Adjust based on expected output length
            do_sample=False,  # Prioritize deterministic outputs for fidelity
            temperature=0.3,  # Low randomness for precise translations
            top_k=None,  # Not needed when do_sample=False
            top_p=None,  # Not needed when do_sample=False
            repetition_penalty=1.1,  # Penalize repetitions if necessary
        )

        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        results.extend(gen_texts)

    for k in range(len(all_prompts[:2])):
        print(all_prompts[k])
        print(results[k])
        print("*" * 100)
    
    return results
