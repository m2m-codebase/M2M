import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

model_id = "CohereForAI/aya-23-35B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')


def get_translations(all_prompts, all_save_paths, batch_size=32):
    results = []
    idxs = list(range(0, len(all_prompts), batch_size))
    for i in tqdm(idxs, desc=f"Processing Batch"):
        batch_prompts = all_prompts[i:i + batch_size]
        save_paths = all_save_paths[i:i + batch_size]
       
        assert len(batch_prompts) == len(save_paths), "save paths and batches len dont match"

        if all(os.path.exists(p) for p in save_paths):
            #print(i, save_paths)
            continue

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
        
        for save_p, gen_text in zip(save_paths, gen_texts):
            print(gen_text)
            print(f"Save path: {save_p}")
            print("="*100)
            with open(save_p, 'w') as f:
                f.write(gen_text)

        #results.extend(gen_texts)

    
    return 
