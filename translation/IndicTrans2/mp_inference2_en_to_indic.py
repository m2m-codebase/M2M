import multiprocessing as mp
import pandas as pd
import torch
from inference.engine import Model
import sys
from tqdm import tqdm

# Worker function
def worker_task(task_queue, result_queue, gpu_id, ckpt_dir):
    # Load the model on the specified GPU
    model = Model(ckpt_dir, model_type="fairseq", device=f"cuda:{gpu_id}")
    print(f"Worker on GPU {gpu_id} initialized.")

    while True:
        task = task_queue.get()
        if task is None:  # Stop signal
            break
        
        sents = task["sents"]
        src_lang = task["src_lang"]
        tgt_lang = task["tgt_lang"]

        if isinstance(sents, str):
            sents = [sents]

        # Translate a batch of sentences
        trans_sent = model.batch_translate(sents, src_lang, tgt_lang)
        
        result_queue.put({
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "id": task["id"],
            "trans_sent": trans_sent,
        })

    print(f"Worker on GPU {gpu_id} shutting down.")

def main():
    ckpt_dir = sys.argv[1]
    eng_csv_file = sys.argv[2]
    en_col = sys.argv[3]
    batch_size = int(sys.argv[4])

    num_workers = 4
    gpu_ids = [0, 1]

    # Load the input CSV
    df = pd.read_csv(eng_csv_file)
    sents = df[en_col].tolist()

    # Create batches
    batches = [
        (i, sents[i:i + batch_size])
        for i in range(0, len(sents), batch_size)
    ]

    data = []
    src_lang = "eng_Latn"
    #tgt_langs = ["hin_Deva"]
    
    tgt_langs = [
        "ben_Beng", "guj_Gujr", "hin_Deva",
        "kan_Knda", "mal_Mlym", "mar_Deva",
        "npi_Deva", "pan_Guru", "tam_Taml",
        "tel_Telu", "urd_Arab",
    ]
    
    for tgt_lang in tgt_langs:
        for idx, b in batches:
            sample = {
                "id": idx,
                "sents": b,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }
            data.append(sample)

    print(f"Total data: {len(data)}")

    # Initialize queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Add tasks to the queue
    for sample in data:
        task_queue.put(sample)

    # Add stop signals
    for _ in range(num_workers):
        task_queue.put(None)

    # Start worker processes
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker_task, args=(task_queue, result_queue, gpu_ids[i % len(gpu_ids)], ckpt_dir))
        p.start()
        processes.append(p)

    # Collect results
    results = []
    for _ in tqdm(range(len(data)), desc="Translating"):
        results.append(result_queue.get())

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Sort results by task ID
    results = sorted(results, key=lambda x: x["id"])
    df_dict = {tgt: [] for tgt in tgt_langs}

    for r in results:
        tgt_lang = r["tgt_lang"]
        trans_sent = r["trans_sent"]
        df_dict[tgt_lang].extend(trans_sent)

    # Create translated DataFrame
    trans_df = pd.DataFrame(df_dict).reset_index(drop=True)
    df = df.reset_index(drop=True)
    
    assert len(df) == len(trans_df), f"df shapes dont match, {df.shape} and {trans_df.shape}"

    new_df = pd.concat([df, trans_df], axis=1)

    # Save the results
    save_path = eng_csv_file.replace('.csv', '_indictrans2.csv')
    new_df.to_csv(save_path, index=False)
    print(f"Translation saved to {save_path}")

if __name__ == "__main__":
    main()

