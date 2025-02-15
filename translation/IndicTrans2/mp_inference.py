import multiprocessing as mp
import torch
from inference.engine import Model
import sys
from tqdm import tqdm
import pandas as pd


# Worker function
def worker_task(task_queue, result_queue, gpu_id, ckpt_dir):
    # Load the model on the specified GPU
    model = Model(ckpt_dir, model_type="fairseq", device=f"cuda:{gpu_id}")

    print(f"Worker on GPU {gpu_id} initialized.")

    while True:
        task = task_queue.get()  # Get task from the queue
        if task is None:  # Stop signal
            break
        
        sents = task["sents"]
        src_lang = task["src_lang"] 
        tgt_lang = task["tgt_lang"]


        if isinstance(sents, str):
            sents = [sents]

        # for a batch of sentences
        trans_sent = model.batch_translate(sents, src_lang, tgt_lang)
        #model.translate_paragraph(text, src_lang, tgt_lang)
        
        # Put result in the result queue
        result_queue.put({
            "src_lang": src_lang, 
            "tgt_lang": tgt_lang, 
            "id": task['id'], 
            "trans_sent": trans_sent
        })

    print(f"Worker on GPU {gpu_id} shutting down.")

def main():
    ckpt_dir = sys.argv[1]
    eng_csv_file = sys.argv[2]
    en_col = sys.argv[3] 
    batch_size = sys.argv[4]

    
    # round-robin for workers on gpus
    num_workers = 2  # Number of workers/GPU devices
    gpu_ids = [0, 1]
    num_gpus = len(gpu_ids)


    # Initialize queues
    # each worker has its own task queue
    task_queue = [mp.Queue() for _ in range(num_workers)]
    # all workers share result queue
    result_queue = mp.Queue()

    # Create worker processes
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker_task, args=(task_queue[i], result_queue, gpu_ids[i%num_gpus], ckpt_dir))
        p.start()
        processes.append(p)

    print("p.start")
    df = pd.read_csv(eng_csv_file).head(11)
    sents = df[en_col].tolist()

    batches = [
        [i, sents[i:i+batch_size]]
        for i in range(0, len(sents), batch_size)
    ]

    data = []
    
    src_lang = "eng_Latn"
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

    print("Total data", len(data))
    for idx, sample in enumerate(data):
        print("task queue", idx%num_workers)
        task_queue[idx%num_workers].put(sample)

    # Add stop signals for workers
    for tq in task_queue:
        tq.put(None)

    # Collect results
    results = []
    for _ in tqdm(data, desc="Translating"):
        results.append(result_queue.get())

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Print results
    results = sorted(results, key=lambda x: x["id"])  # Sort by task ID
    
    df_dict = {tgt: [] for tgt in tgt_langs}
    
    for r in results:
        l = r["tgt_lang"]
        trans_sent = r["trans_sent"]
    
        df_dict[l].extend(trans_sent)

    trans_df = pd.DataFrame(df_dict)
    new_df = pd.concat([df, trans_df])

    save_p = eng_csv_file.replace('.csv', '_indictrans2.csv')
    new_df.to_csv(save_p, index=False)


if __name__ == "__main__":
    main()

