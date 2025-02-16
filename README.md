# Mutlilingul-To-Multimodal (M2M): Unlocking New Languages with Monolingual Text
This repository contains the code and resources for M2M. Below is an overview of the directory structure:


### Directory Structure 
1. `m2m_alignment_training` - Code for training the models.  
2. `translation` - Code for translation of synthetic data.  
3. `Synthetic_Multilingual_Eval_Dataset` - Synthetic data curated by us for AudioCaps (34 languages), Clotho (34 languages), MSCOCO-30K (10 languages).  
4. `eval_data` - Helper scripts supporting various tasks.  
5. `Checkpoints` - Trained checkpoints for the models.  

---

## Model & Data License  

All models used in this project are subject to the licenses of their respective sources. We strongly recommend users independently verify the licenses to ensure compliance with their intended use cases.  

### Models  
- Models from the [sentence-transformers](https://www.sbert.net/) library, including:  
  - **Multilingual CLIP (MCLIP-ST)**, **Multilingual MPNET (M-MPNET)**, and **Multilingual MiniLM (M-MiniLM)**  
- **LaBSE**, **KakaoBrain-ALIGN**, **Jina-CLIP-v1**, and **LAION-CLAP** (CLAP-General, CLAP-HTSAT-Fused)  
  - Licensed under **Apache License 2.0**.  
- **FLUX.1-dev**  
  - Generated outputs can be used for personal, scientific, and commercial purposes as specified in the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).  
- **Multilingual CLIP**, **OpenAI-CLIP**, and **IndicTrans2**  
  - Licensed under **MIT License**.  
- **Jina-CLIP-v2**, **Jina-embeddings-v3**, and **AYA-23-35B**  
  - Licensed under **CC-BY-NC-4.0**.  

Use of any combination of these models aligned using our method must adhere to the licenses of all individual models.  

### Datasets  
We release extended datasets in new languages for AudioCaps, Clotho, and MSCOCO2014-30K under **CC-BY-NC-4.0** license. This complies with the licenses of the source datasets and the models used to generate the data:  
- **AudioCaps** - MIT License  
- **Clotho** - [Tampere University License](https://github.com/audio-captioning/clotho-dataset?tab=License-1-ov-file#readme) (non-commercial with attribution)  
- **MSCOCO** - CC-BY-4.0  

### Disclaimer  
We have provided the licensing information to the best of our knowledge. However, users are **strongly encouraged to independently verify the licenses** of all models and datasets used, especially if using them for commercial purposes. We are not liable for any legal issues that may arise from the use of these resources.  

---

