# TnDRR – AI/ML Internship Assignment

> This repository contains my implementations and experiments for the TnDRR AI/ML Internship Assignment.The assignment focuses on three core areas of modern AI engineering: LLM fine-tuning, Retrieval-Augmented Generation (RAG), and local LLM inference optimization.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine_Learning-orange)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![Git](https://img.shields.io/badge/Git-Version_Control-f05032)

---

## Project Structure

```text
tndrr_assignment/
│
├── task1_finetuning/
│   ├── notebooks/
│   └── src/
│
├── task2_rag/
│   ├── notebooks/
│   └── src/
│
├── task3_local_llm/
│   ├── notebooks/
│   └── src/
│
├── report/
│
├── requirements.txt
└── README.md
```

## Tasks

### Task 1 – Fine-tuning with LoRA/QLoRA
Fine-tune a small open-weight model (≤ 1.5B parameters, e.g. Qwen2.5-0.5B/1.5B or SmolLM2) on a narrow task of your choice, for example turning plain-English requests into a fixed JSON format, or a classification task. A free Colab or Kaggle GPU is enough.
Submit:
• Base model vs fine-tuned model on a held-out test set (at least 50 examples), using a clear metric
• Your LoRA settings (rank, alpha, target modules, learning rate, epochs) and why you chose them
• 3 examples where the fine-tuned model still fails, and why you think it fails
• A short calculation, with working shown: roughly how much GPU memory is needed to fine-tune a 7B model with (a) full fine-tuning, (b) LoRA, (c) QLoRA

### Task 2 – Retrieval-Augmented Generation (RAG) from scratch
Build a question-answering system over any public document set of at least 30 pages, without using LangChain, LlamaIndex or similar frameworks. Use only an embedding model plus NumPy/FAISS (and any LLM for the final answer).
Submit:
• 15 test questions you wrote yourself, each with the correct source passage
• Retrieval accuracy (e.g. recall@5) for two different chunk sizes, and which worked better and why
• 2 cases where retrieval failed, and what would fix them
• In your own words: what BM25 adds to embedding search, and what a re-ranker does

### Task 3 – Running a model locally & Optimization
Run a small open model on your own machine or a free GPU using Ollama, llama.cpp or vLLM, in two settings, e.g. 16-bit vs 4-bit quantised, or two quantisation levels.
Submit:
• Tokens per second and memory usage for each setting
• Output quality compared on the same 10 prompts, with your observations
• In your own words: what the KV cache is, and why quantisation saves memory
• A back-of-envelope plan: how would you serve 20 users at the same time on one 24 GB GPU? State your assumptions.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- Sentence Transformers
- NumPy / FAISS
- Ollama / llama.cpp

## Objective

The objective is to gain practical experience with LLM fine-tuning, retrieval systems, model optimization, and experimentation while understanding the engineering decisions behind each implementation.

---

*This repository is actively being developed. Results, experiments, and documentation will be updated as the tasks progress.*