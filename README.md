The idea here is to use the corpus of FP7, H2020 and HORIZON projects to finetune an encoder-only model like ModernBERT to be a better retriever in this type of data.

I will:

1. Establish an eval dataset to assess ModernBERT performance.
2. Generate a synthetic dataset for ModernBERT finetuning given the corpus of FP7, H2020 and HORIZON. For this:
- Summarize each project with a local LLM
- For each project, generate two queries (a more technical one, and a more natural one) simulating what a user wants to find.
- Use the metadata such as keywords and origin topic to create hard and easy negatives.
- Store each record on a jsonl file containing *project_id*, *query*, *positive_text*, *easy_negative_text*, *hard_negative_text*
3. Fine-tune (full, as ModernBERT is small) and compare model improvement. We can finetune using MNRL (no negatives needed) and triplet loss (with easy and hard negatives)