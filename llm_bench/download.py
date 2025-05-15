from datasets import load_dataset

dataset = load_dataset(
    "ccdv/arxiv-summarization",
    cache_dir="/root/autodl-tmp/data"
)
