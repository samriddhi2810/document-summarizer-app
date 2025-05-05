from transformers import pipeline

pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-248M", cache_dir="./lamini-lm")
