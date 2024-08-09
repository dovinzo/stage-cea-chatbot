from datasets import load_dataset

raw_datasets = load_dataset("glue","mrpc") # Récupération du jeu de données MRPC à partir du benchmark GLUE
print(raw_datasets)

# Accès à un split à partir de son nom
print(raw_datasets["train"])

# Accès à un élément d'un split
print(raw_datasets["train"][6])

# Accès à une tranche de notre jeu de données "train"
print(raw_datasets["train"][:5])

# Informations sur le jeu de données "train"
print(raw_datasets["train"].features)

from transformers import AutoTokenizer

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
	return tokenizer(example["sentence1"], example["sentence2"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = raw_datasets.map(tokenize_function)
print(tokenized_datasets.column_names)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) # C'est plus rapide. La fonction fournie recevra plusieurs exemples à chaque appel
