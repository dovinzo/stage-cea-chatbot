from datasets import load_dataset
from transformers import AutoTokenizer

# Récupération du jeu de données Yelp Reviews
dataset = load_dataset("yelp_review_full")

# Exemple d'une donnée de ce jeu de données ("train")
print(dataset["train"][100])

# Récupération du tokenizer pré-entraîné BERT
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Fonction de prétraitement (tokenization) qui sera appliquée sur tout le jeu de données
def tokenize_function(examples):
	return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenization de chaque donnée du jeu de données
tokenized_datasets = dataset.map(tokenize_function, batched=True) # C'est plus rapide de mettre "batched=True". La fonction fournie recevra plusieurs exemples à chaque appel

# Pour mélanger le jeu de données et sélectionner certains éléments ("train")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

# Pour mélanger le jeu de données et sélectionner certains éléments ("test")
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
