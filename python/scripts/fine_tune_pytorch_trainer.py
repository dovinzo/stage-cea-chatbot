######################################################################
# Ce script Python donne un exemple complet de l'utilisation         #
# de la classe PyTorch Trainer fourni par la librairie Transformers  #
# pour fine-tune un modèle                                         . #
######################################################################


# Pour pouvoir utiliser la fonction np.argmax
import numpy as np

# Pour pouvoir évaluer la performance du modèle durant l'entraînement
import evaluate

# Pour pouvoir récupérer un modèle de classification de séquences.
from transformers import AutoModelForSequenceClassification

# Pour pouvoir modifier les hyperparamètres (qui contrôlent le comportement de l'algorithme d'apprentissage)
from transformers import TrainingArguments

# Pour pouvoir entraîner le modèle (fine-tunning)
from transformers import Trainer

# Pour récupérer des jeux de données sur le Hub
from datasets import load_dataset

# Pour pouvoir récupérer un tokenizer
from transformers import AutoTokenizer


# Récupération d'un jeu de données, puis on le tokenize
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
def tokenize_function(examples):
	return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

# Récupération d'un modèle de classification de séquences.
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5) # num_labels indique le nombre de labels (i.e. classes). Ici, il y en a 5 (note sur 5 étoiles).

# Récupération des informations concernant les hyperparamètres à considérer
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch") # - À modifier selon ses besoins
                                                                                    # - Le paramètre eval_strategy permet de rapporter l'évaluation de la métrique à la fin de chaque époque

# Récupération d'une métrique permettant de calculer la performance du modèle pendant l'entraînement
metric = evaluate.load("accuracy")

# Fonction permettant de calculer la précision de nos prédictions à partir de la métrique
def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)

# Création d'un objet Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune le modèle
trainer.train()
