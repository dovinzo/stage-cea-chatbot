from transformers import AutoTokenizer


# Récupération du tokenizer pré-entraîné BERT
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Séquence à prétraiter par le tokenizer
#sequence = "J'aime manger des pommes."
sequence = "L'imposante voiture bleue se gare sur un parking."


####################################################
#                 Apperçu global                   #
####################################################

inputs = tokenizer(sequence)
print(inputs)


####################################################
#      Diviser le texte d'entrée en tokens         #
####################################################

tokens = tokenizer.tokenize(sequence)
print(tokens)


####################################################
#    Association des tokens à leur ID respectif    #
#    (définis par le vocabulaire du tokenizer)     #
####################################################

input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)


####################################################
#       Ajout des IDs des tokens spéciaux          #
####################################################

final_inputs = tokenizer.prepare_for_model(input_ids)
print(final_inputs["input_ids"])


####################################################
# Affichage des tokens (avec les tokens spéciaux)  #
####################################################

print(tokenizer.tokenize(tokenizer.decode(final_inputs["input_ids"])))


####################################################
#       Plusieurs phrases à prétraiter             #
####################################################

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences)
print(encoded_inputs)


####################################################
#                    Padding                       #
# a strategy for ensuring tensors are rectangular  #
#        by adding a special padding token         #
#             to shorter sentences.                #
####################################################

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences, padding=True)
print(encoded_inputs)


####################################################
#                   Truncation                     #
#   to truncate a sequence to the maximum length   #
#              accepted by the model               #
####################################################

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True)
print(encoded_inputs)


####################################################
#           Construction des tenseurs              #
####################################################

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt") # "pt" pour PyTorch
print(encoded_inputs)
