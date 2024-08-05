"""
Nom du module: forums.py

Description: Ce module permet de récupérer/manipuler/modifier les données "essentielles" des forums de la plateforme SALOME pour pouvoir ensuite les placer dans un fichier JSON.
             J'entends par "essentielle" :
                     - Une page qui contient au moins deux chaats, c'est-à-dire de manière générale qu'il y a au moins une réponse à la question.

Informations:
	- Il y six forums : forum_9, forum_10, forum_11, forum_12, forum_13 et forum_14.
	- Chaque forum possède plusieurs pages, identifiées par un nombre entier. Par exemple, dans forum_9, il y a la page numéro 7203642.
	- Chaque page contient un titre et une discussion entre plusieurs personnes. L'objectif de cette discussion est de répondre à la question d'un utilisateur.
	- Voici l'arborescence du fichier JSON qu'on souhaite produire :
	
	    "forums" (list) : La liste de chaque forum (dict) :
	            "number" (int) : Le numéro du forum
	            "pages" (list) : La liste de chaque page "essentielle" du forum (dict) :
	                    "path" (str) : Le nom du chemin vers la page
	                    "number" (int) : Le numéro de la page
	                    "title" (str) : Le titre de la page
	                    "discussion" (list) : La liste de chaque chat (str) dans la discussion

Dépendances: os, re, json, bs4, filecmp

Auteur: Kelvin LEFORT

Date: 30/07/2024
"""

import os
import re
import json
from bs4 import BeautifulSoup
import filecmp

def get_forums_data(forums_numbers, forums_directory_path_name, summarizer):
    """
    get_forums_data - Récupère les données "essentielles" des forums de la plateforme SALOME.

    Argument:
        forums_numbers (list): La liste des numéros des forums qu'on considère. 
        forums_directory_path_name (str): Le nom du chemin vers le répertoire où se trouve les données "essentielles" des forums de la plateforme SALOME.
    
    Sortie:
        forums_data (dict): Les données "essentielles" des forums de la plateforme SALOME. Elles ont le même format que les données dans le fichiern JSON forums.json.
    
    Exemple:
        forums_data = get_forums_data([9, 10], "/home/projects/stage-cea-chatbot/data/forums/")
    """

    # Définir le pattern indiquant que seules les chaînes de caractères contenant que des chiffres sont pris en compte
    pattern = re.compile(r'^\d+$')

    # Déclarer les données des forums comme un dictionnaire vide
    forums_data = {}

    # Déclarer le champ "forums" de data comme une liste vide
    forums_data["forums"] = []

    # ==================== Récupération des informations de chaque forum ====================
    for forum_number in forums_numbers:
        forum_informations = get_forum_informations(forum_number, forums_directory_path_name, pattern, summarizer)
        
        # Ajouter les informations du forum en fin de liste de data["forums"]
        forums_data["forums"].append(forum_informations)

    return forums_data

def get_forum_informations(forum_number, forums_directory_path_name, pattern, summarizer):
    """
    get_forum_informations - Récupère les informations "essentielles" d'un seul forum de la plateforme SALOME.

    Arguments:
        forum_number (int): Le numéro du forum.
        forums_directory_path_name (str): Le nom du chemin vers le répertoire où se trouve les données "essentielles" des forums de la plateforme SALOME.
        pattern (re.Pattern): Le pattern pour sélectionner que certains sous-répertoires (pages) du forum.
    
    Sortie:
        forum_informations (dict): Les informations "essentielles" du forum. Elles ont le format suivant :

            "number" (int) : Le numéro du forum,
            "pages" (list) : La liste de chaque page du forum (dict) :
                    "path" (str) : Le nom du chemin vers la page
                    "number" (int) : Le numéro de la page,
                    "title" (str) : Le titre de la page,
                    "discussion" (list) : La liste de chaque chat (str) dans la discussion.
    
    Exemple:
        pattern = re.compile(r'^\d+$')
        forum_informations = get_forum_informations(9, "/home/projects/stage-cea-chatbot/data/forums/", pattern)
    """

    # Déclarer les informations du forum comme un dictionnaire vide
    forum_informations = {}

    # Écrire le numéro du forum dans le champ "number" de forum_informations
    forum_informations["number"] = forum_number

    # Déclarer le champ "pages" de forum_informations comme une liste vide
    forum_informations["pages"] = []

    # Déterminer le chemin vers le répertoire où se trouve les données de ce forum
    forum_directory_path_name = os.path.join(forums_directory_path_name, f"forum_{forum_number}")

    # Déterminer le nom des sous-répertoires où se trouvent les données de chaque page de ce forum
    pages_number = [name for name in os.listdir(forum_directory_path_name) if os.path.isdir(os.path.join(forum_directory_path_name, name)) and pattern.match(name)]

    # ==================== Récupération des informations de chaque page du forum ====================
    for page_number in pages_number:
        page_informations = get_page_informations(page_number, forum_directory_path_name, summarizer)

        if page_informations:
            # Ajouter les informations de la page en fin de liste de forum_informations["pages"]
            forum_informations["pages"].append(page_informations)
    
    return forum_informations

def get_page_informations(page_number, forum_directory_path_name, summarizer):
    """
    get_page_informations - Récupère les informations "essentielles" d'une seule page d'un seul forum de la plateforme SALOME.

    Arguments:
        page_number (int): Le numéro de la page.
        forum_directory_path_name (str): Le nom du chemin vers le répertoire où se trouve les données "essentielles" du forum.
    
    Sortie:
        page_informations (dict):
            - S'il y a au moins une réponse à la question : Les informations "essentielles" de la page. Elles ont le format suivant :

                "path" (str) : Le nom du chemin vers la page
                "number" (int) : Le numéro de la page,
                "title" (str) : Le titre de la page,
                "discussion" (list) : La liste de chaque chat (str) dans la discussion.
                
            - S'il n'y a pas de réponse(s) à la question : {}

    Exemple:
        page_informations = get_page_informations(7203642, "/home/projects/stage-cea-chatbot/data/forums/forum_9/")
    """

    # Déclarer les informations de la page comme un dictionnaire vide
    page_informations = {}
    
    # Déterminer le chemin vers le répertoire où se trouve les données de cette page
    page_directory_path_name = os.path.join(forum_directory_path_name, page_number)
    
    # Déterminer le nom de tous les fichiers HTML du répertoire page_directory
    html_files_name = [f for f in os.listdir(page_directory_path_name) if f.endswith('.html')]

    # Garder un seul de ces fichiers et récupérer son chemin
    html_file_path_name = os.path.join(page_directory_path_name, html_files_name[0])
    
    # Écrire le nom du chemin vers la page HTML dans le champ "path" de page_informations
    page_informations["path"] = html_file_path_name

    # Utiliser le package BeautifulSoup
    with open(html_file_path_name, "r") as f:
        doc = BeautifulSoup(f, "html.parser")
    
    # Écrire le numéro de la page dans le champ "number" de page_informations
    page_informations["number"] = int(page_number)

    # Écrire le titre de la page dans le champ "title" de page_informations
    page_informations["title"] = doc.title.string

    # Écrire la discussion ayant lieu dans la page comme une liste de str dans le champ "discussion" de page_informations
    page_informations["discussion"] = [chat.get_text(strip=True) for chat in doc.find_all('div', class_="boardCommentContent")]
    
    if len(page_informations["discussion"]) <= 1:
        page_informations = {}
    else:
        try:
            text = page_informations["discussion"][0]
            page_informations["summary"] = summarizer(text, max_length=256, do_sample=False)
        except:
            page_informations["summary"] = "Exception raised while calling pipeline"
    return page_informations

def set_forums_data_2_json(forums_data, json_file_path_name, append):
    """
    set_forums_data_2_json - Écrit les données des forums dans un fichier JSON.

    Arguments:
        forums_data (dict): Les données des forums (voir la fonction get_forums_data pour le format).
        json_file_path_name (str): Le nom du chemin vers le fichier JSON.
        append (bool): Indique que les données des forums seront ajoutées au fichier JSON, et ils n'écraseront donc pas les données existantes. Si le fichier JSON n'existe pas, il sera créé.
    
    Exemple:
        forums_data = get_forums_data("/home/projects/stage-cea-chatbot/data/forums/")
        set_forums_data_2_json(forums_data, "/home/projects/stage-cea-chatbot/data/json/forums.json", True)
    """
    
    if append:
        # Vérifier si le fichier existe
        if os.path.exists(json_file_path_name):
            # Lire les données existantes du fichier JSON
            with open(json_file_path_name, 'r') as file:
                existing_forums_data = json.load(file)
        else:
            # Initialiser les données si le fichier n'existe pas
            existing_forums_data = {"forums": []}
        for forum_informations in forums_data["forums"]:
            forum_found = False
            for existing_forum_informations in existing_forums_data["forums"]:
                if forum_informations["number"] == existing_forum_informations["number"]:
                    forum_found = True
                    break
            if not forum_found:
                existing_forums_data["forums"].append(forum_informations)
            else:
                for page_informations in forum_informations["pages"]:
                    if page_informations["number"] not in [existing_page_informations["number"] for existing_page_informations in existing_forum_informations["pages"]]:
                        existing_forum_informations["pages"].append(page_informations)
    else:
        existing_forums_data = forums_data

    with open(json_file_path_name, 'w', encoding='utf-8') as file:
        json.dump(existing_forums_data, file, ensure_ascii=False, indent=4)
