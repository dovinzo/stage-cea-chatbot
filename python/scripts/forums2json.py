#!/bin/env python3

"""
Nom du script:
    forums2json.py

Synopsis:
    python chemin/vers/forums2json.py [--help] [--forums_numbers ...] [--json_file_path_name ...] [--append] --forums_directory_path_name ...

Description:
    Ce script récupère les données "essentielles" des forums de la plateforme SALOME se trouvant dans un répetoire pour les placer dans un fichier JSON.

Options:
    --help
        Affiche cette aide et quitte.

    --forums_numbers
        La liste des numéros des forums qu'on considère.
        Exemple: 9,11
        Par défaut: Il s'agit de la liste 9,10,11,12,13,14.
        Attention: Il est très important de tout coller ! Par exemple, 9, 11 ne fonctionnera pas !
    
    --json_file_path_name
        Le nom du chemin vers le fichier JSON.
        Exemple: "chemin/vers/fichier.json"
        Par défaut, il s'agit de "./forums.json".
    
    --append
        Indique que les données des forums seront ajoutées au fichier JSON, et ils n'écraseront donc pas les données existantes. Si le fichier JSON n'existe pas, il sera créé.

Argument:
    --forums_directory_path_name
        Le nom du chemin vers le répertoire où se trouve les données "essentielles" des forums de la plateforme SALOME.
        Exemple: "chemin/vers/forums"

Dépendances: argparse, os, sys, transformers, forums

Auteur: Kelvin LEFORT

Date: 05/08/2024
"""

import argparse
import os
import sys
from transformers import pipeline

# Ajouter le répertoire parent du package 'forums' au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'packages'))

from forums import get_forums_data, get_forum_informations, get_page_informations, set_forums_data_2_json

def print_help():
    print(__doc__)
    sys.exit(0)

def parse_numbers_list(s):
    try:
        return [int(item) for item in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Les valeurs doivent être des nombres séparés par des virgules.\nPar exemple:\n\t9,11")

def get_arguments():
    """
    get_arguments - Récupère les arguments donnés avant l'éxécution du script
    
    Sortie:
        args (argparse.Namespace): Les arguments donnés avant l'éxécution du script

    Exemple:
        args = get_arguments()
    """

    # Créer le parseur
    parser = argparse.ArgumentParser(add_help=False)

    # Ajouter l'option --help
    parser.add_argument('--help', action='store_true')

    # Ajouter l'option --forums_numbers
    parser.add_argument('--forums_numbers', type=parse_numbers_list)

    # Ajouter l'option --json_file_path_name
    parser.add_argument('--json_file_path_name', type=str)

    # Ajouter l'option --append
    parser.add_argument('--append', action='store_true')

    # Ajouter l'argument forums_directory_path_name
    parser.add_argument('--forums_directory_path_name', type=str)

    # Analyser les arguments
    args = parser.parse_args()

    return args

def main():
    """
    main - Fonction principale orchestrant les tâches à effectuer
    """

    # Récupérer les arguments
    args = get_arguments()

    #print(f"--help:\n\t{args.help}\n")
    #print(f"--forums_numbers:\n\t{args.forums_numbers}\n")
    #print(f"--json_file_path_name:\n\t{args.json_file_path_name}\n")
    #print(f"--forums_directory_path_name:\n\t{args.forums_directory_path_name}")

    # ==================== Gestion des arguments ====================
    if args.help:
        print_help()
    if args.forums_numbers == None:
        args.forums_numbers = [9, 10, 11, 12, 13, 14]
    if args.json_file_path_name == None:
        args.json_file_path_name = os.path.join(os.getcwd(), 'forums.json')
    if args.forums_directory_path_name == None:
        print("Erreur: L'argument --forums_directory_path_name est obligatoire") # TODO : À MODIFIER
        sys.exit(0)

    summarizer = pipeline("summarization")

    # Récupérer les données des forums
    forums_data = get_forums_data(args.forums_numbers, args.forums_directory_path_name, summarizer)

    # Écrire les données dans le fichier JSON
    set_forums_data_2_json(forums_data, args.json_file_path_name, args.append)

if __name__ == "__main__":
    main()
