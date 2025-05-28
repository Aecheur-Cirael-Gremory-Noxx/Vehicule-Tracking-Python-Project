import sys
import os
from pathlib import Path
from collections import defaultdict

# Forcer l'encodage de stdout en UTF-8
sys.stdout.reconfigure(encoding='utf-8')


# Obtenir la liste des fichiers suivis par Git
tracked_files = os.popen('git ls-files').read().splitlines()

# Créer un arbre hiérarchique
tree = lambda: defaultdict(tree)
root = tree()

for file in tracked_files:
    parts = file.split('/')
    current = root
    for part in parts:
        current = current[part]

# Fonction récursive pour afficher joliment
def print_tree(d, prefix=''):
    entries = list(d.items())
    for i, (name, subtree) in enumerate(entries):
        connector = '└── ' if i == len(entries) - 1 else '├── '
        print(prefix + connector + name)
        if subtree:
            extension = '    ' if i == len(entries) - 1 else '│   '
            print_tree(subtree, prefix + extension)

# Exemple : suppose que ton projet est dans "traffic_monitor/"
print("traffic_monitor/")
print_tree(root)
