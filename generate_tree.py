import os
from collections import defaultdict

# Load list of git-tracked files
with os.popen('git ls-files') as f:
    paths = [line.strip() for line in f]

# Build a nested tree structure
tree = lambda: defaultdict(tree)
root = tree()
for path in paths:
    parts = path.split('/')
    current = root
    for part in parts:
        current = current[part]

# Output the tree as indented lines (tabs or spaces)
def print_indented_tree(d, indent=""):
    for key in sorted(d):
        print(f"{indent}{key}/" if d[key] else f"{indent}{key}")
        print_indented_tree(d[key], indent + "  ")

print_indented_tree(root)
