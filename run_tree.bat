@echo off
cd /d "%~dp0"
echo Génération de l'arborescence Git suivie...
python generate_tree.py > tree.txt
echo Arborescence générée dans tree.txt
pause
