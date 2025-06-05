import os
import torch

# =============================================================================
# CONFIGURATION CENTRALISÉE POUR LE PROJET DE DÉTECTION DE CONDUITE AGRESSIVE
# =============================================================================

# -----------------------------------------------------------------------------
# CHEMINS ET RÉPERTOIRES
# -----------------------------------------------------------------------------

# Chemin vers la vidéo d'entrée à analyser
INPUT_VIDEO_PATH = "data/input_video.mp4"

# Dossier de sortie pour les résultats
OUTPUT_DIR = "output"

# Chemin vers le modèle YOLO (téléchargé automatiquement si inexistant)
YOLO_MODEL_PATH = "yolov8n.pt"

# Création automatique des dossiers s'ils n'existent pas
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True)

# -----------------------------------------------------------------------------
# PARAMÈTRES YOLO
# -----------------------------------------------------------------------------

# Seuil de confiance pour la détection (0.0 à 1.0)
CONFIDENCE_THRESHOLD = 0.5

# Seuil IoU pour la suppression non-maximale
IOU_THRESHOLD = 0.45

# Classes de véhicules dans COCO dataset
# 2: car, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 5, 7]

# Device de calcul (détection automatique CUDA/CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# PARAMÈTRES VIDÉO
# -----------------------------------------------------------------------------

# Traiter 1 frame sur N (1 = toutes les frames, 2 = une frame sur deux, etc.)
FRAME_SKIP = 1

# Largeur cible pour redimensionnement (None = pas de redimensionnement)
RESIZE_WIDTH = None

# Maintenir le ratio d'aspect lors du redimensionnement
MAINTAIN_ASPECT_RATIO = True

# FPS de sortie (None = même que l'entrée)
OUTPUT_FPS = None

# -----------------------------------------------------------------------------
# PARAMÈTRES D'EXPORT
# -----------------------------------------------------------------------------

# Codec vidéo pour l'export ('mp4v', 'H264', 'XVID')
OUTPUT_CODEC = 'mp4v'

# Qualité de compression vidéo (0-100, 100 = meilleure qualité)
OUTPUT_QUALITY = 90

# Dessiner les trajectoires des véhicules
DRAW_TRAJECTORIES = True

# Nombre de points de trajectoire à conserver par véhicule
TRAJECTORY_LENGTH = 30

# -----------------------------------------------------------------------------
# PARAMÈTRES DEEPSORT
# -----------------------------------------------------------------------------

# Nombre de frames avant de supprimer un track perdu
MAX_AGE = 30

# Nombre de détections consécutives avant de confirmer un track
N_INIT = 3

# Seuil de distance IoU pour l'association des détections
MAX_IOU_DISTANCE = 0.7

# Seuil de distance cosinus pour la similarité d'apparence
MAX_COSINE_DISTANCE = 0.3

# Taille du budget pour les features d'apparence
NN_BUDGET = 100

# -----------------------------------------------------------------------------
# PARAMÈTRES D'ANALYSE COMPORTEMENTALE
# -----------------------------------------------------------------------------

# Seuil de vitesse pour détecter un excès (km/h)
SPEED_THRESHOLD = 50

# Seuil d'accélération pour détecter un freinage/accélération brusque (m/s²)
ACCELERATION_THRESHOLD = 3.0

# Seuil de changement de direction pour détecter un zigzag (degrés)
DIRECTION_CHANGE_THRESHOLD = 15

# Nombre de frames pour calculer les moyennes mobiles
SMOOTHING_WINDOW = 5

# -----------------------------------------------------------------------------
# PARAMÈTRES D'AFFICHAGE
# -----------------------------------------------------------------------------

# Couleurs pour les niveaux d'agressivité (BGR format)
COLORS = {
    'normal': (0, 255, 0),      # Vert
    'moderate': (0, 165, 255),  # Orange
    'aggressive': (0, 0, 255)   # Rouge
}

# Épaisseur des boîtes englobantes
BOX_THICKNESS = 2

# Taille de la police pour les textes
FONT_SCALE = 0.6

# Épaisseur du texte
FONT_THICKNESS = 2

# -----------------------------------------------------------------------------
# PARAMÈTRES CSV
# -----------------------------------------------------------------------------

# Nom du fichier CSV de sortie
CSV_OUTPUT_FILENAME = "detection_results.csv"

# Délimiteur CSV
CSV_DELIMITER = ','

# Inclure l'en-tête dans le CSV
CSV_INCLUDE_HEADER = True