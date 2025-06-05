# 🚗 Système de Détection et Tracking de Véhicules

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)](https://ultralytics.com)
[![DeepSORT](https://img.shields.io/badge/DeepSORT-1.3+-red.svg)](https://github.com/levan92/deep_sort_realtime)

Prototype de surveillance routière intelligente développé pour l'Académie Technique Militaire de Bucarest. Le système détecte et suit les véhicules dans des vidéos de surveillance routière en utilisant YOLO pour la détection et DeepSORT pour le tracking.

## 🎯 Fonctionnalités

### 🔍 Détection et Tracking
- **Détection multi-véhicules** avec YOLOv8 (voitures, bus, camions)
- **Tracking persistant** avec DeepSORT et similarité d'apparence
- **IDs uniques** maintenant la cohérence temporelle
- **Filtrage intelligent** des tracks confirmés vs tentatives

### 🎨 Visualisation Avancée
- **Couleurs persistantes** par track_id (même véhicule = même couleur)
- **Annotations différenciées** : boîtes solides (confirmés) vs pointillés (tentatives)
- **Labels informatifs** : `ID:{track_id} {class} ({état})`
- **Trajectoires optionnelles** avec historique configurable

### 📹 Export Professionnel
- **Codec H264 optimisé** avec détection automatique
- **Même FPS/résolution** que la vidéo originale
- **Validation export** et vérification de lisibilité
- **Métadonnées complètes** exportées en JSON

### 📊 Gestion des Données
- **Stockage trajectoires** avec timestamps précis
- **Historique complet** des positions et métadonnées
- **Statistiques détaillées** de tracking et performance
- **Nettoyage automatique** des tracks inactifs

## 🚀 Installation

### Prérequis
- Python 3.8+
- CUDA (optionnel, pour accélération GPU)
- FFmpeg (pour certains codecs vidéo)

### Installation des dépendances

```bash
# Cloner le repository
git clone <repository-url>
cd Vehicule-Tracking-Python-Project

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
- `opencv-python` - Traitement vidéo et vision par ordinateur
- `ultralytics` - YOLOv8 pour la détection d'objets
- `deep-sort-realtime` - Algorithme de tracking DeepSORT
- `torch` - Framework deep learning (avec support CUDA)
- `numpy`, `scipy`, `matplotlib`, `pandas` - Calculs scientifiques

## 📖 Utilisation

### Commande de base

```bash
python main.py input_video.mp4
```

### Options avancées

```bash
# Spécifier fichier de sortie
python main.py input_video.mp4 -o output_tracked.mp4

# Mode verbose (debug)
python main.py input_video.mp4 -v

# Désactiver affichage progression
python main.py input_video.mp4 --no-progress

# Aide complète
python main.py --help
```

### Formats vidéo supportés
- MP4, AVI, MOV, MKV, WMV, FLV
- Codecs : H264, MP4V, XVID, etc.

## ⚙️ Configuration

Le fichier `config.py` permet de personnaliser tous les paramètres :

### Paramètres YOLO
```python
CONFIDENCE_THRESHOLD = 0.5    # Seuil de confiance détection
IOU_THRESHOLD = 0.45         # Seuil pour suppression non-maximale
VEHICLE_CLASSES = [2, 5, 7]  # Classes véhicules (car, bus, truck)
DEVICE = 'cuda'              # Device (auto-détection CUDA/CPU)
```

### Paramètres DeepSORT
```python
MAX_AGE = 30                 # Frames avant suppression track
N_INIT = 3                   # Détections pour confirmer track
MAX_IOU_DISTANCE = 0.7       # Distance IoU association
MAX_COSINE_DISTANCE = 0.3    # Similarité apparence
NN_BUDGET = 100              # Taille buffer features
```

### Paramètres Export
```python
OUTPUT_CODEC = 'H264'        # Codec vidéo de sortie
OUTPUT_QUALITY = 90          # Qualité compression (0-100)
DRAW_TRAJECTORIES = True     # Dessiner trajectoires
TRAJECTORY_LENGTH = 30       # Points de trajectoire à conserver
```

## 🏗️ Architecture

```
src/
├── detection/
│   └── detector_tracker.py      # Pipeline YOLO + DeepSORT intégré
├── tracking/
│   ├── track_manager.py          # Gestion états des tracks
│   └── trajectory_storage.py     # Stockage trajectoires avec timestamps
├── utils/
│   ├── video_capture.py          # Capture vidéo robuste
│   ├── visualization.py          # Rendu annotations et couleurs
│   └── export_manager.py         # Export vidéo H264 optimisé
├── video_processor.py            # Orchestrateur pipeline complet
└── __init__.py
```

### Flux de traitement

1. **VideoCapture** → Lecture vidéo avec métadonnées
2. **VehicleDetectorTracker** → Détection YOLO + Tracking DeepSORT
3. **TrackManager** → Filtrage confirmés/tentatives
4. **TrajectoryStorage** → Stockage historique avec timestamps
5. **VisualizationRenderer** → Annotations avec couleurs persistantes
6. **ExportManager** → Export H264 MP4 avec validation

## 📊 Outputs

### Vidéo annotée
- Format : MP4 avec codec H264
- Annotations : Bounding boxes colorées + IDs + états
- Trajectoires : Historique de mouvement (optionnel)
- Même résolution/FPS que l'original

### Métadonnées JSON
```json
{
  "input_file": "input_video.mp4",
  "output_file": "output/input_video_tracked_20241205_143022.mp4",
  "duration_seconds": 125.4,
  "frames_processed": 3762,
  "codec": "H264",
  "file_size_mb": 45.2,
  "compression_ratio": 2.1
}
```

### Logs détaillés
- Progression temps réel (FPS, tracks actifs)
- Statistiques finales (tracks créés, points trajectoire)
- Erreurs et avertissements
- Sauvegarde : `vehicle_tracking.log`

## 🎛️ Exemples d'usage

### Surveillance parking
```bash
# Détection véhicules avec trajectoires
python main.py parking_surveillance.mp4 -o parking_tracked.mp4
```

### Analyse trafic routier
```bash
# Mode verbose pour debugging
python main.py traffic_highway.mov -v
```

### Traitement par lots
```bash
# Script pour traiter plusieurs vidéos
for video in *.mp4; do
    python main.py "$video" -o "tracked_$video"
done
```

## 🐛 Troubleshooting

### Erreurs communes

**Erreur CUDA**
```
Solution: Installer CUDA toolkit ou forcer CPU dans config.py
DEVICE = 'cpu'
```

**Codec H264 non disponible**
```
Solution: Installer FFmpeg ou utiliser fallback
OUTPUT_CODEC = 'mp4v'
```

**Mémoire insuffisante**
```
Solution: Réduire résolution ou utiliser frame skip
RESIZE_WIDTH = 640
FRAME_SKIP = 2
```

### Optimisation performance

- **GPU** : Utiliser CUDA pour YOLO et DeepSORT
- **Résolution** : Réduire si nécessaire (`RESIZE_WIDTH`)
- **Frame skip** : Traiter 1 frame sur N (`FRAME_SKIP`)
- **Trajectoires** : Désactiver si non nécessaires (`DRAW_TRAJECTORIES = False`)

## 📈 Performances

### Configuration recommandée
- **CPU** : Intel i5+ ou AMD Ryzen 5+
- **RAM** : 8GB minimum, 16GB recommandé
- **GPU** : NVIDIA GTX 1060+ pour accélération CUDA
- **Stockage** : SSD pour vidéos volumineuses

### Benchmarks typiques
- **1080p @ 30fps** : ~15-20 FPS en traitement
- **720p @ 30fps** : ~25-30 FPS en traitement
- **Détection** : ~50ms par frame (GPU), ~200ms (CPU)
- **Tracking** : ~10ms par frame

## 🔬 Contexte académique

### Projet Académie Technique Militaire de Bucarest
- **Durée** : 20 semaines
- **Objectif** : Prototype surveillance routière intelligente
- **Contraintes** : Caméras fixes, traitement offline acceptable
- **Focus** : Apprentissage vision par ordinateur et IA

### Applications potentielles
- Surveillance trafic routier
- Détection violations (future extension)
- Analyse comportements conducteurs
- Systèmes de sécurité routière

## 📄 Licence

Projet académique - Académie Technique Militaire de Bucarest

## 🤝 Contribution

Projet développé dans un cadre académique. Pour questions ou suggestions :
- Consulter la documentation dans `documentation/`
- Vérifier les logs dans `vehicle_tracking.log`
- Tester avec vidéos courtes d'abord

---

**Développé avec ❤️ pour l'apprentissage de la vision par ordinateur et de l'IA appliquée à la sécurité routière**