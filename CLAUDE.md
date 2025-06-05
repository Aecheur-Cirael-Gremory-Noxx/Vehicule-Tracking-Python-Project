# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 20-week academic project from the Military Technical Academy of Bucharest for developing an intelligent road surveillance application. The system detects and tracks vehicles from fixed camera video feeds using YOLOv8 and DeepSORT.

**Current Status**: ✅ **PROTOTYPE COMPLET** - Pipeline fonctionnel de détection et tracking avec export vidéo H264.

## Project Architecture

### Implemented Core Components ✅
- **Vehicle Detection & Tracking**: YOLOv8 + DeepSORT pipeline complet
- **Persistent Tracking**: IDs uniques avec couleurs cohérentes
- **Professional Visualization**: Annotations différenciées confirmés/tentatives
- **H264 Video Export**: Export MP4 optimisé avec métadonnées JSON
- **Trajectory Storage**: Historique complet avec timestamps

### Future Extensions 🔮
- **Behavioral Analysis**: Classification normale/agressive/suspecte (non implémentée)
- **Violation Detection**: Vitesse, changements de voie, franchissement lignes
- **Alert System**: Notifications automatiques d'anomalies
- **License Plate Recognition**: Identification véhicules

### Technology Stack
- **Primary Language**: Python 3.8+
- **Computer Vision**: OpenCV 4.8+, YOLOv8 (ultralytics)
- **Tracking**: DeepSORT avec similarité d'apparence
- **Deep Learning**: PyTorch avec support CUDA
- **Video Processing**: H264 codec, métadonnées export

### Current Structure
```
├── main.py                          # Entry point CLI complet ✅
├── config.py                        # Configuration centralisée ✅
├── requirements.txt                 # Dépendances YOLO+DeepSORT ✅
├── src/
│   ├── detection/
│   │   └── detector_tracker.py      # Pipeline YOLO+DeepSORT ✅
│   ├── tracking/
│   │   ├── track_manager.py         # Gestion états tracks ✅
│   │   └── trajectory_storage.py    # Stockage trajectoires ✅
│   ├── utils/
│   │   ├── video_capture.py         # Capture vidéo robuste ✅
│   │   ├── visualization.py         # Rendu couleurs persistantes ✅
│   │   └── export_manager.py        # Export H264 optimisé ✅
│   └── video_processor.py           # Orchestrateur principal ✅
├── data/                            # Vidéos d'entrée
├── output/                          # Résultats annotés
└── documentation/                   # Spécifications LaTeX
```

## Development Commands

### Running the Application
```bash
# Usage basique
python main.py input_video.mp4

# Avec options
python main.py input_video.mp4 -o output_tracked.mp4 -v

# Aide complète
python main.py --help
```

### Installation
```bash
# Installer dépendances
pip install -r requirements.txt

# Test rapide (vérifier imports)
python -c "import torch, cv2, ultralytics; print('✅ Dépendances OK')"
```

### Configuration
Modifier `config.py` pour personnaliser :
- **YOLO**: Seuils confiance, classes véhicules, device CUDA/CPU
- **DeepSORT**: Paramètres tracking, âge tracks, distances
- **Export**: Codec H264, qualité, trajectoires, résolution

## Project Context

### Technical Constraints
- ✅ Fixed camera setup only (no mobile tracking)
- ✅ Offline processing (pipeline optimisé pour vidéos)
- ✅ Normal weather conditions
- ✅ Academic timeline: 20 weeks (prototype terminé)
- ✅ Focus on proof of concept demonstration

### Current Functional Scope ✅
- ✅ Multi-vehicle simultaneous processing
- ✅ Persistent tracking with unique IDs
- ✅ Professional visualization with color-coded states
- ✅ H264 video export with metadata
- ✅ Trajectory storage and analysis-ready data

### Future Scope 🔮
- Trajectory analysis over time (données disponibles)
- Behavior classification (normal/aggressive/suspicious)
- Traffic violation detection (speed, safety distance, lane changes)

## Usage Guidelines for Claude Code

### When working on this project:

1. **Architecture is complete** - All core components implemented and integrated
2. **Configuration-driven** - Use `config.py` for all parameters
3. **Modular design** - Each component has clear responsibilities
4. **Professional logging** - Use existing logger patterns
5. **Error handling** - All components have robust error management

### Key Files to Understand:
- `main.py` - CLI interface and argument parsing
- `src/video_processor.py` - Main orchestrator linking all components
- `config.py` - Central configuration for all parameters
- `src/detection/detector_tracker.py` - YOLO+DeepSORT integration

### Common Modifications:
- **New features**: Add to appropriate module (detection/, tracking/, utils/)
- **Performance tuning**: Modify config.py parameters
- **Visualization changes**: Update `src/utils/visualization.py`
- **Export formats**: Extend `src/utils/export_manager.py`

The project emphasizes practical learning of computer vision and AI applied to real road safety problems. The current prototype provides a solid foundation for future behavioral analysis extensions.

## Current Limitations & Future Work

### Completed ✅
- Vehicle detection and tracking pipeline
- Professional video annotation and export
- Persistent ID management with visual consistency
- Robust error handling and logging

### Not Implemented 🔮
- Real-time behavioral analysis (speeding, aggressive driving)
- Camera calibration for pixel-to-real-world conversion
- CSV export of detected incidents
- Advanced trajectory analysis algorithms

The prototype successfully demonstrates core computer vision concepts and provides a functional vehicle tracking system ready for academic evaluation.