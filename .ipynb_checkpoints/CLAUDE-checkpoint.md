# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 20-week academic project from the Military Technical Academy of Bucharest for developing an intelligent road surveillance application. The system detects and tracks vehicles from fixed camera video feeds, identifying dangerous behaviors and traffic violations.

## Project Architecture

### Core Components
- **Vehicle Detection & Tracking**: Multi-vehicle detection using computer vision
- **Behavioral Analysis**: Classification of driving behaviors (normal, aggressive, suspicious trajectories)
- **Violation Detection**: Identification of traffic infractions (speeding, unsafe lane changes, line crossing)
- **Alert System**: Automated anomaly notification

### Technology Stack
- **Primary Language**: Python (for rapid prototyping and rich ecosystem)
- **Computer Vision**: OpenCV, YOLO (YOLOv5/YOLOv8) for real-time detection
- **Tracking**: Kalman filters for temporal vehicle tracking
- **Machine Learning**: SVM, Random Forest for behavior classification
- **License Plate Recognition**: OpenALPR library
- **Potential Frameworks**: TensorFlow/PyTorch if deep learning needed

### Current Structure
- `main.py`: Entry point (currently minimal)
- `src/`: Main source code directory
- `documentation/`: Project documentation including LaTeX specifications document

## Development Commands

### Running the Application
```bash
python main.py
```

### Documentation
The project specifications are maintained in LaTeX format:
```bash
# Compile documentation (if LaTeX tools available)
cd "documentation/cahier des charges"
pdflatex cahier_des_charges.tex
```

## Project Context

### Technical Constraints
- Fixed camera setup only (no mobile tracking)
- Offline processing acceptable (no strict real-time requirements)
- Normal weather conditions
- Academic timeline: 20 weeks
- Focus on proof of concept demonstration

### Functional Scope
- Multi-vehicle simultaneous processing
- Trajectory analysis over time
- Behavior classification (normal/aggressive/suspicious)
- Traffic violation detection (speed, safety distance, lane changes, line crossing)

The project emphasizes practical learning of computer vision and AI applied to real road safety problems, balancing functionality with academic constraints.