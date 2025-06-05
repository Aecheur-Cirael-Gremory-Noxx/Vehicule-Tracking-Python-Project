#!/usr/bin/env python3

import cv2
import sys
from src.detection.detector_tracker import VehicleDetectorTracker

def quick_test():
    print("Test rapide de 5 frames...")
    
    # Initialiser le détecteur
    detector = VehicleDetectorTracker()
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture("cars_exemple_footage.mp4")
    
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"Frame {i+1}...")
        try:
            tracks, annotated_frame = detector.process_frame(frame)
            print(f"  ✅ Réussi - {len(tracks)} tracks")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            break
    
    cap.release()
    print("Test terminé")

if __name__ == "__main__":
    quick_test()