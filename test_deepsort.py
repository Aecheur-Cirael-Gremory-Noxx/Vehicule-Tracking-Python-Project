#!/usr/bin/env python3

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Test simple de YOLOv8 + DeepSORT
def test_detection():
    print("Test de détection YOLOv8 + DeepSORT")
    
    # Charger le modèle YOLO
    model = YOLO("yolov8n.pt")
    
    # Initialiser DeepSORT
    tracker = DeepSort(max_age=30, n_init=3)
    
    # Charger une frame de test
    cap = cv2.VideoCapture("cars_exemple_footage.mp4")
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur: impossible de lire la vidéo")
        return
    
    print(f"Frame chargée: {frame.shape}")
    
    # Faire une détection YOLO
    results = model(frame, verbose=False)
    
    print(f"Résultats YOLO: {len(results)}")
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        print(f"Boxes détectées: {len(boxes)}")
        
        # Convertir pour DeepSORT
        detections = []
        if len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            
            for bbox, conf, cls_id in zip(xyxy, confidences, class_ids):
                # Filtrer les véhicules (classes 2, 5, 7)
                if cls_id in [2, 5, 7] and conf >= 0.5:
                    x1, y1, x2, y2 = bbox
                    x, y, w, h = x1, y1, x2-x1, y2-y1
                    
                    # Format pour DeepSORT
                    detections.append(([float(x), float(y), float(w), float(h)], float(conf), "vehicle"))
        
        print(f"Détections pour DeepSORT: {len(detections)}")
        
        # Test DeepSORT
        try:
            tracks = tracker.update_tracks(detections, frame=frame)
            print(f"Tracks créés: {len(tracks)}")
            print("✅ Test réussi!")
        except Exception as e:
            print(f"❌ Erreur DeepSORT: {e}")
    else:
        print("Aucune détection YOLO")
    
    cap.release()

if __name__ == "__main__":
    test_detection()