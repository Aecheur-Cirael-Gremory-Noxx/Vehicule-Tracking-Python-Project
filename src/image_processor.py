#!/usr/bin/env python3
"""
Processeur d'images pour la détection de véhicules.
Traite une image statique avec YOLO et sauvegarde le résultat annoté.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import config
from .detection.detector_tracker import VehicleDetectorTracker


class ImageProcessor:
    """
    Processeur pour la détection de véhicules sur images statiques.
    """
    
    def __init__(self, input_path: str, output_path: Optional[str] = None):
        """
        Initialise le processeur d'images.
        
        Args:
            input_path: Chemin vers l'image d'entrée
            output_path: Chemin de sortie (optionnel)
        """
        self.logger = logging.getLogger(__name__)
        
        self.input_path = Path(input_path)
        
        # Générer le chemin de sortie si non spécifié
        if output_path:
            self.output_path = Path(output_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = self.input_path.stem
            self.output_path = Path(config.OUTPUT_DIR) / f"{stem}_detected_{timestamp}.png"
        
        # Créer le dossier de sortie
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le détecteur (sans tracker pour les images statiques)
        self.detector_tracker = None
        
        self.logger.info(f"ImageProcessor initialisé: {self.input_path} → {self.output_path}")
    
    def _init_detector(self):
        """Initialise le détecteur YOLO."""
        try:
            self.detector_tracker = VehicleDetectorTracker()
            self.logger.info("Détecteur YOLO initialisé")
        except Exception as e:
            self.logger.error(f"Erreur initialisation détecteur: {e}")
            raise
    
    def _load_image(self) -> np.ndarray:
        """
        Charge l'image d'entrée.
        
        Returns:
            Image BGR numpy array
        """
        try:
            image = cv2.imread(str(self.input_path))
            
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {self.input_path}")
            
            height, width = image.shape[:2]
            self.logger.info(f"Image chargée: {width}x{height}")
            
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur chargement image: {e}")
            raise
    
    def _detect_vehicles(self, image: np.ndarray) -> tuple:
        """
        Détecte les véhicules dans l'image.
        
        Args:
            image: Image BGR
            
        Returns:
            Tuple (detections, annotated_image)
        """
        try:
            # Faire la détection YOLO uniquement (pas de tracking)
            yolo_results = self.detector_tracker.yolo_model(
                image,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                classes=config.VEHICLE_CLASSES,
                verbose=False
            )
            
            # Convertir les détections
            detections = self.detector_tracker._yolo_to_deepsort_format(yolo_results, image.shape[:2])
            
            # Annoter l'image avec les détections
            annotated_image = self._annotate_image(image.copy(), yolo_results)
            
            return detections, annotated_image
            
        except Exception as e:
            self.logger.error(f"Erreur détection véhicules: {e}")
            return [], image
    
    def _annotate_image(self, image: np.ndarray, yolo_results) -> np.ndarray:
        """
        Annote l'image avec les boîtes de détection.
        
        Args:
            image: Image à annoter
            yolo_results: Résultats de détection YOLO
            
        Returns:
            Image annotée
        """
        if not yolo_results or len(yolo_results) == 0:
            return image
        
        boxes = yolo_results[0].boxes
        if boxes is None or len(boxes) == 0:
            return image
        
        # Extraire les informations
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        
        vehicle_count = 0
        
        for bbox, conf, cls_id in zip(xyxy, confidences, class_ids):
            # Filtrer par classe et confiance
            if cls_id in config.VEHICLE_CLASSES and conf >= config.CONFIDENCE_THRESHOLD:
                vehicle_count += 1
                
                # Coordonnées de la boîte
                x1, y1, x2, y2 = map(int, bbox)
                
                # Couleur en fonction de la classe
                color = self._get_class_color(cls_id)
                
                # Dessiner la boîte
                cv2.rectangle(image, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
                
                # Label avec classe et confiance
                class_name = self._get_class_name(cls_id)
                label = f"{class_name}: {conf:.2f}"
                
                # Taille du texte
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           config.FONT_SCALE, config.FONT_THICKNESS)[0]
                
                # Background pour le texte
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Texte du label
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           config.FONT_SCALE, (255, 255, 255), config.FONT_THICKNESS)
        
        # Ajouter un compteur global
        counter_text = f"Vehicules detectes: {vehicle_count}"
        cv2.putText(image, counter_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2)
        
        return image
    
    def _get_class_color(self, class_id: int) -> tuple:
        """
        Retourne une couleur pour chaque classe de véhicule.
        
        Args:
            class_id: ID de classe COCO
            
        Returns:
            Couleur BGR
        """
        colors = {
            2: (0, 255, 0),    # car - vert
            5: (255, 0, 0),    # bus - bleu  
            7: (0, 0, 255)     # truck - rouge
        }
        return colors.get(class_id, (255, 255, 255))  # blanc par défaut
    
    def _get_class_name(self, class_id: int) -> str:
        """
        Convertit l'ID de classe en nom.
        
        Args:
            class_id: ID de classe COCO
            
        Returns:
            Nom de la classe
        """
        names = {
            2: 'car',
            5: 'bus',
            7: 'truck'
        }
        return names.get(class_id, 'vehicle')
    
    def _save_image(self, image: np.ndarray) -> bool:
        """
        Sauvegarde l'image annotée.
        
        Args:
            image: Image à sauvegarder
            
        Returns:
            True si succès
        """
        try:
            success = cv2.imwrite(str(self.output_path), image)
            
            if success:
                # Informations sur le fichier
                file_size = self.output_path.stat().st_size / (1024 * 1024)  # MB
                self.logger.info(f"Image sauvegardée: {self.output_path}")
                self.logger.info(f"Taille fichier: {file_size:.2f} MB")
                return True
            else:
                self.logger.error("Échec de la sauvegarde")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde: {e}")
            return False
    
    def process_image(self) -> bool:
        """
        Traite l'image complète: chargement, détection, annotation, sauvegarde.
        
        Returns:
            True si succès
        """
        try:
            self.logger.info("Début du traitement d'image...")
            
            # 1. Initialiser le détecteur
            self._init_detector()
            
            # 2. Charger l'image
            image = self._load_image()
            
            # 3. Détecter les véhicules
            detections, annotated_image = self._detect_vehicles(image)
            
            # 4. Afficher les résultats
            vehicle_count = len(detections)
            self.logger.info(f"Véhicules détectés: {vehicle_count}")
            
            if vehicle_count > 0:
                for i, detection in enumerate(detections):
                    bbox, conf, class_name = detection
                    x, y, w, h = bbox
                    self.logger.info(f"  Véhicule {i+1}: {class_name} @ ({x:.0f},{y:.0f}) {w:.0f}x{h:.0f}, conf={conf:.2f}")
            
            # 5. Sauvegarder le résultat
            success = self._save_image(annotated_image)
            
            if success:
                self.logger.info("Traitement d'image terminé avec succès")
                return True
            else:
                self.logger.error("Échec du traitement d'image")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement: {e}")
            return False
    
    def get_processing_stats(self) -> dict:
        """
        Retourne les statistiques de traitement.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        return {
            'input_path': str(self.input_path),
            'output_path': str(self.output_path),
            'output_exists': self.output_path.exists()
        }