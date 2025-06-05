import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import config


class VehicleDetectorTracker:
    """
    Classe unifiée combinant YOLO pour la détection et DeepSORT pour le tracking
    des véhicules dans une vidéo de surveillance routière.
    """
    
    def __init__(self):
        """
        Initialise le détecteur YOLO et le tracker DeepSORT avec les paramètres
        définis dans config.py
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Initialiser YOLO
        self._init_yolo()
        
        # Initialiser DeepSORT
        self._init_deepsort()
        
        self.logger.info("VehicleDetectorTracker initialisé avec succès")
        
    def _init_yolo(self):
        """Initialise le modèle YOLO avec les paramètres de configuration."""
        try:
            self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
            self.yolo_model.to(config.DEVICE)
            
            self.logger.info(f"YOLO chargé: {config.YOLO_MODEL_PATH} sur {config.DEVICE}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de YOLO: {e}")
            raise
    
    def _init_deepsort(self):
        """Initialise DeepSORT avec les paramètres de configuration."""
        try:
            self.deepsort = DeepSort(
                max_age=config.MAX_AGE,
                n_init=config.N_INIT,
                max_iou_distance=config.MAX_IOU_DISTANCE,
                max_cosine_distance=config.MAX_COSINE_DISTANCE,
                nn_budget=config.NN_BUDGET,
                override_track_class=None,
                embedder="mobilenet",  # Utilise MobileNet pour l'extraction de features
                half=True,  # Utilise FP16 si disponible
                bgr=True,  # Format BGR d'OpenCV
                embedder_gpu=True if config.DEVICE == 'cuda' else False,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            
            self.logger.info("DeepSORT initialisé avec les paramètres de config")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de DeepSORT: {e}")
            raise
    
    def _yolo_to_deepsort_format(self, yolo_results, frame_shape: Tuple[int, int]) -> List[Tuple]:
        """
        Convertit les détections YOLO au format attendu par DeepSORT.
        
        Args:
            yolo_results: Résultats de détection YOLO
            frame_shape: (height, width) de la frame
            
        Returns:
            Liste de tuples (bbox, confidence, class_name) au format DeepSORT
            où bbox = [x, y, w, h] (coordonnées absolues)
        """
        detections = []
        
        try:
            if yolo_results and len(yolo_results) > 0:
                # Extraire les boîtes, scores et classes
                boxes = yolo_results[0].boxes
                
                if boxes is not None and len(boxes) > 0:
                    # Convertir en numpy arrays de manière sécurisée
                    xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy()
                    
                    # Vérifier que nous avons des données
                    if len(xyxy) > 0 and len(confidences) > 0 and len(class_ids) > 0:
                        # Convertir class_ids en entiers de manière sécurisée
                        class_ids = np.array(class_ids, dtype=np.int64)
                        
                        for i in range(len(xyxy)):
                            bbox = xyxy[i]
                            conf = float(confidences[i])
                            cls_id = int(class_ids[i])
                            
                            # Filtrer par classe de véhicule et confiance
                            if cls_id in config.VEHICLE_CLASSES and conf >= config.CONFIDENCE_THRESHOLD:
                                # Convertir de [x1, y1, x2, y2] vers [x, y, w, h]
                                x1, y1, x2, y2 = map(float, bbox)
                                x, y, w, h = x1, y1, x2-x1, y2-y1
                                
                                # Vérifier que la boîte est valide
                                if w > 0 and h > 0:
                                    # Format DeepSORT: ([x, y, w, h], confidence, class_name)
                                    detections.append(([x, y, w, h], conf, "vehicle"))
                                    
        except Exception as e:
            self.logger.error(f"Erreur conversion YOLO vers DeepSORT: {e}")
            
        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """
        Convertit l'ID de classe COCO en nom de classe.
        
        Args:
            class_id: ID de la classe COCO
            
        Returns:
            Nom de la classe
        """
        class_names = {
            2: 'car',
            5: 'bus', 
            7: 'truck'
        }
        return class_names.get(class_id, 'vehicle')
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Traite une frame complète: détection YOLO + tracking DeepSORT.
        
        Args:
            frame: Image BGR de la frame à traiter
            
        Returns:
            Tuple (tracks, frame_annotee) où:
            - tracks: Liste des objets trackés avec leurs informations
            - frame_annotee: Frame avec les annotations visuelles
        """
        try:
            # 1. Détection avec YOLO
            yolo_results = self.yolo_model(
                frame,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                classes=config.VEHICLE_CLASSES,
                verbose=False
            )
            
            # 2. Conversion au format DeepSORT
            detections = self._yolo_to_deepsort_format(yolo_results, frame.shape[:2])
            
            # Debug: Afficher les détections
            if len(detections) > 0:
                self.logger.debug(f"Détections trouvées: {len(detections)}")
                self.logger.debug(f"Premier élément: {detections[0]}")
                self.logger.debug(f"Type bbox: {type(detections[0][0])}")
                self.logger.debug(f"Contenu bbox: {detections[0][0]}")
            
            # 3. Mise à jour du tracker
            tracks = []
            if len(detections) > 0:
                try:
                    tracks = self.deepsort.update_tracks(detections, frame=frame)
                except Exception as e:
                    self.logger.error(f"Erreur update_tracks avec détections: {e}")
            else:
                try:
                    # Maintenir la continuité du tracking même sans nouvelles détections
                    tracks = self.deepsort.update_tracks([], frame=frame)
                except Exception as e:
                    self.logger.debug(f"Pas de tracking à mettre à jour: {e}")
                    tracks = []
            
            # 4. Annotation de la frame
            annotated_frame = self._annotate_frame(frame.copy(), tracks)
            
            return tracks, annotated_frame
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de la frame: {e}")
            return [], frame
    
    def _annotate_frame(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """
        Annote la frame avec les boîtes englobantes et IDs des tracks.
        
        Args:
            frame: Frame à annoter
            tracks: Liste des tracks actifs
            
        Returns:
            Frame annotée
        """
        try:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                # Extraire les informations du track
                track_id = track.track_id
                bbox = track.to_ltrb()  # [left, top, right, bottom]
                
                # Debug pour voir le contenu exact
                self.logger.debug(f"Track {track_id} bbox type: {type(bbox)}, contenu: {bbox}")
                if len(bbox) > 0:
                    self.logger.debug(f"Premier élément bbox type: {type(bbox[0])}, valeur: {repr(bbox[0])}")
                
                # Convertir les coordonnées en entiers de manière sécurisée
                try:
                    # Vérifier si bbox contient des valeurs numériques
                    if any(isinstance(coord, str) for coord in bbox):
                        self.logger.error(f"Bbox contient des chaînes: {bbox}")
                        continue
                    
                    x1, y1, x2, y2 = [int(float(coord)) for coord in bbox]
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Bbox invalide pour track {track_id}: {bbox}, erreur: {e}")
                    continue
                
                # Vérifier que les coordonnées sont valides
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Dessiner la boîte englobante
                color = self._get_track_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, int(config.BOX_THICKNESS))
                
                # Ajouter l'ID du track
                label = f"ID: {track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           float(config.FONT_SCALE), int(config.FONT_THICKNESS))[0]
                
                # Background pour le texte
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Texte de l'ID
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           float(config.FONT_SCALE), (255, 255, 255), int(config.FONT_THICKNESS))
                           
        except Exception as e:
            self.logger.error(f"Erreur annotation frame: {e}")
        
        return frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Génère une couleur unique pour chaque track basée sur son ID.
        
        Args:
            track_id: ID du track
            
        Returns:
            Couleur BGR
        """
        # Utiliser l'ID pour générer une couleur unique
        np.random.seed(track_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
    
    def get_active_tracks_info(self, tracks: List) -> List[dict]:
        """
        Extrait les informations des tracks actifs pour analyse ultérieure.
        
        Args:
            tracks: Liste des tracks
            
        Returns:
            Liste de dictionnaires avec les informations des tracks
        """
        tracks_info = []
        
        for track in tracks:
            if track.is_confirmed():
                bbox = track.to_ltrb()
                
                track_info = {
                    'id': track.track_id,
                    'bbox': bbox,
                    'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'age': track.age,
                    'time_since_update': track.time_since_update
                }
                
                tracks_info.append(track_info)
        
        return tracks_info
    
    def reset_tracker(self):
        """Remet à zéro le tracker (utile entre différentes vidéos)."""
        try:
            self._init_deepsort()
            self.logger.info("Tracker DeepSORT réinitialisé")
        except Exception as e:
            self.logger.error(f"Erreur lors de la réinitialisation: {e}")
    
    def __del__(self):
        """Nettoyage des ressources."""
        try:
            if hasattr(self, 'deepsort'):
                del self.deepsort
        except Exception:
            pass