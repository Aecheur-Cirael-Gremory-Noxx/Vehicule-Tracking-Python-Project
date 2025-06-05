import time
import json
import pickle
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path

import config


@dataclass
class TrajectoryPoint:
    """Point de trajectoire avec toutes les données nécessaires."""
    timestamp: float           # Timestamp Unix
    frame_number: int         # Numéro de frame
    bbox: List[float]         # [x1, y1, x2, y2]
    center: Tuple[float, float]  # (x, y) centre de la bbox
    area: float              # Aire de la bbox
    confidence: float        # Confiance de la détection
    class_name: str         # Classe du véhicule (car, bus, truck)
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy) si calculée
    acceleration: Optional[float] = None            # Accélération si calculée


class TrajectoryExtractor:
    """Extracteur optimisé pour les données des tracks DeepSORT."""
    
    @staticmethod
    def extract_track_data(track, frame_number: int, timestamp: float) -> Optional[Dict]:
        """
        Extrait toutes les données pertinentes d'un track DeepSORT.
        
        Args:
            track: Objet Track de DeepSORT
            frame_number: Numéro de la frame actuelle
            timestamp: Timestamp de la frame
            
        Returns:
            Dictionnaire avec les données extraites ou None si erreur
        """
        try:
            # Vérifications de base
            if not hasattr(track, 'track_id') or not hasattr(track, 'to_ltrb'):
                return None
            
            # Extraire l'ID du track
            track_id = track.track_id
            
            # Extraire la bounding box
            bbox = track.to_ltrb()  # [left, top, right, bottom]
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                return None
            
            # Calculer le centre et l'aire
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Extraire la classe et la confiance
            class_name = "vehicle"  # Valeur par défaut
            confidence = 0.0
            
            # Tenter d'extraire la classe depuis DeepSORT
            if hasattr(track, 'get_det_class') and track.get_det_class():
                class_name = track.get_det_class()
            elif hasattr(track, 'det_class') and track.det_class:
                class_name = track.det_class
            
            # Tenter d'extraire la confiance
            if hasattr(track, 'get_det_conf') and track.get_det_conf() is not None:
                confidence = float(track.get_det_conf())
            elif hasattr(track, 'det_conf') and track.det_conf is not None:
                confidence = float(track.det_conf)
            
            # Créer le point de trajectoire
            trajectory_point = TrajectoryPoint(
                timestamp=timestamp,
                frame_number=frame_number,
                bbox=list(bbox),
                center=center,
                area=area,
                confidence=confidence,
                class_name=class_name
            )
            
            return {
                'track_id': track_id,
                'point': trajectory_point,
                'is_confirmed': track.is_confirmed() if hasattr(track, 'is_confirmed') else False,
                'age': getattr(track, 'age', 0),
                'time_since_update': getattr(track, 'time_since_update', 0)
            }
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Erreur extraction track: {e}")
            return None


class TrajectoryStorage:
    """
    Gestionnaire de stockage des trajectoires par track_id avec historique
    complet des positions, timestamps et métadonnées.
    """
    
    def __init__(self, max_trajectory_length: Optional[int] = None):
        """
        Initialise le stockage des trajectoires.
        
        Args:
            max_trajectory_length: Longueur max des trajectoires (None = illimité)
        """
        self.logger = logging.getLogger(__name__)
        
        # Utiliser config ou paramètre fourni
        self.max_length = max_trajectory_length or config.TRAJECTORY_LENGTH
        
        # Stockage principal : Dict[track_id] -> deque[TrajectoryPoint]
        self.trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.max_length)
        )
        
        # Métadonnées par track
        self.track_metadata: Dict[int, Dict] = {}
        
        # Statistiques
        self.total_points_stored = 0
        self.active_tracks = set()
        
        self.logger.info(f"TrajectoryStorage initialisé (max_length: {self.max_length})")
    
    def add_track_point(self, track_id: int, point: TrajectoryPoint, metadata: Optional[Dict] = None):
        """
        Ajoute un point de trajectoire pour un track.
        
        Args:
            track_id: ID du track
            point: Point de trajectoire à ajouter
            metadata: Métadonnées optionnelles du track
        """
        # Ajouter le point à la trajectoire
        self.trajectories[track_id].append(point)
        self.total_points_stored += 1
        
        # Marquer le track comme actif
        self.active_tracks.add(track_id)
        
        # Mettre à jour les métadonnées si fournies
        if metadata:
            if track_id not in self.track_metadata:
                self.track_metadata[track_id] = {}
            self.track_metadata[track_id].update(metadata)
    
    def process_tracks_batch(self, tracks_data: List[Dict], timestamp: float = None):
        """
        Traite un batch de tracks extraits.
        
        Args:
            tracks_data: Liste de dictionnaires de données de tracks
            timestamp: Timestamp de la frame (utilise time.time() si None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        for track_data in tracks_data:
            if not track_data:
                continue
            
            track_id = track_data['track_id']
            point = track_data['point']
            
            # Métadonnées du track
            metadata = {
                'is_confirmed': track_data.get('is_confirmed', False),
                'age': track_data.get('age', 0),
                'time_since_update': track_data.get('time_since_update', 0),
                'last_seen_frame': point.frame_number,
                'last_seen_timestamp': timestamp
            }
            
            self.add_track_point(track_id, point, metadata)
    
    def get_trajectory(self, track_id: int) -> List[TrajectoryPoint]:
        """
        Retourne la trajectoire complète d'un track.
        
        Args:
            track_id: ID du track
            
        Returns:
            Liste des points de trajectoire
        """
        return list(self.trajectories.get(track_id, []))
    
    def get_trajectory_centers(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Retourne uniquement les centres de la trajectoire.
        
        Args:
            track_id: ID du track
            
        Returns:
            Liste des positions (x, y)
        """
        trajectory = self.trajectories.get(track_id, [])
        return [point.center for point in trajectory]
    
    def get_trajectory_timespan(self, track_id: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Retourne la période temporelle d'un track.
        
        Args:
            track_id: ID du track
            
        Returns:
            Tuple (timestamp_début, timestamp_fin) ou (None, None)
        """
        trajectory = self.trajectories.get(track_id, [])
        if not trajectory:
            return None, None
        
        return trajectory[0].timestamp, trajectory[-1].timestamp
    
    def get_recent_points(self, track_id: int, seconds: float) -> List[TrajectoryPoint]:
        """
        Retourne les points récents d'un track.
        
        Args:
            track_id: ID du track
            seconds: Période en secondes
            
        Returns:
            Liste des points récents
        """
        trajectory = self.trajectories.get(track_id, [])
        if not trajectory:
            return []
        
        cutoff_time = time.time() - seconds
        return [point for point in trajectory if point.timestamp >= cutoff_time]
    
    def calculate_velocities(self, track_id: int) -> bool:
        """
        Calcule les vitesses pour un track basé sur ses positions.
        
        Args:
            track_id: ID du track
            
        Returns:
            True si le calcul a réussi
        """
        trajectory = self.trajectories.get(track_id, [])
        if len(trajectory) < 2:
            return False
        
        try:
            # Convertir en liste pour modification
            points = list(trajectory)
            
            for i in range(1, len(points)):
                prev_point = points[i-1]
                curr_point = points[i]
                
                # Calculer le delta temps
                dt = curr_point.timestamp - prev_point.timestamp
                if dt <= 0:
                    continue
                
                # Calculer la vitesse (pixels/seconde)
                dx = curr_point.center[0] - prev_point.center[0]
                dy = curr_point.center[1] - prev_point.center[1]
                
                vx = dx / dt
                vy = dy / dt
                
                # Mettre à jour le point avec la vitesse
                points[i].velocity = (vx, vy)
            
            # Reconstruire la deque
            self.trajectories[track_id] = deque(points, maxlen=self.max_length)
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur calcul vitesse track {track_id}: {e}")
            return False
    
    def get_track_statistics(self, track_id: int) -> Dict:
        """
        Retourne les statistiques d'un track.
        
        Args:
            track_id: ID du track
            
        Returns:
            Dictionnaire avec les statistiques
        """
        trajectory = self.trajectories.get(track_id, [])
        metadata = self.track_metadata.get(track_id, {})
        
        if not trajectory:
            return {'track_id': track_id, 'length': 0}
        
        # Calculer les statistiques de base
        start_time, end_time = self.get_trajectory_timespan(track_id)
        duration = (end_time - start_time) if (start_time and end_time) else 0
        
        # Statistiques de mouvement
        centers = [point.center for point in trajectory]
        distances = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            distances.append(np.sqrt(dx*dx + dy*dy))
        
        total_distance = sum(distances) if distances else 0
        avg_speed = total_distance / duration if duration > 0 else 0
        
        return {
            'track_id': track_id,
            'length': len(trajectory),
            'duration': duration,
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'start_frame': trajectory[0].frame_number,
            'end_frame': trajectory[-1].frame_number,
            'class_name': trajectory[-1].class_name,
            'average_confidence': np.mean([p.confidence for p in trajectory]),
            **metadata
        }
    
    def export_trajectories(self, filepath: Union[str, Path], format: str = 'json') -> bool:
        """
        Exporte les trajectoires vers un fichier.
        
        Args:
            filepath: Chemin du fichier de sortie
            format: Format d'export ('json' ou 'pickle')
            
        Returns:
            True si l'export a réussi
        """
        try:
            filepath = Path(filepath)
            
            # Préparer les données pour l'export
            export_data = {
                'trajectories': {},
                'metadata': self.track_metadata,
                'config': {
                    'max_length': self.max_length,
                    'total_points': self.total_points_stored
                }
            }
            
            # Convertir les trajectoires
            for track_id, trajectory in self.trajectories.items():
                export_data['trajectories'][track_id] = [
                    asdict(point) for point in trajectory
                ]
            
            # Sauvegarder selon le format
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(export_data, f)
            else:
                raise ValueError(f"Format non supporté: {format}")
            
            self.logger.info(f"Trajectoires exportées vers {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur export trajectoires: {e}")
            return False
    
    def get_all_active_tracks(self) -> List[int]:
        """Retourne la liste des IDs de tracks actifs."""
        return list(self.active_tracks)
    
    def cleanup_inactive_tracks(self, max_age_seconds: float = None):
        """
        Nettoie les tracks inactifs.
        
        Args:
            max_age_seconds: Âge maximum en secondes (utilise config si None)
        """
        if max_age_seconds is None:
            # Convertir MAX_AGE (frames) en secondes (approximation)
            max_age_seconds = config.MAX_AGE / 30.0  # Assume 30 FPS
        
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id in list(self.trajectories.keys()):
            trajectory = self.trajectories[track_id]
            if trajectory:
                last_point = trajectory[-1]
                age = current_time - last_point.timestamp
                
                if age > max_age_seconds:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.trajectories[track_id]
            self.track_metadata.pop(track_id, None)
            self.active_tracks.discard(track_id)
        
        if tracks_to_remove:
            self.logger.info(f"Nettoyé {len(tracks_to_remove)} tracks inactifs")
    
    def get_storage_statistics(self) -> Dict:
        """Retourne les statistiques globales du stockage."""
        return {
            'total_tracks': len(self.trajectories),
            'active_tracks': len(self.active_tracks),
            'total_points_stored': self.total_points_stored,
            'max_trajectory_length': self.max_length,
            'memory_usage_estimate': sum(len(traj) for traj in self.trajectories.values())
        }
    
    def reset(self):
        """Remet à zéro tout le stockage."""
        self.trajectories.clear()
        self.track_metadata.clear()
        self.active_tracks.clear()
        self.total_points_stored = 0
        self.logger.info("TrajectoryStorage réinitialisé")