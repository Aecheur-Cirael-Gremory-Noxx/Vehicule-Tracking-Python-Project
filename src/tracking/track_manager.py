import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

import config


class TrackState(Enum):
    """États possibles d'un track."""
    TENTATIVE = "tentative"      # Track en cours de confirmation
    CONFIRMED = "confirmed"      # Track confirmé et actif
    DELETED = "deleted"          # Track supprimé/perdu


@dataclass
class TrackInfo:
    """Structure pour stocker les informations d'un track."""
    id: int
    state: TrackState
    bbox: List[float]  # [x1, y1, x2, y2]
    center: Tuple[float, float]  # (x, y)
    area: float
    age: int
    time_since_update: int
    class_name: str
    confidence: float
    frame_number: int


class TrackManager:
    """
    Gestionnaire pour le processing et le filtrage des tracks DeepSORT.
    Gère l'état des tracks, leur historique et les statistiques.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de tracks."""
        self.logger = logging.getLogger(__name__)
        
        # Historique des tracks par ID
        self.track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.TRAJECTORY_LENGTH)
        )
        
        # Statistiques globales
        self.total_tracks_created = 0
        self.active_track_count = 0
        self.confirmed_track_count = 0
        
        # Cache des derniers tracks traités
        self.last_tracks: List[TrackInfo] = []
        self.current_frame = 0
        
        self.logger.info("TrackManager initialisé")
    
    def process_tracks(self, raw_tracks: List, frame_number: int) -> Tuple[List[TrackInfo], List[TrackInfo]]:
        """
        Traite les tracks bruts de DeepSORT et les filtre selon leur état.
        
        Args:
            raw_tracks: Liste des tracks retournés par DeepSORT
            frame_number: Numéro de la frame actuelle
            
        Returns:
            Tuple (confirmed_tracks, tentative_tracks)
        """
        self.current_frame = frame_number
        confirmed_tracks = []
        tentative_tracks = []
        
        # Compteurs pour statistiques
        active_count = 0
        confirmed_count = 0
        
        for track in raw_tracks:
            # Extraire les informations du track
            track_info = self._extract_track_info(track, frame_number)
            
            if track_info is None:
                continue
            
            active_count += 1
            
            # Mettre à jour l'historique
            self._update_track_history(track_info)
            
            # Filtrer selon l'état
            if track.is_confirmed():
                track_info.state = TrackState.CONFIRMED
                confirmed_tracks.append(track_info)
                confirmed_count += 1
            else:
                track_info.state = TrackState.TENTATIVE
                tentative_tracks.append(track_info)
        
        # Mettre à jour les statistiques
        self.active_track_count = active_count
        self.confirmed_track_count = confirmed_count
        
        # Sauvegarder pour référence
        self.last_tracks = confirmed_tracks + tentative_tracks
        
        self.logger.debug(f"Frame {frame_number}: {confirmed_count} confirmés, "
                         f"{len(tentative_tracks)} tentatives")
        
        return confirmed_tracks, tentative_tracks
    
    def _extract_track_info(self, track, frame_number: int) -> Optional[TrackInfo]:
        """
        Extrait les informations pertinentes d'un track DeepSORT.
        
        Args:
            track: Track DeepSORT
            frame_number: Numéro de frame
            
        Returns:
            TrackInfo ou None si le track n'est pas valide
        """
        try:
            # Vérifier si le track est valide
            if not hasattr(track, 'track_id') or not hasattr(track, 'to_ltrb'):
                return None
            
            # Extraire la bounding box
            bbox = track.to_ltrb()  # [left, top, right, bottom]
            
            # Vérifier que la bbox est valide
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                return None
            
            # Calculer le centre et l'aire
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Extraire les autres informations
            track_id = track.track_id
            age = getattr(track, 'age', 0)
            time_since_update = getattr(track, 'time_since_update', 0)
            
            # Récupérer la classe et la confiance depuis la dernière détection
            class_name = "vehicle"  # Par défaut
            confidence = 0.0
            
            if hasattr(track, 'get_det_class') and track.get_det_class():
                class_name = track.get_det_class()
            
            if hasattr(track, 'get_det_conf') and track.get_det_conf():
                confidence = track.get_det_conf()
            
            return TrackInfo(
                id=track_id,
                state=TrackState.TENTATIVE,  # Sera mis à jour par process_tracks
                bbox=list(bbox),
                center=center,
                area=area,
                age=age,
                time_since_update=time_since_update,
                class_name=class_name,
                confidence=confidence,
                frame_number=frame_number
            )
            
        except Exception as e:
            self.logger.warning(f"Erreur lors de l'extraction du track: {e}")
            return None
    
    def _update_track_history(self, track_info: TrackInfo):
        """
        Met à jour l'historique d'un track.
        
        Args:
            track_info: Informations du track à ajouter
        """
        track_id = track_info.id
        
        # Ajouter à l'historique (deque avec taille max automatique)
        self.track_history[track_id].append(track_info)
        
        # Compter le nouveau track si c'est sa première apparition
        if len(self.track_history[track_id]) == 1:
            self.total_tracks_created += 1
    
    def get_confirmed_tracks(self) -> List[TrackInfo]:
        """
        Retourne uniquement les tracks confirmés de la dernière frame.
        
        Returns:
            Liste des tracks confirmés
        """
        return [track for track in self.last_tracks 
                if track.state == TrackState.CONFIRMED]
    
    def get_tentative_tracks(self) -> List[TrackInfo]:
        """
        Retourne uniquement les tracks tentatives de la dernière frame.
        
        Returns:
            Liste des tracks tentatives
        """
        return [track for track in self.last_tracks 
                if track.state == TrackState.TENTATIVE]
    
    def get_track_history(self, track_id: int) -> List[TrackInfo]:
        """
        Retourne l'historique complet d'un track.
        
        Args:
            track_id: ID du track
            
        Returns:
            Liste chronologique des TrackInfo
        """
        return list(self.track_history.get(track_id, []))
    
    def get_track_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Retourne la trajectoire (centres) d'un track.
        
        Args:
            track_id: ID du track
            
        Returns:
            Liste des positions du centre
        """
        history = self.track_history.get(track_id, [])
        return [track.center for track in history]
    
    def is_track_active(self, track_id: int) -> bool:
        """
        Vérifie si un track est encore actif.
        
        Args:
            track_id: ID du track
            
        Returns:
            True si le track est actif
        """
        if track_id not in self.track_history:
            return False
        
        history = self.track_history[track_id]
        if not history:
            return False
        
        # Un track est actif s'il a été vu récemment
        last_seen_frame = history[-1].frame_number
        frames_since_last_seen = self.current_frame - last_seen_frame
        
        return frames_since_last_seen <= config.MAX_AGE
    
    def cleanup_old_tracks(self):
        """
        Nettoie les tracks trop anciens pour libérer la mémoire.
        """
        tracks_to_remove = []
        
        for track_id in self.track_history:
            if not self.is_track_active(track_id):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
        
        if tracks_to_remove:
            self.logger.debug(f"Nettoyé {len(tracks_to_remove)} tracks inactifs")
    
    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques du tracking.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        return {
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': self.active_track_count,
            'confirmed_tracks': self.confirmed_track_count,
            'tentative_tracks': self.active_track_count - self.confirmed_track_count,
            'tracks_in_history': len(self.track_history),
            'current_frame': self.current_frame
        }
    
    def reset(self):
        """Remet à zéro toutes les données du gestionnaire."""
        self.track_history.clear()
        self.last_tracks.clear()
        self.total_tracks_created = 0
        self.active_track_count = 0
        self.confirmed_track_count = 0
        self.current_frame = 0
        
        self.logger.info("TrackManager réinitialisé")