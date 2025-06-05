import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
import colorsys

import config
from ..tracking import TrackInfo, TrackState


class VisualizationStyle(Enum):
    """Styles de visualisation disponibles."""
    SOLID = "solid"              # Boîte pleine
    DASHED = "dashed"           # Boîte en pointillés (tentative)
    THICK = "thick"             # Boîte épaisse (confirmé)


class ColorManager:
    """Gestionnaire de couleurs persistantes pour les tracks."""
    
    def __init__(self):
        """Initialise le gestionnaire de couleurs."""
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}
        self.used_colors = set()
        self.color_seed = 42
        
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Retourne une couleur persistante pour un track_id.
        
        Args:
            track_id: ID du track
            
        Returns:
            Couleur BGR
        """
        if track_id not in self.track_colors:
            # Générer une nouvelle couleur unique
            color = self._generate_unique_color(track_id)
            self.track_colors[track_id] = color
            self.used_colors.add(color)
        
        return self.track_colors[track_id]
    
    def _generate_unique_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Génère une couleur unique et visuellement distincte.
        
        Args:
            track_id: ID du track pour la génération
            
        Returns:
            Couleur BGR
        """
        # Utiliser le golden ratio pour répartir les teintes
        golden_ratio = 0.618033988749895
        hue = (track_id * golden_ratio) % 1.0
        
        # Saturation et valeur élevées pour des couleurs vives
        saturation = 0.8 + 0.2 * ((track_id * 0.1) % 1.0)  # 0.8-1.0
        value = 0.8 + 0.2 * ((track_id * 0.07) % 1.0)      # 0.8-1.0
        
        # Convertir HSV vers RGB puis BGR
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        bgr = tuple(int(c * 255) for c in reversed(rgb))
        
        return bgr
    
    def reset(self):
        """Remet à zéro toutes les couleurs."""
        self.track_colors.clear()
        self.used_colors.clear()


class VisualizationRenderer:
    """
    Gestionnaire de rendu pour l'annotation des frames avec tracking.
    Gère les couleurs persistantes et les différents états de tracks.
    """
    
    def __init__(self):
        """Initialise le renderer de visualisation."""
        self.logger = logging.getLogger(__name__)
        self.color_manager = ColorManager()
        
        # Configuration des styles
        self.styles = {
            TrackState.CONFIRMED: {
                'thickness': config.BOX_THICKNESS + 1,
                'style': VisualizationStyle.THICK,
                'alpha': 1.0
            },
            TrackState.TENTATIVE: {
                'thickness': config.BOX_THICKNESS,
                'style': VisualizationStyle.DASHED,
                'alpha': 0.7
            }
        }
        
        # Configuration des textes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.FONT_SCALE
        self.font_thickness = config.FONT_THICKNESS
        
        self.logger.info("VisualizationRenderer initialisé")
    
    def render_tracks(self, frame: np.ndarray, tracks: List[TrackInfo]) -> np.ndarray:
        """
        Rend les tracks sur une frame avec annotations complètes.
        
        Args:
            frame: Frame à annoter
            tracks: Liste des tracks à afficher
            
        Returns:
            Frame annotée
        """
        if not tracks:
            return frame
        
        # Copier la frame pour éviter les modifications
        annotated_frame = frame.copy()
        
        # Trier les tracks : tentatives d'abord, confirmés au dessus
        sorted_tracks = sorted(tracks, key=lambda t: t.state == TrackState.CONFIRMED)
        
        for track in sorted_tracks:
            annotated_frame = self._render_single_track(annotated_frame, track)
        
        # Ajouter les informations globales
        annotated_frame = self._render_global_info(annotated_frame, tracks)
        
        return annotated_frame
    
    def _render_single_track(self, frame: np.ndarray, track: TrackInfo) -> np.ndarray:
        """
        Rend un track individuel sur la frame.
        
        Args:
            frame: Frame à modifier
            track: Track à afficher
            
        Returns:
            Frame modifiée
        """
        # Obtenir la couleur persistante
        color = self.color_manager.get_track_color(track.id)
        
        # Obtenir le style selon l'état
        style_config = self.styles.get(track.state, self.styles[TrackState.TENTATIVE])
        
        # Extraire les coordonnées de la bbox
        x1, y1, x2, y2 = map(int, track.bbox)
        
        # Dessiner la bounding box
        frame = self._draw_bounding_box(
            frame, (x1, y1, x2, y2), color, 
            style_config['thickness'], style_config['style']
        )
        
        # Créer le label
        label = self._create_track_label(track)
        
        # Dessiner le label
        frame = self._draw_label(frame, label, (x1, y1), color, track.state)
        
        return frame
    
    def _draw_bounding_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                          color: Tuple[int, int, int], thickness: int, 
                          style: VisualizationStyle) -> np.ndarray:
        """
        Dessine une bounding box avec le style spécifié.
        
        Args:
            frame: Frame à modifier
            bbox: Coordonnées (x1, y1, x2, y2)
            color: Couleur BGR
            thickness: Épaisseur du trait
            style: Style de la boîte
            
        Returns:
            Frame modifiée
        """
        x1, y1, x2, y2 = bbox
        
        if style == VisualizationStyle.DASHED:
            # Boîte en pointillés pour les tracks tentatives
            self._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        else:
            # Boîte solide pour les tracks confirmés
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        return frame
    
    def _draw_dashed_rectangle(self, frame: np.ndarray, pt1: Tuple[int, int], 
                              pt2: Tuple[int, int], color: Tuple[int, int, int], 
                              thickness: int, dash_length: int = 10):
        """
        Dessine un rectangle en pointillés.
        
        Args:
            frame: Frame à modifier
            pt1: Point supérieur gauche
            pt2: Point inférieur droit
            color: Couleur BGR
            thickness: Épaisseur du trait
            dash_length: Longueur des tirets
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Lignes horizontales
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Lignes verticales
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def _create_track_label(self, track: TrackInfo) -> str:
        """
        Crée le label d'un track selon le format spécifié.
        
        Args:
            track: Track à labelliser
            
        Returns:
            Label formaté: "ID:{track_id} {class} ({état})"
        """
        state_text = track.state.value
        return f"ID:{track.id} {track.class_name} ({state_text})"
    
    def _draw_label(self, frame: np.ndarray, label: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int], state: TrackState) -> np.ndarray:
        """
        Dessine un label avec fond coloré.
        
        Args:
            frame: Frame à modifier
            label: Texte du label
            position: Position (x, y) du coin supérieur gauche
            color: Couleur du track
            state: État du track
            
        Returns:
            Frame modifiée
        """
        x, y = position
        
        # Calculer la taille du texte
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.font_thickness
        )
        
        # Ajuster la position pour éviter de sortir de l'image
        label_y = max(text_height + 5, y - 5)
        
        # Couleur de fond selon l'état
        if state == TrackState.CONFIRMED:
            bg_color = color
            text_color = (255, 255, 255)  # Blanc
        else:
            # Pour les tentatives, utiliser une version plus transparente
            bg_color = tuple(int(float(c) * 0.7) for c in color)
            text_color = (255, 255, 255)
        
        # Dessiner le fond du label
        cv2.rectangle(
            frame,
            (x, label_y - text_height - 5),
            (x + text_width + 10, label_y + 5),
            bg_color,
            -1
        )
        
        # Dessiner le contour
        cv2.rectangle(
            frame,
            (x, label_y - text_height - 5),
            (x + text_width + 10, label_y + 5),
            (0, 0, 0),
            1
        )
        
        # Dessiner le texte
        cv2.putText(
            frame,
            label,
            (x + 5, label_y - 2),
            self.font,
            self.font_scale,
            text_color,
            self.font_thickness
        )
        
        return frame
    
    def _render_global_info(self, frame: np.ndarray, tracks: List[TrackInfo]) -> np.ndarray:
        """
        Ajoute les informations globales sur la frame.
        
        Args:
            frame: Frame à modifier
            tracks: Liste des tracks
            
        Returns:
            Frame modifiée
        """
        if not tracks:
            return frame
        
        # Compter les tracks par état
        confirmed_count = sum(1 for t in tracks if t.state == TrackState.CONFIRMED)
        tentative_count = sum(1 for t in tracks if t.state == TrackState.TENTATIVE)
        
        # Créer le texte d'info
        info_text = f"Tracks: {len(tracks)} ({confirmed_count} confirmed, {tentative_count} tentative)"
        
        # Position en haut à gauche
        position = (10, 30)
        
        # Style du texte d'info
        font_scale = self.font_scale * 0.8
        thickness = max(1, self.font_thickness - 1)
        
        # Calculer la taille du texte
        (text_width, text_height), _ = cv2.getTextSize(
            info_text, self.font, font_scale, thickness
        )
        
        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (position[0] - 5, position[1] - text_height - 5),
            (position[0] + text_width + 10, position[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Mélanger avec l'image originale
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Dessiner le texte
        cv2.putText(
            frame,
            info_text,
            position,
            self.font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        return frame
    
    def render_trajectory(self, frame: np.ndarray, trajectory: List[Tuple[float, float]], 
                         track_id: int, max_points: Optional[int] = None) -> np.ndarray:
        """
        Dessine la trajectoire d'un track sur la frame.
        
        Args:
            frame: Frame à modifier
            trajectory: Liste des points (x, y)
            track_id: ID du track
            max_points: Nombre maximum de points à afficher
            
        Returns:
            Frame modifiée
        """
        if len(trajectory) < 2:
            return frame
        
        # Limiter le nombre de points si spécifié
        if max_points and len(trajectory) > max_points:
            trajectory = trajectory[-max_points:]
        
        # Obtenir la couleur du track
        color = self.color_manager.get_track_color(track_id)
        
        # Dessiner les lignes de la trajectoire
        points = [(int(x), int(y)) for x, y in trajectory]
        
        for i in range(1, len(points)):
            # Alpha décroissant pour les points plus anciens
            alpha = i / len(points)
            
            # Couleur avec transparence
            line_color = tuple(int(float(c) * alpha + 50 * (1 - alpha)) for c in color)
            
            cv2.line(frame, points[i-1], points[i], line_color, 2)
        
        # Marquer le point actuel
        if points:
            cv2.circle(frame, points[-1], 4, color, -1)
        
        return frame
    
    def reset_colors(self):
        """Remet à zéro toutes les couleurs persistantes."""
        self.color_manager.reset()
        self.logger.info("Couleurs de tracking réinitialisées")
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Retourne la couleur d'un track (utile pour cohérence entre frames).
        
        Args:
            track_id: ID du track
            
        Returns:
            Couleur BGR
        """
        return self.color_manager.get_track_color(track_id)