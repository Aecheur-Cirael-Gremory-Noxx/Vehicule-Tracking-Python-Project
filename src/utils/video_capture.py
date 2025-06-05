import cv2
import os
import logging
from typing import Optional, Tuple

class VideoCapture:
    """
    Wrapper autour d'OpenCV VideoCapture pour gérer la lecture vidéo
    avec extraction de métadonnées et gestion d'erreurs robuste.
    """
    
    def __init__(self, video_path: str):
        """
        Initialise la capture vidéo avec vérifications et extraction des métadonnées.
        
        Args:
            video_path: Chemin vers le fichier vidéo
            
        Raises:
            FileNotFoundError: Si le fichier vidéo n'existe pas
            ValueError: Si le fichier n'est pas une vidéo valide
        """
        self.video_path = video_path
        self.cap = None
        self.current_frame = 0
        
        # Vérifier l'existence du fichier
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Le fichier vidéo n'existe pas : {video_path}")
        
        # Initialiser la capture
        self.cap = cv2.VideoCapture(video_path)
        
        # Vérifier que la vidéo peut être ouverte
        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")
        
        # Extraire les métadonnées
        self._extract_metadata()
        
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Vidéo chargée: {video_path}")
        self.logger.info(f"Métadonnées: {self.fps} FPS, {self.width}x{self.height}, "
                        f"{self.total_frames} frames, {self.duration:.2f}s")
    
    def _extract_metadata(self):
        """Extrait et stocke les métadonnées de la vidéo."""
        # FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0  # Valeur par défaut
            self.logger.warning("FPS invalide détecté, utilisation de 30 FPS par défaut")
        
        # Résolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Nombre total de frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Durée en secondes
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Codec (FOURCC)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Lit la frame suivante avec gestion d'erreur.
        
        Returns:
            Tuple (success, frame) où success indique si la lecture a réussi
        """
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Tentative de lecture sur une vidéo fermée")
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame += 1
        else:
            if self.current_frame < self.total_frames:
                self.logger.warning(f"Frame manquante détectée à la position {self.current_frame}")
        
        return ret, frame
    
    def skip_frames(self, n: int) -> bool:
        """
        Avance de N frames sans les lire.
        
        Args:
            n: Nombre de frames à ignorer
            
        Returns:
            True si réussi, False sinon
        """
        if not self.cap or not self.cap.isOpened():
            return False
        
        target_frame = self.current_frame + n
        if target_frame >= self.total_frames:
            self.logger.warning(f"Tentative de skip au-delà de la fin de la vidéo")
            return False
        
        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        if success:
            self.current_frame = target_frame
        else:
            self.logger.error(f"Impossible de skip à la frame {target_frame}")
        
        return success
    
    def seek_to_frame(self, frame_num: int) -> bool:
        """
        Va directement à une frame spécifique.
        
        Args:
            frame_num: Numéro de la frame cible (0-indexé)
            
        Returns:
            True si réussi, False sinon
        """
        if not self.cap or not self.cap.isOpened():
            return False
        
        if frame_num < 0 or frame_num >= self.total_frames:
            self.logger.warning(f"Numéro de frame invalide: {frame_num}")
            return False
        
        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        if success:
            self.current_frame = frame_num
        else:
            self.logger.error(f"Impossible d'aller à la frame {frame_num}")
        
        return success
    
    def seek_to_time(self, time_seconds: float) -> bool:
        """
        Va à un temps spécifique dans la vidéo.
        
        Args:
            time_seconds: Temps en secondes
            
        Returns:
            True si réussi, False sinon
        """
        if time_seconds < 0 or time_seconds > self.duration:
            self.logger.warning(f"Temps invalide: {time_seconds}s")
            return False
        
        frame_num = int(time_seconds * self.fps)
        return self.seek_to_frame(frame_num)
    
    def get_progress(self) -> float:
        """
        Retourne le pourcentage de progression de la lecture.
        
        Returns:
            Pourcentage de progression (0.0 à 100.0)
        """
        if self.total_frames == 0:
            return 0.0
        return (self.current_frame / self.total_frames) * 100.0
    
    def get_current_time(self) -> float:
        """
        Retourne le temps actuel en secondes.
        
        Returns:
            Temps actuel en secondes
        """
        return self.current_frame / self.fps if self.fps > 0 else 0.0
    
    def reset(self) -> bool:
        """
        Retourne au début de la vidéo.
        
        Returns:
            True si réussi, False sinon
        """
        return self.seek_to_frame(0)
    
    def is_opened(self) -> bool:
        """
        Vérifie si la vidéo est ouverte et lisible.
        
        Returns:
            True si la vidéo est ouverte, False sinon
        """
        return self.cap is not None and self.cap.isOpened()
    
    def get_metadata(self) -> dict:
        """
        Retourne un dictionnaire avec toutes les métadonnées.
        
        Returns:
            Dictionnaire contenant les métadonnées de la vidéo
        """
        return {
            'path': self.video_path,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'codec': self.codec,
            'current_frame': self.current_frame,
            'current_time': self.get_current_time(),
            'progress': self.get_progress()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit avec nettoyage automatique."""
        self.release()
    
    def release(self):
        """Libère les ressources de la capture vidéo."""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.logger.info(f"Ressources vidéo libérées pour {self.video_path}")
    
    def __del__(self):
        """Destructeur pour s'assurer que les ressources sont libérées."""
        self.release()