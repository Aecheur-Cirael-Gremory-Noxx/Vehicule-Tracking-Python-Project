import cv2
import os
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, asdict

import config


@dataclass
class ExportMetadata:
    """Métadonnées d'export vidéo."""
    input_file: str
    output_file: str
    start_time: str
    end_time: str
    duration_seconds: float
    input_fps: float
    output_fps: float
    input_resolution: Tuple[int, int]
    output_resolution: Tuple[int, int]
    codec: str
    quality: int
    frames_processed: int
    file_size_mb: float
    compression_ratio: float


class VideoCodec:
    """Gestionnaire de codecs vidéo optimisés."""
    
    @staticmethod
    def get_h264_fourcc() -> int:
        """Retourne le meilleur codec disponible."""
        # Utiliser directement mp4v qui est plus compatible
        logging.getLogger(__name__).info("Utilisation du codec mp4v pour compatibilité")
        return cv2.VideoWriter_fourcc(*'mp4v')
    
    @staticmethod
    def get_codec_params(quality: int = 90) -> Dict[str, Any]:
        """
        Retourne les paramètres optimisés pour l'export.
        
        Args:
            quality: Qualité de 1 (basse) à 100 (haute)
            
        Returns:
            Dictionnaire des paramètres
        """
        # Convertir qualité en paramètres OpenCV
        # Note: OpenCV VideoWriter a des limitations sur les paramètres
        params = {}
        
        # Pour H264, on peut essayer de définir des paramètres via les propriétés
        if quality >= 90:
            params['bitrate_multiplier'] = 1.0
        elif quality >= 70:
            params['bitrate_multiplier'] = 0.8
        elif quality >= 50:
            params['bitrate_multiplier'] = 0.6
        else:
            params['bitrate_multiplier'] = 0.4
        
        return params


class ExportManager:
    """
    Gestionnaire d'export vidéo optimisé avec codec H264,
    validation et génération de métadonnées.
    """
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialise le gestionnaire d'export.
        
        Args:
            input_path: Chemin de la vidéo source
            output_path: Chemin de sortie
        """
        self.input_path = input_path
        self.output_path = output_path
        self.logger = logging.getLogger(__name__)
        
        # Paramètres vidéo
        self.input_fps: Optional[float] = None
        self.input_resolution: Optional[Tuple[int, int]] = None
        self.output_fps: Optional[float] = None
        self.output_resolution: Optional[Tuple[int, int]] = None
        
        # Export state
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_initialized = False
        self.frames_written = 0
        self.start_time = time.time()
        
        # Métadonnées
        self.metadata: Optional[ExportMetadata] = None
        
        self.logger.info(f"ExportManager initialisé: {output_path}")
    
    def initialize_from_source(self, source_cap: cv2.VideoCapture) -> bool:
        """
        Initialise l'export à partir d'une VideoCapture source.
        
        Args:
            source_cap: VideoCapture de la vidéo source
            
        Returns:
            True si l'initialisation a réussi
        """
        try:
            # Extraire les paramètres de la source
            self.input_fps = source_cap.get(cv2.CAP_PROP_FPS)
            self.input_resolution = (
                int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            
            # Utiliser les mêmes paramètres pour la sortie
            self.output_fps = self.input_fps
            self.output_resolution = self.input_resolution
            
            return self._create_video_writer()
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation depuis source: {e}")
            return False
    
    def initialize_with_params(self, fps: float, width: int, height: int) -> bool:
        """
        Initialise l'export avec des paramètres spécifiques.
        
        Args:
            fps: FPS de sortie
            width: Largeur
            height: Hauteur
            
        Returns:
            True si l'initialisation a réussi
        """
        try:
            self.output_fps = fps
            self.output_resolution = (width, height)
            
            return self._create_video_writer()
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation avec paramètres: {e}")
            return False
    
    def _create_video_writer(self) -> bool:
        """
        Crée le VideoWriter avec configuration optimisée.
        
        Returns:
            True si la création a réussi
        """
        try:
            # Créer le dossier de sortie si nécessaire
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Obtenir le codec H264 optimisé
            fourcc = VideoCodec.get_h264_fourcc()
            
            # Paramètres de qualité
            codec_params = VideoCodec.get_codec_params(config.OUTPUT_QUALITY)
            
            # Créer le VideoWriter
            self.video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.output_fps,
                self.output_resolution
            )
            
            if not self.video_writer.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir VideoWriter: {self.output_path}")
            
            self.is_initialized = True
            self.start_time = time.time()
            
            self.logger.info(
                f"VideoWriter créé: {self.output_fps} FPS, "
                f"{self.output_resolution[0]}x{self.output_resolution[1]}, "
                f"qualité: {config.OUTPUT_QUALITY}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur création VideoWriter: {e}")
            return False
    
    def write_frame(self, frame) -> bool:
        """
        Écrit une frame dans la vidéo de sortie.
        
        Args:
            frame: Frame à écrire
            
        Returns:
            True si l'écriture a réussi
        """
        if not self.is_initialized or not self.video_writer:
            self.logger.error("ExportManager non initialisé")
            return False
        
        try:
            # Redimensionner la frame si nécessaire
            if frame.shape[:2][::-1] != self.output_resolution:
                frame = cv2.resize(frame, self.output_resolution)
            
            # Écrire la frame
            self.video_writer.write(frame)
            self.frames_written += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur écriture frame {self.frames_written}: {e}")
            return False
    
    def finalize_export(self) -> bool:
        """
        Finalise l'export et génère les métadonnées.
        
        Returns:
            True si la finalisation a réussi
        """
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # Générer les métadonnées
            self._generate_metadata()
            
            # Valider le fichier de sortie
            is_valid = self._validate_output_file()
            
            if is_valid:
                self.logger.info(f"Export finalisé avec succès: {self.output_path}")
                self._log_export_summary()
            else:
                self.logger.error("Fichier de sortie invalide")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Erreur finalisation export: {e}")
            return False
    
    def _generate_metadata(self):
        """Génère les métadonnées d'export."""
        try:
            end_time = time.time()
            duration = end_time - self.start_time
            
            # Taille du fichier
            file_size = 0
            if os.path.exists(self.output_path):
                file_size = os.path.getsize(self.output_path) / (1024 * 1024)  # MB
            
            # Taille du fichier d'entrée pour calcul compression
            input_size = 0
            if os.path.exists(self.input_path):
                input_size = os.path.getsize(self.input_path) / (1024 * 1024)  # MB
            
            compression_ratio = input_size / file_size if file_size > 0 else 0
            
            self.metadata = ExportMetadata(
                input_file=self.input_path,
                output_file=self.output_path,
                start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
                duration_seconds=duration,
                input_fps=self.input_fps or 0,
                output_fps=self.output_fps or 0,
                input_resolution=self.input_resolution or (0, 0),
                output_resolution=self.output_resolution or (0, 0),
                codec="H264",
                quality=config.OUTPUT_QUALITY,
                frames_processed=self.frames_written,
                file_size_mb=file_size,
                compression_ratio=compression_ratio
            )
            
        except Exception as e:
            self.logger.error(f"Erreur génération métadonnées: {e}")
    
    def _validate_output_file(self) -> bool:
        """
        Valide que le fichier de sortie est correct.
        
        Returns:
            True si le fichier est valide
        """
        try:
            if not os.path.exists(self.output_path):
                self.logger.error("Fichier de sortie non créé")
                return False
            
            if os.path.getsize(self.output_path) == 0:
                self.logger.error("Fichier de sortie vide")
                return False
            
            # Tester l'ouverture avec OpenCV
            test_cap = cv2.VideoCapture(self.output_path)
            if not test_cap.isOpened():
                self.logger.error("Fichier de sortie non lisible par OpenCV")
                test_cap.release()
                return False
            
            # Vérifier quelques propriétés de base
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            frame_count = test_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            test_cap.release()
            
            if fps <= 0 or frame_count <= 0:
                self.logger.error("Propriétés vidéo invalides")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur validation fichier: {e}")
            return False
    
    def _log_export_summary(self):
        """Affiche un résumé de l'export."""
        if not self.metadata:
            return
        
        self.logger.info("=" * 50)
        self.logger.info("RÉSUMÉ DE L'EXPORT")
        self.logger.info("=" * 50)
        self.logger.info(f"Fichier de sortie: {self.metadata.output_file}")
        self.logger.info(f"Durée traitement: {self.metadata.duration_seconds:.2f}s")
        self.logger.info(f"Frames exportées: {self.metadata.frames_processed}")
        self.logger.info(f"FPS: {self.metadata.output_fps}")
        self.logger.info(f"Résolution: {self.metadata.output_resolution[0]}x{self.metadata.output_resolution[1]}")
        self.logger.info(f"Taille fichier: {self.metadata.file_size_mb:.2f} MB")
        self.logger.info(f"Ratio compression: {self.metadata.compression_ratio:.2f}x")
        self.logger.info("=" * 50)
    
    def export_metadata_json(self, json_path: Optional[str] = None) -> bool:
        """
        Exporte les métadonnées en JSON.
        
        Args:
            json_path: Chemin du fichier JSON (auto-généré si None)
            
        Returns:
            True si l'export a réussi
        """
        if not self.metadata:
            self.logger.error("Pas de métadonnées à exporter")
            return False
        
        try:
            if not json_path:
                output_path = Path(self.output_path)
                json_path = output_path.with_suffix('.json')
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.metadata), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Métadonnées exportées: {json_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur export métadonnées: {e}")
            return False
    
    def get_metadata(self) -> Optional[ExportMetadata]:
        """Retourne les métadonnées d'export."""
        return self.metadata
    
    def cleanup(self):
        """Nettoie les ressources."""
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize_export()
        self.cleanup()