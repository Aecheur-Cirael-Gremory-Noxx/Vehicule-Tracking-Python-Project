import cv2
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

import config
from .utils import VideoCapture, VisualizationRenderer, ExportManager
from .detection import VehicleDetectorTracker
from .tracking import TrackManager, TrajectoryStorage, TrajectoryExtractor


class VideoProcessor:
    """
    Classe principale orchestrant le pipeline complet de détection et tracking
    de véhicules : lecture vidéo → détection YOLO → tracking DeepSORT → visualisation.
    """
    
    def __init__(self, input_video_path: str, output_video_path: Optional[str] = None):
        """
        Initialise le processeur vidéo.
        
        Args:
            input_video_path: Chemin vers la vidéo d'entrée
            output_video_path: Chemin de sortie (optionnel, généré automatiquement si None)
        """
        self.input_path = input_video_path
        self.output_path = output_video_path or self._generate_output_path(input_video_path)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialiser les composants
        self.video_capture: Optional[VideoCapture] = None
        self.detector_tracker: Optional[VehicleDetectorTracker] = None
        self.track_manager: Optional[TrackManager] = None
        self.trajectory_storage: Optional[TrajectoryStorage] = None
        self.visualizer: Optional[VisualizationRenderer] = None
        self.export_manager: Optional[ExportManager] = None
        
        # Statistiques de traitement
        self.frame_count = 0
        self.processed_frames = 0
        self.start_time = None
        
        self.logger.info(f"VideoProcessor initialisé: {input_video_path} → {self.output_path}")
    
    def _generate_output_path(self, input_path: str) -> str:
        """Génère automatiquement le chemin de sortie."""
        input_file = Path(input_path)
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_file.stem}_tracked_{timestamp}.mp4"
        
        return str(output_dir / output_filename)
    
    def initialize_components(self) -> bool:
        """
        Initialise tous les composants du pipeline.
        
        Returns:
            True si l'initialisation a réussi
        """
        try:
            # 1. Capture vidéo
            self.video_capture = VideoCapture(self.input_path)
            self.logger.info(f"Vidéo chargée: {self.video_capture.fps} FPS, "
                           f"{self.video_capture.width}x{self.video_capture.height}")
            
            # 2. Détecteur-tracker
            self.detector_tracker = VehicleDetectorTracker()
            
            # 3. Gestionnaire de tracks
            self.track_manager = TrackManager()
            
            # 4. Stockage des trajectoires
            self.trajectory_storage = TrajectoryStorage()
            
            # 5. Visualisateur
            self.visualizer = VisualizationRenderer()
            
            # 6. Export manager
            self._initialize_export_manager()
            
            self.logger.info("Tous les composants initialisés avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            return False
    
    def _initialize_export_manager(self):
        """Initialise l'export manager avec les mêmes paramètres que l'entrée."""
        if not self.video_capture:
            raise RuntimeError("VideoCapture doit être initialisé avant ExportManager")
        
        # Créer l'export manager
        self.export_manager = ExportManager(self.input_path, self.output_path)
        
        # Initialiser avec les paramètres de la vidéo source
        success = self.export_manager.initialize_from_source(self.video_capture.cap)
        
        if not success:
            raise RuntimeError(f"Impossible d'initialiser l'export manager: {self.output_path}")
        
        self.logger.info(f"ExportManager initialisé: {self.video_capture.fps} FPS, "
                        f"{self.video_capture.width}x{self.video_capture.height}")
    
    def process_video(self, display_progress: bool = True) -> bool:
        """
        Traite la vidéo complète frame par frame.
        
        Args:
            display_progress: Afficher la progression
            
        Returns:
            True si le traitement a réussi
        """
        if not self.initialize_components():
            return False
        
        try:
            self.start_time = time.time()
            self.frame_count = 0
            
            self.logger.info("Début du traitement vidéo...")
            
            while True:
                # Lire la frame suivante
                success, frame = self.video_capture.read_frame()
                if not success:
                    break
                
                self.frame_count += 1
                
                # Appliquer le frame skip si configuré
                if config.FRAME_SKIP > 1 and self.frame_count % config.FRAME_SKIP != 0:
                    continue
                
                # Traiter la frame
                annotated_frame = self._process_single_frame(frame, self.frame_count)
                
                # Écrire la frame annotée
                if annotated_frame is not None and self.export_manager:
                    success = self.export_manager.write_frame(annotated_frame)
                    if success:
                        self.processed_frames += 1
                
                # Afficher la progression
                if display_progress and self.frame_count % 30 == 0:
                    self._display_progress()
            
            # Finaliser
            self._finalize_processing()
            
            self.logger.info(f"Traitement terminé: {self.processed_frames} frames traitées")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement: {e}")
            return False
        
        finally:
            self._cleanup()
    
    def _process_single_frame(self, frame: np.ndarray, frame_number: int) -> Optional[np.ndarray]:
        """
        Traite une frame individuelle à travers tout le pipeline.
        
        Args:
            frame: Frame à traiter
            frame_number: Numéro de la frame
            
        Returns:
            Frame annotée ou None en cas d'erreur
        """
        try:
            # 1. Détection et tracking avec YOLO + DeepSORT (includes annotation)
            raw_tracks, annotated_frame = self.detector_tracker.process_frame(frame)
            
            # 2. Gestion des tracks (filtrage confirmés/tentatives) pour stats
            timestamp = time.time()
            confirmed_tracks, tentative_tracks = self.track_manager.process_tracks(
                raw_tracks, frame_number
            )
            
            # 3. Stockage des trajectoires
            all_tracks_data = []
            for track in raw_tracks:
                track_data = TrajectoryExtractor.extract_track_data(
                    track, frame_number, timestamp
                )
                if track_data:
                    all_tracks_data.append(track_data)
            
            self.trajectory_storage.process_tracks_batch(all_tracks_data, timestamp)
            
            # 4. Optionnel: dessiner les trajectoires sur l'image déjà annotée
            if config.DRAW_TRAJECTORIES:
                for track_info in confirmed_tracks:
                    trajectory = self.trajectory_storage.get_trajectory_centers(track_info.id)
                    if len(trajectory) > 1:
                        annotated_frame = self.visualizer.render_trajectory(
                            annotated_frame, trajectory, track_info.id, config.TRAJECTORY_LENGTH
                        )
            
            return annotated_frame
            
        except Exception as e:
            import traceback
            self.logger.error(f"Erreur traitement frame {frame_number}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return frame  # Retourner frame originale en cas d'erreur
    
    def _display_progress(self):
        """Affiche la progression du traitement."""
        if not self.start_time:
            return
        
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        progress = self.video_capture.get_progress()
        
        # Statistiques des tracks
        stats = self.track_manager.get_statistics()
        
        self.logger.info(
            f"Frame {self.frame_count} | "
            f"Progress: {progress:.1f}% | "
            f"FPS: {fps:.1f} | "
            f"Tracks: {stats['confirmed_tracks']}/{stats['active_tracks']}"
        )
    
    def _finalize_processing(self):
        """Finalise le traitement et génère les statistiques finales."""
        if not self.start_time:
            return
        
        # Finaliser l'export
        if self.export_manager:
            export_success = self.export_manager.finalize_export()
            if export_success:
                # Exporter les métadonnées JSON
                self.export_manager.export_metadata_json()
        
        total_time = time.time() - self.start_time
        avg_fps = self.processed_frames / total_time if total_time > 0 else 0
        
        # Statistiques finales
        track_stats = self.track_manager.get_statistics() if self.track_manager else {}
        storage_stats = self.trajectory_storage.get_storage_statistics() if self.trajectory_storage else {}
        
        self.logger.info("=" * 50)
        self.logger.info("STATISTIQUES FINALES")
        self.logger.info("=" * 50)
        self.logger.info(f"Temps total: {total_time:.2f}s")
        self.logger.info(f"Frames traitées: {self.processed_frames}")
        self.logger.info(f"FPS moyen: {avg_fps:.1f}")
        self.logger.info(f"Tracks créés: {track_stats.get('total_tracks_created', 0)}")
        self.logger.info(f"Points de trajectoire: {storage_stats.get('total_points_stored', 0)}")
        self.logger.info(f"Vidéo sauvegardée: {self.output_path}")
        self.logger.info("=" * 50)
    
    def _cleanup(self):
        """Nettoie les ressources."""
        try:
            if self.export_manager:
                self.export_manager.cleanup()
            if self.video_capture:
                self.video_capture.release()
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {e}")
    
    def get_processing_stats(self) -> dict:
        """
        Retourne les statistiques du traitement.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        stats = {
            'input_path': self.input_path,
            'output_path': self.output_path,
            'frames_processed': self.processed_frames,
            'total_frames': self.frame_count,
        }
        
        if self.track_manager:
            stats.update(self.track_manager.get_statistics())
        
        if self.trajectory_storage:
            stats.update(self.trajectory_storage.get_storage_statistics())
        
        if self.export_manager and self.export_manager.get_metadata():
            export_metadata = self.export_manager.get_metadata()
            stats.update({
                'export_file_size_mb': export_metadata.file_size_mb,
                'export_compression_ratio': export_metadata.compression_ratio,
                'export_codec': export_metadata.codec,
                'export_quality': export_metadata.quality
            })
        
        return stats