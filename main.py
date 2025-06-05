#!/usr/bin/env python3
"""
Pipeline principal pour la détection et le tracking de véhicules.
Prototype de surveillance routière avec YOLO + DeepSORT.
"""

import sys
import logging
import argparse
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from src.video_processor import VideoProcessor


def setup_logging(verbose: bool = False):
    """Configure le système de logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vehicle_tracking.log')
        ]
    )


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Détection et tracking de véhicules dans une vidéo"
    )
    
    parser.add_argument(
        "input_video",
        help="Chemin vers la vidéo d'entrée"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Chemin de sortie (optionnel, généré automatiquement si non spécifié)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mode verbose (debug)"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Désactiver l'affichage de progression"
    )
    
    return parser.parse_args()


def validate_input(input_path: str) -> bool:
    """
    Valide que le fichier d'entrée existe et est accessible.
    
    Args:
        input_path: Chemin vers le fichier d'entrée
        
    Returns:
        True si valide
    """
    path = Path(input_path)
    
    if not path.exists():
        print(f"❌ Erreur: Le fichier '{input_path}' n'existe pas")
        return False
    
    if not path.is_file():
        print(f"❌ Erreur: '{input_path}' n'est pas un fichier")
        return False
    
    # Vérifier l'extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    if path.suffix.lower() not in valid_extensions:
        print(f"⚠️  Avertissement: Extension '{path.suffix}' non standard")
        print(f"Extensions recommandées: {', '.join(valid_extensions)}")
    
    return True


def main():
    """Fonction principale."""
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer le logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print("🚗 Système de Détection et Tracking de Véhicules")
    print("=" * 50)
    
    # Valider l'entrée
    if not validate_input(args.input_video):
        sys.exit(1)
    
    try:
        # Afficher la configuration
        print(f"📹 Vidéo d'entrée: {args.input_video}")
        print(f"🎯 Modèle YOLO: {config.YOLO_MODEL_PATH}")
        print(f"💻 Device: {config.DEVICE}")
        print(f"🎨 Dossier de sortie: {config.OUTPUT_DIR}")
        print("-" * 50)
        
        # Initialiser le processeur
        processor = VideoProcessor(args.input_video, args.output)
        
        # Traiter la vidéo
        print("🚀 Début du traitement...")
        success = processor.process_video(display_progress=not args.no_progress)
        
        if success:
            print("✅ Traitement terminé avec succès!")
            
            # Afficher les statistiques
            stats = processor.get_processing_stats()
            print(f"📊 Frames traitées: {stats['frames_processed']}")
            print(f"🎯 Tracks créés: {stats.get('total_tracks_created', 0)}")
            print(f"💾 Vidéo sauvegardée: {stats['output_path']}")
            
        else:
            print("❌ Erreur lors du traitement")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Traitement interrompu par l'utilisateur")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()