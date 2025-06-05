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
from src.image_processor import ImageProcessor


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
        description="Détection et tracking de véhicules dans une vidéo ou image"
    )
    
    parser.add_argument(
        "input_file",
        help="Chemin vers la vidéo ou image d'entrée"
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
    
    parser.add_argument(
        "-png", "--image",
        action="store_true",
        help="Traiter une image au lieu d'une vidéo"
    )
    
    return parser.parse_args()


def validate_input(input_path: str, is_image: bool = False) -> bool:
    """
    Valide que le fichier d'entrée existe et est accessible.
    
    Args:
        input_path: Chemin vers le fichier d'entrée
        is_image: True si c'est une image, False pour vidéo
        
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
    
    # Vérifier l'extension selon le type
    if is_image:
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        type_name = "image"
    else:
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        type_name = "vidéo"
    
    if path.suffix.lower() not in valid_extensions:
        print(f"⚠️  Avertissement: Extension '{path.suffix}' non standard pour {type_name}")
        print(f"Extensions recommandées: {', '.join(valid_extensions)}")
    
    return True


def main():
    """Fonction principale."""
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer le logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Déterminer le mode de traitement
    is_image_mode = args.image
    file_type = "image" if is_image_mode else "vidéo"
    
    print(f"🚗 Système de Détection de Véhicules - Mode {file_type.upper()}")
    print("=" * 50)
    
    # Valider l'entrée
    if not validate_input(args.input_file, is_image_mode):
        sys.exit(1)
    
    try:
        # Afficher la configuration
        print(f"📁 Fichier d'entrée: {args.input_file}")
        print(f"🎯 Modèle YOLO: {config.YOLO_MODEL_PATH}")
        print(f"💻 Device: {config.DEVICE}")
        print(f"🎨 Dossier de sortie: {config.OUTPUT_DIR}")
        print("-" * 50)
        
        if is_image_mode:
            # Mode image
            processor = ImageProcessor(args.input_file, args.output)
            
            print("🚀 Début de la détection sur image...")
            success = processor.process_image()
            
            if success:
                print("✅ Détection terminée avec succès!")
                
                # Afficher les statistiques
                stats = processor.get_processing_stats()
                print(f"📁 Image d'entrée: {stats['input_path']}")
                print(f"💾 Image annotée: {stats['output_path']}")
                
            else:
                print("❌ Erreur lors de la détection")
                sys.exit(1)
        
        else:
            # Mode vidéo (comportement original)
            processor = VideoProcessor(args.input_file, args.output)
            
            print("🚀 Début du traitement vidéo...")
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