"""
Master entry point for the LEGO Brick Segmentation and Classification Project.
Provides CLI arguments to execute individual stages or the entire end-to-end pipeline.
"""
import os
import sys
import argparse
import random
import config
from preprocessing.dataset_explorer import explore_dataset
from preprocessing.data_loader import create_or_load_splits
from segmentation.segmentation_evaluator import evaluate_segmentation
from features.feature_extractor import prepare_and_cache_features
from evaluation.benchmark import run_full_benchmark
from pipeline import LEGODetectionPipeline

def main():
    parser = argparse.ArgumentParser(description="Proyecto LEGO: Segmentación y Clasificación con ML y DL")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "explore", "segment", "features", "train", "inference"],
        help="Etapa a ejecutar: 'all', 'explore', 'segment', 'features', 'train', 'inference'"
    )
    parser.add_argument(
        "--num_infer_samples",
        type=int,
        default=6,
        help="Cantidad de imágenes base para inferencia demostrativa"
    )
    args = parser.parse_args()
    
    print("\n" + "#" * 70)
    print("PROYECTO: SEGMENTACIÓN Y CLASIFICACIÓN DE PIEZAS LEGO")
    print("Autores: Sistema de Machine Learning & Deep Learning")
    print(f"Modo seleccionado: {args.stage.upper()}")
    print("#" * 70 + "\n")
    
    if args.stage in ["all", "explore"]:
        explore_dataset()
        
    if args.stage in ["all", "segment"]:
        evaluate_segmentation(num_samples=35)
        
    if args.stage in ["all", "features"]:
        prepare_and_cache_features()
        
    if args.stage in ["all", "train"]:
        run_full_benchmark()
        
    if args.stage in ["all", "inference"]:
        print("\n" + "=" * 60)
        print("INFERENCIA END-TO-END CON EL PIPELINE INTEGRADO")
        print("=" * 60)
        
        pipeline = LEGODetectionPipeline(model_type="cnn")
        
        # Select test Base Images from distinct categories
        base_classes = [d for d in os.listdir(config.BASE_IMAGES_DIR) if os.path.isdir(os.path.join(config.BASE_IMAGES_DIR, d))]
        random.seed(config.RANDOM_SEED)
        sampled_classes = random.sample(base_classes, min(args.num_infer_samples, len(base_classes)))
        
        for cls_name in sampled_classes:
            cls_dir = os.path.join(config.BASE_IMAGES_DIR, cls_name)
            imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith('.jpg')]
            if imgs:
                sample_path = os.path.join(cls_dir, random.choice(imgs))
                res = pipeline.process_image(sample_path, segmentation_method="hsv_otsu")
                print(f"Imagen: {os.path.basename(sample_path)} | Piezas detectadas: {res['total_detected']}")
                for p in res["pieces"]:
                    print(f"  -> Pieza #{p['piece_id']}: Clase Predicha = '{p['class_name']}' | Confianza = {p['confidence']*100:.1f}%")

if __name__ == "__main__":
    main()

