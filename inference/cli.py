"""
CLI for FerretNet inference.

Usage:
    python inference/cli.py single --image path/to/image.jpg --checkpoint path/to/ckpt
    python inference/cli.py batch --dir path/to/images/ --checkpoint path/to/ckpt --output results.csv
"""
import argparse
import csv
import os
from pathlib import Path
from inference.pipeline import FerretNetPipeline


def single_command(args):
    pipeline = FerretNetPipeline(
        checkpoint_path=args.checkpoint,
        retinaface_weights=args.retinaface_weights,
        device=args.device,
    )
    result = pipeline.predict(args.image)
    print(f"\nResult: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Fused score: {result['fused_score']:.4f}")
    print(f"Full image score: {result['full_score']:.4f}")
    print(f"Faces detected: {result['faces_detected']}")
    if result['face_scores']:
        print(f"Face scores: {[f'{s:.4f}' for s in result['face_scores']]}")


def batch_command(args):
    pipeline = FerretNetPipeline(
        checkpoint_path=args.checkpoint,
        retinaface_weights=args.retinaface_weights,
        device=args.device,
    )

    image_dir = Path(args.dir)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in extensions)

    print(f"Processing {len(images)} images...")
    results = []
    for img_path in images:
        try:
            result = pipeline.predict(str(img_path))
            results.append({
                'path': str(img_path),
                'label': result['label'],
                'confidence': result['confidence'],
                'fused_score': result['fused_score'],
                'full_score': result['full_score'],
                'faces_detected': result['faces_detected'],
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    output_path = args.output
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    n_ai = sum(1 for r in results if r['label'] == 'AI')
    n_real = sum(1 for r in results if r['label'] == 'Real')
    print(f"\nDone: {n_ai} AI, {n_real} Real out of {len(results)} images")
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FerretNet: AI vs Real Image Detector")
    subparsers = parser.add_subparsers(dest='command')

    # Single
    single_parser = subparsers.add_parser('single')
    single_parser.add_argument('--image', required=True)
    single_parser.add_argument('--checkpoint', required=True)
    single_parser.add_argument('--retinaface-weights', default='weights/Resnet50_Final.pth')
    single_parser.add_argument('--device', default='mps')

    # Batch
    batch_parser = subparsers.add_parser('batch')
    batch_parser.add_argument('--dir', required=True)
    batch_parser.add_argument('--checkpoint', required=True)
    batch_parser.add_argument('--retinaface-weights', default='weights/Resnet50_Final.pth')
    batch_parser.add_argument('--output', default='results/batch_results.csv')
    batch_parser.add_argument('--device', default='mps')

    args = parser.parse_args()
    if args.command == 'single':
        single_command(args)
    elif args.command == 'batch':
        batch_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
