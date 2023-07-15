'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-07-15 14:17:00
'''
import argparse 
def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Tensorflow ViT(MoE) training on cifar',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log-dir', default='./logs',
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoint/',
                        help='checkpoint file format')
    parser.add_argument('--data-dir', default='./dataset/',
                        help='datasets for training and evaluating')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train')
    parser.add_argument('--device', choices=['cpu','cuda'], 
                        default='cuda',
                        help="machine device")
    parser.add_argument('--base-lr', type=float, default=3e-3,
                        help='learning rate for a single GPU')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.03,
                        help='weight decay')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    # model details
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--use_moe", action='store_true', default=False,
                        help='use ViT-MoE model')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "ViT-MoE-B"],
                        default="ViT-B_16",
                        help="Which model to use.")
    args = parser.parse_args()
    return args