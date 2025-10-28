import argparse
from src.preprocessing import Preprocessing
from src.models import UNet, GAN, Diffusion
from src.training import (
    train_UNet, train_GAN, train_diffusion
)
from src.evaluation import Evaluation

def main():
    parser = argparse.ArgumentParser(description="Cell Channel Generation Pipeline")
    parser.add_argument("--model", type=str, choices=["unet", "gan", "diffusion"], required=True,
                        help="Choose the model to train or evaluate.")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train",
                        help="Choose whether to train or evaluate the model.")
    args = parser.parse_args()

    # Preprocessing
    print("🔹 Preprocessing data...")
    Preprocessing.run()

    # Model selection
    if args.model == "unet":
        model = UNet()
        if args.mode == "train":
            train_UNet.train(model)
    elif args.model == "gan":
        model = GAN()
        if args.mode == "train":
            train_GAN.train(model)
    elif args.model == "diffusion":
        model = Diffusion()
        if args.mode == "train":
            train_diffusion.train(model)

    # Evaluation
    if args.mode == "eval":
        print("🔹 Running evaluation...")
        Evaluation.evaluate(model)

if __name__ == "__main__":
    main()
