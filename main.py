import argparse

from evaluation.evaluator import GNNEvaluator
from training.trainer import GNNTrainer
from utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="Node Classification on Cora Dataset")
    parser.add_argument("action", choices=["train", "eval"], help="Action to perform")

    args = parser.parse_args()

    if args.action == "train":
        trainer = GNNTrainer(Config)
        trainer.train()
    elif args.action == "eval":
        evaluator = GNNEvaluator(Config)
        evaluator.evaluate()
    else:
        print("Invalid action. Use --help for available actions.")


if __name__ == "__main__":
    main()
