import argparse

from evaluation.cross_dataset_evaluator import CrossDatasetEvaluator
from evaluation.evaluator import GNNEvaluator
from training.trainer import GNNTrainer
from utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="Node Classification and Cross-Dataset Testing")
    parser.add_argument(
        "action",
        choices=["train", "evaluate", "cross_test"],
        help="Action to perform",
    )

    args = parser.parse_args()

    if args.action == "train":
        trainer = GNNTrainer(Config)
        trainer.train()
    elif args.action == "evaluate":
        evaluator = GNNEvaluator(Config)
        evaluator.evaluate()
    elif args.action == "cross_test":
        cross_evaluator = CrossDatasetEvaluator(Config)
        cross_evaluator.evaluate()
    else:
        print("Invalid action. Use --help for available actions.")


if __name__ == "__main__":
    main()
