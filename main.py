import argparse

from evaluation.cross_dataset_evaluator import CrossDatasetEvaluator
from evaluation.evaluator import GNNEvaluator
from training.trainer import GNNTrainer
from utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="GNN Training and Cross-Dataset Evaluation")

    # Add positional arguments for actions
    parser.add_argument(
        "actions",
        nargs="+",
        choices=["train", "evaluate", "cross_test"],
        help="Specify the actions to perform: train, evaluate, cross_test"
    )

    # Model type for training and evaluation
    parser.add_argument(
        "--model_type",
        choices=["simple", "generalized"],
        default="simple",
        help="Type of GNN to use: 'simple' (default) or 'generalized'",
    )

    # Model architecture arguments for Generalized GNN
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size (default: 64)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers (default: 2)")
    parser.add_argument("--variant", choices=["gcn", "sage", "gat"], default="gcn", help="GNN variant (default: 'gcn')")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
    parser.add_argument("--use_residual", action="store_true", help="Enable residual connections")
    parser.add_argument("--use_layer_norm", action="store_true", help="Enable layer normalization")

    # Path to the model file
    parser.add_argument("--model_path", type=str, default="models/gnn_model.pth", help="Path to the saved model")

    # Cross-dataset handling argument
    parser.add_argument(
        "--default_handling",
        choices=["auto", "pca", "zero_pad", "replicate"],
        default="auto",
        help="Default method to handle feature dimension mismatches (default: auto)",
    )

    args = parser.parse_args()

    execution_order = ["train", "evaluate", "cross_test"]
    actions = sorted(set(args.actions), key=lambda action: execution_order.index(action))

    for action in actions:
        if action == "train":
            if action == "train":
                trainer = GNNTrainer(
                    model_type=args.model_type,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    variant=args.variant,
                    dropout=args.dropout,
                    use_residual=args.use_residual,
                    use_layer_norm=args.use_layer_norm
                )
                trainer.train()

        if action == "evaluate":
            print("\nStarting evaluation...")
            evaluator = GNNEvaluator(
                model_type=args.model_type,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                variant=args.variant,
                dropout=args.dropout,
                use_residual=args.use_residual,
                use_layer_norm=args.use_layer_norm,
                model_path=args.model_path
            )
            evaluator.evaluate()

        if action == "cross_test":
            print("\nStarting cross-dataset testing...")
            config = Config()
            cross_evaluator = CrossDatasetEvaluator(
                config=config,
                model_type=args.model_type,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                variant=args.variant,
                dropout=args.dropout,
                use_residual=args.use_residual,
                use_layer_norm=args.use_layer_norm,
                model_path=args.model_path,
                default_handling=args.default_handling
            )
            cross_evaluator.evaluate()


if __name__ == "__main__":
    main()
