import argparse
import itertools
import os
import subprocess

from evaluation.cross_dataset_evaluator import CrossDatasetEvaluator
from evaluation.evaluator import GNNEvaluator
from training.trainer import GNNTrainer
from utils.config import Config


def run_single_execution(args):
    execution_order = ["train", "evaluate", "cross_test"]
    actions = sorted(set(args.actions), key=lambda action: execution_order.index(action))

    trainer = None

    for action in actions:
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
            if not trainer:
                print("Error: Training must be performed before evaluation.")
                return
            print("\nStarting evaluation...")
            evaluator = GNNEvaluator(
                save_dir=trainer.save_dir,
                model_type=args.model_type,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                variant=args.variant,
                dropout=args.dropout,
                use_residual=args.use_residual,
                use_layer_norm=args.use_layer_norm
            )
            evaluator.evaluate()

        if action == "cross_test":
            if not trainer:
                print("Error: Training must be performed before cross-dataset testing.")
                return
            print("\nStarting cross-dataset testing...")
            config = Config()
            cross_evaluator = CrossDatasetEvaluator(
                save_dir=trainer.save_dir,
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


def run_experiments(args):
    hidden_dim_values = [64, 128, 256]
    num_layers_values = [8, 12]
    variant_values = ["gcn"]
    dropout_values = [0.3, 0.5, 0.7]
    use_residual_values = [True, False]
    use_layer_norm_values = [True, False]

    combinations = list(itertools.product(
        hidden_dim_values, num_layers_values, variant_values, dropout_values, use_residual_values, use_layer_norm_values
    ))

    for idx, (hidden_dim, num_layers, variant, dropout, use_residual, use_layer_norm) in enumerate(combinations,
                                                                                                   start=1):
        print(f"\nRunning experiment {idx}/{len(combinations)}:")
        command = [
            os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe"), "main.py", "single", "train", "evaluate",
            "cross_test",
            "--model_type", "generalized",
            "--hidden_dim", str(hidden_dim),
            "--num_layers", str(num_layers),
            "--variant", variant,
            "--dropout", str(dropout)
        ]
        if use_residual:
            command.append("--use_residual")
        if use_layer_norm:
            command.append("--use_layer_norm")

        subprocess.run(command, check=True)


def find_best_model(args):
    from find_best_models import find_best_models  # Import dynamically to avoid circular dependencies

    saved_models_dir = args.saved_models_dir
    best_pubmed, best_citeseer, best_cora = find_best_models(saved_models_dir)

    print("Best Model for PubMed Dataset:")
    print(f"  Model Name: {best_pubmed['name']}")
    print(f"  Accuracy: {best_pubmed['accuracy']:.4f}\n")

    print("Best Model for CiteSeer Dataset:")
    print(f"  Model Name: {best_citeseer['name']}")
    print(f"  Accuracy: {best_citeseer['accuracy']:.4f}\n")

    print("Best Model for Cora Dataset:")
    print(f"  Model Name: {best_cora['name']}")
    print(f"  Accuracy: {best_cora['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="GNN Training, Evaluation, Cross-Testing, and Experimentation")

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command")

    # Single execution parser
    single_parser = subparsers.add_parser("single", help="Run single execution (train, evaluate, cross_test)")
    single_parser.add_argument("actions", nargs="+", choices=["train", "evaluate", "cross_test"])
    single_parser.add_argument("--model_type", choices=["simple", "generalized"], default="simple")
    single_parser.add_argument("--hidden_dim", type=int, default=256)
    single_parser.add_argument("--num_layers", type=int, default=4)
    single_parser.add_argument("--variant", choices=["gcn", "sage", "gat"], default="gat")
    single_parser.add_argument("--dropout", type=float, default=0.5)
    single_parser.add_argument("--use_residual", action="store_true", default=True)
    single_parser.add_argument("--use_layer_norm", action="store_true", default=True)
    single_parser.add_argument("--model_path", type=str, default="models/gnn_model.pth")
    single_parser.add_argument("--default_handling", choices=["auto", "pca", "zero_pad", "replicate"], default="auto")

    # Experiment parser
    experiment_parser = subparsers.add_parser("experiment", help="Run batch experiments with different parameters")

    # Best model finder parser
    best_model_parser = subparsers.add_parser("find_best", help="Find the best model based on saved results")
    best_model_parser.add_argument("--saved_models_dir", type=str, default="saved_models",
                                   help="Path to the saved models directory")

    args = parser.parse_args()

    if args.command == "single":
        run_single_execution(args)
    elif args.command == "experiment":
        run_experiments(args)
    elif args.command == "find_best":
        find_best_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
