import argparse
import itertools
import os
import subprocess

from data.dataset_loader import DatasetLoader
from embeddings.analyze_embeddings import extract_embeddings
from embeddings.visualize_embeddings import visualize_embeddings
from evaluation.cross_dataset_evaluator import CrossDatasetEvaluator
from evaluation.evaluator import GNNEvaluator
from models.gnn_explainer import GNNExplainability
from models.node2vec_model import Node2VecModel
from training.trainer import GNNTrainer
from training.trainer_link_prediction import EdgePredictionTrainer
from utils.config import Config
from utils.find_best_models import find_best_models
from utils.load_generalized_gnn import load_generalized_gnn
from utils.prepare_link_prediction_data import prepare_link_prediction_data


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


def run_link_prediction(args):
    """
    Run link prediction training and evaluation using EdgePredictionTrainer.
    """
    print("\nStarting link prediction...")

    # Load the dataset
    dataset = DatasetLoader(args.dataset).load()

    # Define model configuration
    model_config = {
        "input_dim": dataset.num_features,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "num_layers": args.num_layers,
        "variant": args.variant,
        "dropout": args.dropout,
        "use_residual": args.use_residual,
        "use_layer_norm": args.use_layer_norm,
    }

    # Initialize the trainer
    trainer = EdgePredictionTrainer(dataset, model_config, lr=0.01, weight_decay=5e-4, val_split=0.1)

    # Perform actions: train and/or evaluate
    if "train" in args.actions:
        print("\nStarting training...")
        trainer.train(epochs=args.epochs)

    if "evaluate" in args.actions:
        print("\nStarting evaluation...")
        trainer.evaluate()


def analyze_and_visualize_embeddings(args):
    print("\nExtracting embeddings...")
    embeddings, labels = extract_embeddings(
        model_dir=args.model_path,
        layer_index=-1
    )

    print("\nVisualizing embeddings...")
    visualize_embeddings(
        embeddings=embeddings,
        labels=labels,
        method=args.method,
        output_path=args.output_path
    )


def run_node2vec(args):
    # Load the dataset
    dataset = DatasetLoader(args.dataset).load()
    data = dataset[0]

    # Initialize Node2Vec model
    node2vec = Node2VecModel(
        edge_index=data.edge_index,
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        epochs=args.epochs
    )

    # Train Node2Vec
    node2vec.train()

    # Get embeddings
    embeddings = node2vec.get_embeddings()

    # Perform tasks
    if "node_classification" in args.tasks:
        metrics = node2vec.evaluate_node_classification(embeddings, data.y, data.train_mask, data.test_mask)
        print("\nNode Classification Metrics:")
        print(metrics)

    if "link_prediction" in args.tasks:
        # Ensure edge_label_index and edge_label exist
        data = prepare_link_prediction_data(data)

        # Split edge_label_index into positive and negative edges
        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        neg_edge_index = data.edge_label_index[:, data.edge_label == 0]

        # Evaluate link prediction
        metrics = node2vec.evaluate_link_prediction(embeddings, pos_edge_index, neg_edge_index)
        print("\nLink Prediction Metrics:")
        print(metrics)


def run_explainability(args):
    # Load dataset
    dataset = DatasetLoader(args.dataset).load()
    data = dataset[0]

    # Load the model
    model = load_generalized_gnn(args.model_path)
    model.eval()

    # Run explainability
    explainer = GNNExplainability(model, data)

    if args.node_idx is not None:
        explanation = explainer.explain_node(args.node_idx)
        explainer.visualize_node(args.node_idx, explanation["edge_mask"], path=args.output_path)
    else:
        explainer.explain_all_nodes(num_nodes=args.num_nodes)


def main():
    parser = argparse.ArgumentParser(description="GNN Training, Evaluation, Cross-Testing, and Experimentation")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Single execution parser
    single_parser = subparsers.add_parser("single", help="Run single execution (train, evaluate, cross_test)")
    single_parser.add_argument("actions", nargs="+", choices=["train", "evaluate", "cross_test"])
    single_parser.add_argument("--model_type", choices=["simple", "generalized"], default="simple")
    single_parser.add_argument("--hidden_dim", type=int, default=256)
    single_parser.add_argument("--num_layers", type=int, default=4)
    single_parser.add_argument("--variant", choices=["gcn", "sage", "gat"], default="gat")
    single_parser.add_argument("--dropout", type=float, default=0.5)
    single_parser.add_argument("--use_residual", action="store_true")
    single_parser.add_argument("--use_layer_norm", action="store_true")
    single_parser.add_argument("--model_path", type=str, default="models/gnn_model.pth")
    single_parser.add_argument("--default_handling", choices=["auto", "pca", "zero_pad", "replicate"], default="auto")

    # Experiment parser
    experiment_parser = subparsers.add_parser("experiment", help="Run batch experiments with different parameters")

    # Best model finder parser
    best_model_parser = subparsers.add_parser("find_best", help="Find the best model based on saved results")
    best_model_parser.add_argument("--saved_models_dir", type=str, default="saved_models")

    # Link Prediction parser
    link_prediction_parser = subparsers.add_parser("link_prediction",
                                                   help="Run training and evaluation for link prediction")
    link_prediction_parser.add_argument("actions", nargs="+", choices=["train", "evaluate"], help="Actions to perform")
    link_prediction_parser.add_argument("--dataset", type=str, required=True,
                                        help="Dataset name (e.g., Cora, CiteSeer, PubMed)")
    link_prediction_parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension")
    link_prediction_parser.add_argument("--output_dim", type=int, default=32, help="Output layer dimension")
    link_prediction_parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    link_prediction_parser.add_argument("--variant", choices=["gcn", "sage", "gat"], default="gcn", help="GNN variant")
    link_prediction_parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    link_prediction_parser.add_argument("--use_residual", action="store_true", help="Use residual connections")
    link_prediction_parser.add_argument("--use_layer_norm", action="store_true", help="Use layer normalization")
    link_prediction_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    link_prediction_parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for optimizer")
    link_prediction_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    link_prediction_parser.add_argument("--device", type=str, default="cuda",
                                        help="Device for training ('cpu' or 'cuda')")

    embedding_parser = subparsers.add_parser(
        "analyze_embeddings", help="Extract and visualize node embeddings"
    )
    embedding_parser.add_argument("--model_path", help="Path to the trained model",
                                  default="saved_models/generalized_20241216_114950")
    embedding_parser.add_argument("--dataset", help="Dataset name (e.g., Cora)", default="Cora")
    embedding_parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Reduction method")
    embedding_parser.add_argument("--layer_index", type=int, default=-1, help="Layer index to extract embeddings from")
    embedding_parser.add_argument("--output_path", type=str, default=None, help="Path to save the plot")
    embedding_parser.set_defaults(func=analyze_and_visualize_embeddings)

    node2vec_parser = subparsers.add_parser("node2vec", help="Run Node2Vec-based tasks")
    node2vec_parser.add_argument("--dataset", required=True, help="Dataset name (e.g., Cora)")
    node2vec_parser.add_argument("--embedding_dim", type=int, default=64, help="Node embedding dimensionality")
    node2vec_parser.add_argument("--walk_length", type=int, default=10, help="Length of random walks")
    node2vec_parser.add_argument("--context_size", type=int, default=10, help="Context size for skip-gram")
    node2vec_parser.add_argument("--walks_per_node", type=int, default=5, help="Number of walks per node")
    node2vec_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    node2vec_parser.add_argument("--tasks", nargs="+", choices=["node_classification", "link_prediction"],
                                 required=True)
    node2vec_parser.set_defaults(func=run_node2vec)

    explain_parser = subparsers.add_parser("explain", help="Run GNN explainability")

    # Default dataset and model path for demonstration
    explain_parser.add_argument("--dataset", default="Cora", help="Dataset name (default: Cora)")
    explain_parser.add_argument("--model_path", default="saved_models/generalized_20241216_094023",
                                help="Path to the trained model (default: saved_models/generalized_20241216_094023)")

    # GNN model configuration
    explain_parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size (default: 64)")
    explain_parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers (default: 2)")
    explain_parser.add_argument("--variant", choices=["gcn", "sage", "gat"], default="gcn",
                                help="GNN variant (default: gcn)")
    explain_parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
    explain_parser.add_argument("--use_residual", action="store_true", help="Use residual connections (default: False)")
    explain_parser.add_argument("--use_layer_norm", action="store_true",
                                help="Use layer normalization (default: False)")

    # Explanation options
    explain_parser.add_argument("--node_idx", type=int, default=None,
                                help="Node index to explain (optional, default: None)")
    explain_parser.add_argument("--num_nodes", type=int, default=5,
                                help="Number of nodes to explain if node_idx is not provided (default: 5)")
    explain_parser.add_argument("--output_path", type=str, default=None,
                                help="Path to save explanation visualization (default: None)")

    explain_parser.set_defaults(func=run_explainability)

    args = parser.parse_args()

    if args.command == "single":
        run_single_execution(args)
    elif args.command == "experiment":
        run_experiments(args)
    elif args.command == "find_best":
        find_best_model(args)
    elif args.command == "link_prediction":
        run_link_prediction(args)
    elif args.command == "analyze_embeddings":
        analyze_and_visualize_embeddings(args)
    elif args.command == "node2vec":
        run_node2vec(args)
    elif args.command == "explain":
        run_explainability(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
