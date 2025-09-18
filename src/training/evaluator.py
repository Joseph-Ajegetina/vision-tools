import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
    top_k_accuracy_score
)
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation suite for comparing MLP and CNN models.

    This class provides detailed performance analysis including:
    - Classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix analysis
    - Per-class performance
    - Model comparison utilities
    - Visualization tools

    Mathematical Foundation:
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    """

    def __init__(
        self,
        class_names: List[str],
        device: torch.device,
        save_dir: str = "./evaluation_results"
    ):
        """
        Initialize the evaluator.

        Args:
            class_names: List of class names for the dataset
            device: Device to run evaluation on
            save_dir: Directory to save evaluation results
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.

        Args:
            model: Trained model to evaluate
            test_loader: Test data loader
            model_name: Name of the model for reporting

        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        model.eval()

        # Storage for predictions and targets
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_times = []

        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                # Move to device
                data, targets = data.to(self.device), targets.to(self.device)

                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None

                if start_time:
                    start_time.record()

                # Forward pass
                outputs = model(data)
                probabilities = torch.softmax(outputs, dim=1)

                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_times.append(start_time.elapsed_time(end_time))

                # Store results
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)

        # Calculate metrics
        results = self._calculate_metrics(
            targets, predictions, probabilities, model_name
        )

        # Add timing information
        if inference_times:
            results['inference_time'] = {
                'mean_ms': np.mean(inference_times),
                'std_ms': np.std(inference_times),
                'total_ms': np.sum(inference_times)
            }

        # Add model information
        if hasattr(model, 'get_model_info'):
            results['model_info'] = model.get_model_info()

        if hasattr(model, 'count_parameters'):
            results['parameters'] = model.count_parameters()

        logger.info(f"{model_name} evaluation completed")
        return results

    def _calculate_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for model evaluation.

        Args:
            targets: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            model_name: Name of the model

        Returns:
            Dictionary containing all metrics
        """
        # Basic metrics
        accuracy = np.mean(predictions == targets) * 100

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )

        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )

        # Top-k accuracy
        top2_accuracy = top_k_accuracy_score(targets, probabilities, k=2) * 100
        top3_accuracy = top_k_accuracy_score(targets, probabilities, k=3) * 100

        # Confusion matrix
        cm = confusion_matrix(targets, predictions)

        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100

        # Classification report
        class_report = classification_report(
            targets, predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        results = {
            'model_name': model_name,
            'overall_metrics': {
                'accuracy': accuracy,
                'top2_accuracy': top2_accuracy,
                'top3_accuracy': top3_accuracy,
                'macro_precision': macro_precision * 100,
                'macro_recall': macro_recall * 100,
                'macro_f1': macro_f1 * 100,
                'weighted_precision': weighted_precision * 100,
                'weighted_recall': weighted_recall * 100,
                'weighted_f1': weighted_f1 * 100
            },
            'per_class_metrics': {
                'class_names': self.class_names,
                'accuracy': per_class_accuracy.tolist(),
                'precision': (precision * 100).tolist(),
                'recall': (recall * 100).tolist(),
                'f1_score': (f1 * 100).tolist(),
                'support': support.tolist()
            },
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities
        }

        return results

    def compare_models(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two models and generate comparison report.

        Args:
            results1: Results from first model (e.g., MLP)
            results2: Results from second model (e.g., CNN)

        Returns:
            Comparison report dictionary
        """
        model1_name = results1['model_name']
        model2_name = results2['model_name']

        logger.info(f"Comparing {model1_name} vs {model2_name}")

        # Overall performance comparison
        metrics_to_compare = [
            'accuracy', 'top2_accuracy', 'top3_accuracy',
            'macro_precision', 'macro_recall', 'macro_f1',
            'weighted_precision', 'weighted_recall', 'weighted_f1'
        ]

        comparison = {
            'model_comparison': {
                'model1': model1_name,
                'model2': model2_name
            },
            'performance_difference': {},
            'statistical_significance': {},
            'per_class_comparison': {}
        }

        # Calculate performance differences
        for metric in metrics_to_compare:
            val1 = results1['overall_metrics'][metric]
            val2 = results2['overall_metrics'][metric]
            diff = val2 - val1
            comparison['performance_difference'][metric] = {
                f'{model1_name}': val1,
                f'{model2_name}': val2,
                'difference': diff,
                'relative_improvement': (diff / val1 * 100) if val1 != 0 else 0
            }

        # Per-class comparison
        for i, class_name in enumerate(self.class_names):
            class_metrics = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                val1 = results1['per_class_metrics'][metric][i]
                val2 = results2['per_class_metrics'][metric][i]
                class_metrics[metric] = {
                    f'{model1_name}': val1,
                    f'{model2_name}': val2,
                    'difference': val2 - val1
                }
            comparison['per_class_comparison'][class_name] = class_metrics

        # Model complexity comparison
        if 'parameters' in results1 and 'parameters' in results2:
            comparison['model_complexity'] = {
                f'{model1_name}_parameters': results1['parameters'],
                f'{model2_name}_parameters': results2['parameters'],
                'parameter_ratio': results2['parameters'] / results1['parameters']
            }

        # Inference time comparison
        if 'inference_time' in results1 and 'inference_time' in results2:
            comparison['inference_time'] = {
                f'{model1_name}_ms': results1['inference_time']['mean_ms'],
                f'{model2_name}_ms': results2['inference_time']['mean_ms'],
                'speedup': results1['inference_time']['mean_ms'] / results2['inference_time']['mean_ms']
            }

        return comparison

    def plot_confusion_matrix(
        self,
        results: Dict[str, Any],
        normalize: bool = True,
        save_plot: bool = True
    ):
        """
        Plot confusion matrix for a model.

        Args:
            results: Evaluation results dictionary
            normalize: Whether to normalize the confusion matrix
            save_plot: Whether to save the plot
        """
        cm = results['confusion_matrix']
        model_name = results['model_name']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = f'Normalized Confusion Matrix - {model_name}'
            fmt = '.2f'
        else:
            title = f'Confusion Matrix - {model_name}'
            fmt = 'd'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        if save_plot:
            filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}"
            if normalize:
                filename += "_normalized"
            filename += ".png"

            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {self.save_dir / filename}")

        plt.show()

    def plot_model_comparison(
        self,
        comparison: Dict[str, Any],
        save_plot: bool = True
    ):
        """
        Plot comparison between two models.

        Args:
            comparison: Comparison results dictionary
            save_plot: Whether to save the plot
        """
        model1_name = comparison['model_comparison']['model1']
        model2_name = comparison['model_comparison']['model2']

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Overall metrics comparison
        metrics = ['accuracy', 'macro_f1', 'weighted_f1']
        model1_values = [comparison['performance_difference'][m][model1_name] for m in metrics]
        model2_values = [comparison['performance_difference'][m][model2_name] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax1.bar(x - width/2, model1_values, width, label=model1_name, alpha=0.8)
        ax1.bar(x + width/2, model2_values, width, label=model2_name, alpha=0.8)
        ax1.set_ylabel('Performance (%)')
        ax1.set_title('Overall Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Per-class accuracy comparison
        class_names = list(comparison['per_class_comparison'].keys())
        model1_acc = [comparison['per_class_comparison'][c]['accuracy'][model1_name] for c in class_names]
        model2_acc = [comparison['per_class_comparison'][c]['accuracy'][model2_name] for c in class_names]

        x_classes = np.arange(len(class_names))
        ax2.bar(x_classes - width/2, model1_acc, width, label=model1_name, alpha=0.8)
        ax2.bar(x_classes + width/2, model2_acc, width, label=model2_name, alpha=0.8)
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Per-Class Accuracy Comparison')
        ax2.set_xticks(x_classes)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Model complexity comparison (if available)
        if 'model_complexity' in comparison:
            models = [model1_name, model2_name]
            param_counts = [
                comparison['model_complexity'][f'{model1_name}_parameters'],
                comparison['model_complexity'][f'{model2_name}_parameters']
            ]

            ax3.bar(models, param_counts, alpha=0.8, color=['skyblue', 'lightcoral'])
            ax3.set_ylabel('Number of Parameters')
            ax3.set_title('Model Complexity Comparison')
            ax3.grid(True, alpha=0.3)

            # Add parameter count annotations
            for i, v in enumerate(param_counts):
                ax3.text(i, v + max(param_counts) * 0.01, f'{v:,}',
                        ha='center', va='bottom', fontweight='bold')

        # Performance improvement per class
        class_improvements = [
            comparison['per_class_comparison'][c]['accuracy']['difference']
            for c in class_names
        ]

        colors = ['green' if imp > 0 else 'red' for imp in class_improvements]
        ax4.bar(range(len(class_names)), class_improvements, alpha=0.8, color=colors)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_ylabel(f'Accuracy Improvement (%)\n({model2_name} - {model1_name})')
        ax4.set_title('Per-Class Performance Improvement')
        ax4.set_xticks(range(len(class_names)))
        ax4.set_xticklabels(class_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            filename = f"model_comparison_{model1_name}_{model2_name}".lower().replace(' ', '_')
            plt.savefig(self.save_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {self.save_dir}/{filename}.png")

        plt.show()

    def generate_report(
        self,
        results_list: List[Dict[str, Any]],
        comparison: Optional[Dict[str, Any]] = None,
        save_report: bool = True
    ) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            results_list: List of model evaluation results
            comparison: Model comparison results (optional)
            save_report: Whether to save the report to file

        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("CINIC-10 MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Individual model results
        for results in results_list:
            model_name = results['model_name']
            metrics = results['overall_metrics']

            report.append(f"MODEL: {model_name}")
            report.append("-" * 40)
            report.append(f"Overall Accuracy:     {metrics['accuracy']:.2f}%")
            report.append(f"Top-2 Accuracy:       {metrics['top2_accuracy']:.2f}%")
            report.append(f"Top-3 Accuracy:       {metrics['top3_accuracy']:.2f}%")
            report.append(f"Macro F1-Score:       {metrics['macro_f1']:.2f}%")
            report.append(f"Weighted F1-Score:    {metrics['weighted_f1']:.2f}%")

            if 'parameters' in results:
                report.append(f"Model Parameters:     {results['parameters']:,}")

            if 'inference_time' in results:
                report.append(f"Avg Inference Time:   {results['inference_time']['mean_ms']:.2f} ms")

            report.append("")

        # Comparison section
        if comparison:
            model1_name = comparison['model_comparison']['model1']
            model2_name = comparison['model_comparison']['model2']

            report.append("MODEL COMPARISON")
            report.append("-" * 40)

            acc_diff = comparison['performance_difference']['accuracy']['difference']
            rel_imp = comparison['performance_difference']['accuracy']['relative_improvement']

            report.append(f"{model2_name} vs {model1_name}:")
            report.append(f"  Accuracy Improvement:    {acc_diff:+.2f}% ({rel_imp:+.1f}%)")

            f1_diff = comparison['performance_difference']['macro_f1']['difference']
            f1_rel_imp = comparison['performance_difference']['macro_f1']['relative_improvement']
            report.append(f"  Macro F1 Improvement:    {f1_diff:+.2f}% ({f1_rel_imp:+.1f}%)")

            if 'model_complexity' in comparison:
                param_ratio = comparison['model_complexity']['parameter_ratio']
                report.append(f"  Parameter Ratio:         {param_ratio:.1f}x")

            report.append("")

        # Best performing classes
        if len(results_list) > 0:
            report.append("PER-CLASS ANALYSIS")
            report.append("-" * 40)

            for results in results_list:
                model_name = results['model_name']
                per_class = results['per_class_metrics']

                # Find best and worst performing classes
                accuracies = per_class['accuracy']
                best_idx = np.argmax(accuracies)
                worst_idx = np.argmin(accuracies)

                report.append(f"{model_name}:")
                report.append(f"  Best Class:  {self.class_names[best_idx]} ({accuracies[best_idx]:.1f}%)")
                report.append(f"  Worst Class: {self.class_names[worst_idx]} ({accuracies[worst_idx]:.1f}%)")
                report.append("")

        report_text = "\n".join(report)

        if save_report:
            report_file = self.save_dir / "evaluation_report.txt"
            with open(report_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {report_file}")

        return report_text