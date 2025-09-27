import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class Plotter:
    """
    A collection of plotting and table generation functions
    for visualizing the results of a machine learning experiment.
    """
    def plot_accuracy_measures_table(self):
        """
        Plots a reference table of key accuracy measures and their formulas.
        """
        # Define accuracy measures and their formulas
        measures = [
            'Confusion Matrix',
            'OOB Accuracy',
            'OOB Error',
            'K-Fold Accuracy',
            'K-Fold Error',
            'Recall (Sensitivity)',
            'Precision',
            'Specificity',
            'F1 Score'
        ]

        formulas = [
            'Visual representation of model performance',
            '$\\text{OOB}_{score}$',
            '$1 - \\text{OOB}_{score}$',
            'Mean of per-fold accuracies',
            '1 - K-Fold Accuracy',
            '$\\frac{TP}{TP + FN}$',
            '$\\frac{TP}{TP + FP}$',
            '$\\frac{TN}{TN + FP}$',
            '$\\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$'
        ]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 5.5))  # Adjust size as needed
        ax.axis('off')  # Hide axes

        # Create table data
        table_data = list(zip(measures, formulas))

        # Create table
        table = ax.table(cellText=table_data,
                         colLabels=['Measure', 'Formula'],
                         cellLoc='left',
                         loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3)  # Stretch rows for readability

        plt.show()

    def plot_hyperparameter_table(self):
        """
        Plots a table of the hyperparameters and values used in the grid search.
        """
        # Define hyperparameters and values
        hyperparameters = ['n_estimators (NREE)', 'max_features (MTRY)', 'cutoff']
        values = ['500, 1000', '0.5√n_features, √n_features, 2√n_features', '0.5 (default)']

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 2))  # adjust size as needed
        ax.axis('off')  # hide axes

        # Create table
        table_data = list(zip(hyperparameters, values))
        table = ax.table(cellText=table_data,
                         colLabels=['Hyperparameter', 'Values'],
                         cellLoc='center',
                         loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # stretch rows for readability

        plt.show()

    def summarize_dbs_and_plot(self, training_path='training_db.csv', verification_path='verification_db.csv'):
        """
        Summarizes and plots the class distribution of the training
        and verification databases for initial data exploration.
        """
        # Load datasets
        training_db = pd.read_csv(training_path)
        verification_db = pd.read_csv(verification_path)

        def summarize(df, name):
            num_samples = df.shape[0]
            num_features = df.shape[1] - 1
            class_counts = df['Label'].value_counts()
            class_ratios = df['Label'].value_counts(normalize=True)
            missing_values = df.isnull().sum().sum()

            return {
                'Dataset': name,
                'Samples': num_samples,
                'Features': num_features,
                'Label 0 Count': class_counts.get(0, 0),
                'Label 1 Count': class_counts.get(1, 0),
                'Label 0 Ratio': round(class_ratios.get(0, 0), 2),
                'Label 1 Ratio': round(class_ratios.get(1, 0), 2),
                'Missing Values': missing_values
            }, class_counts

        # Summarize datasets and get class counts
        train_summary, train_counts = summarize(training_db, 'Training')
        verification_summary, verification_counts = summarize(verification_db, 'Verification')

        # Create summary table
        summary_table = pd.DataFrame([train_summary, verification_summary])
        print(summary_table.to_string(index=False))

        # Plot class distributions
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        train_counts.plot(kind='bar', ax=axes[0], color=['#4F81BD', '#C0504D'])
        axes[0].set_title('Training Dataset Class Distribution')
        axes[0].set_xlabel('Label')
        axes[0].set_ylabel('Count')

        verification_counts.plot(kind='bar', ax=axes[1], color=['#4F81BD', '#C0504D'])
        axes[1].set_title('Verification Dataset Class Distribution')
        axes[1].set_xlabel('Label')
        axes[1].set_ylabel('Count')

        plt.tight_layout()

        plt.tight_layout()

        # Adjust the bottom margin to make space for the text
        plt.subplots_adjust(bottom=0.2)

        # Add caption inside figure (below the plots)
        fig.text(0.5, 0.05,
                 'This chart shows the distribution of sample counts (y-axis) across the class labels (x-axis).',
                 ha='center', fontsize=10)

        plt.show()

    def plot_best_parameters_table(self, best_n_estimators, best_max_features, cutoff_value):
        """
        Plots a table summarizing the optimal hyperparameters found by the grid search.
        """
        # Define the data for the table
        table_data = [
            ['Best n_estimators (NTREE)', str(best_n_estimators)],
            ['Best max_features (MTRY)', str(best_max_features)],
            ['Cutoff', str(cutoff_value)]
        ]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=table_data,
                         colLabels=['Parameter', 'Value'],
                         cellLoc='center',
                         loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        plt.title('Best Trained RF Parameters', loc='center')
        plt.show()

    def plot_accuracy_comparison_table(self, cv_acc, cv_err, oob_acc, oob_err):
        """
        Plots a table comparing the accuracy and error from the
        Cross-Validation and Out-of-Bag validation methods.
        """
        # Define the data for the table
        table_data = [
            ['CV Accuracy', f"{cv_acc:.5f}"],
            ['CV Error', f"{cv_err:.5f}"],
            ['OOB Accuracy', f"{oob_acc:.5f}"],
            ['OOB Error', f"{oob_err:.5f}"]
        ]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=table_data,
                         colLabels=['Method', 'Value'],
                         cellLoc='center',
                         loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        plt.title('CV vs. OOB Accuracy Comparison', loc='center')
        plt.show()

    def plot_final_metrics_table(self, recall, precision, specificity, f1_score):
        """
        Plots a table of the final key performance metrics (recall, precision, etc.)
        calculated from the combined confusion matrix.
        """
        # Define the data for the table
        table_data = [
            ['Recall (Sensitivity)', f"{recall:.5f}"],
            ['Precision', f"{precision:.5f}"],
            ['Specificity', f"{specificity:.5f}"],
            ['F1 Score', f"{f1_score:.5f}"]
        ]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=table_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        plt.title('Final Accuracy Measures', loc='center')
        plt.show()

    def plot_confusion_matrix(self, final_confusion_matrix):
        """
        Plots a non-normalized, visual representation of the final combined
        confusion matrix, showing the true vs. predicted counts.
        """
        # The confusion matrix shall show full sample counts and not be normalized
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(final_confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

        # Set labels as per professor's instructions
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Final Combined Confusion Matrix (Non-Normalized)')
        plt.show()

    def plot_feature_rankings(self, model, feature_names, n=10):
        """
        Plots the top N features ranked by their Gini importance.
        This helps in understanding which features are most impactful to the model.
        """
        # Get the feature importances from the trained model
        importances = model.feature_importances_

        # Pair feature names with their importance scores
        feature_importance_pair = list(zip(feature_names, importances))

        # Sort the features by importance in descending order
        feature_importance_pair.sort(key=lambda x: x[1], reverse=True)

        # Select the top N features
        top_n_features = feature_importance_pair[:n]

        # Unzip the pairs for plotting
        top_features = [x[0] for x in top_n_features]
        top_importances = [x[1] for x in top_n_features]

        print('Top features importances:' + str(top_features))
        print('Top features importances:' + str(top_importances))

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(top_features[::-1], top_importances[::-1], color='skyblue')
        plt.xlabel('Importance (Gini)')
        plt.ylabel('Feature')
        plt.title(f'Top {n} Feature Rankings (Gini Importance)')
        plt.show()

    def plot_verification_results(self, predictions, probabilities, y_verif):
        """
        Plots a table showing the model's predictions and confidence
        for the two samples in the verification database.
        """
        # Create a DataFrame for the table
        results = pd.DataFrame({
            'Sample': ['Positive', 'Negative'],
            'Ground Truth': y_verif,
            'Predicted Class': predictions,
            'Predicted Probability': [f"{p:.5f}" for p in np.max(probabilities, axis=1)]
        })

        fig, ax = plt.subplots(figsize=(9, 2))
        ax.axis('off')

        table = ax.table(cellText=results.values,
                         colLabels=results.columns,
                         loc='center',
                         cellLoc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        plt.title('RF Run Time Engine Results', loc='center')
        plt.show()