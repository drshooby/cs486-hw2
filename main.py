import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
from sklearn.model_selection import cross_val_score
from plotter import Plotter
from math import sqrt


class RandomForestPipeline:
    """
    Orchestrates the entire Random Forest experiment pipeline.
    This class handles data preparation, hyperparameter tuning via manual
    k-fold cross-validation, final model training, evaluation, and plotting.
    """
    def __init__(self, orig_file_path, random_state=0, k_folds=5):
        self.orig_file_path = orig_file_path
        self.random_state = random_state
        self.train_file_path = None
        self.verif_file_path = None
        self.k_folds = k_folds

    def load_data(self):
        """
        Loads the original dataset, confirms class balance, and creates
        a separate verification database with one positive and one negative sample.
        """
        df = pd.read_csv(self.orig_file_path)
        y = df['Label']

        def confirm_balance():
            threshold = 0.1
            sample_size = len(y)
            assert len(y) == 871
            for k, v in Counter(y).items():
                assert int(v) / sample_size > threshold
                print(f"Label {k}: {(int(v) / sample_size)}")
            print("Data is balanced")

        confirm_balance()

        class_0_samples = df[df['Label'] == 0]
        class_1_samples = df[df['Label'] == 1]

        verification_sample_0 = class_0_samples.sample(n=1, random_state=self.random_state)
        verification_sample_1 = class_1_samples.sample(n=1, random_state=self.random_state)

        verification_db = pd.concat([verification_sample_0, verification_sample_1])
        training_db = df.drop(verification_db.index)

        self.verif_file_path = "data/verification_db.csv"
        self.train_file_path = "data/training_db.csv"

        verification_db.to_csv(self.verif_file_path, index=False)
        training_db.to_csv(self.train_file_path, index=False)

        assert os.path.exists(self.verif_file_path)
        vf = pd.read_csv(self.verif_file_path)
        assert len(vf) == 2
        assert os.path.exists(self.train_file_path)
        tf = pd.read_csv(self.train_file_path)
        assert len(df) - len(tf) == len(vf)

    @staticmethod
    def create_param_grid(n_features):
        """
        Defines the hyperparameter grid for the grid search.
        max_features (MTRY) is converted to integers as required by the instructions.
        """
        return {
            'n_estimators': [500, 1000],
            'max_features': list(map(int, [0.5 * sqrt(n_features), sqrt(n_features), 2 * sqrt(n_features)])),
        }

    def k_fold_cross_validation(self, X, y, NTREE, MTRY, verbose=False):
        """
        Performs a manual k-fold cross-validation to evaluate model performance
        for a given set of hyperparameters. A new model is trained for each fold.
        """
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X))

        X = X[indices]
        y = y[indices]

        k = self.k_folds
        fold_size = len(X) // k

        folds = []
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i != k - 1 else len(X)
            folds.append((X[start:end], y[start:end]))

        accuracies = []
        errors = []
        confusion_matrices = []
        recalls = []
        precisions = []
        f1_scores = []
        specificities = []

        for i in range(k):
            X_test, y_test = folds[i]
            X_train = np.vstack([folds[j][0] for j in range(k) if j != i])
            y_train = np.hstack([folds[j][1] for j in range(k) if j != i])

            # Initialize a new model for each fold to ensure independent training
            model = RandomForestClassifier(n_estimators=NTREE, max_features=MTRY, random_state=self.random_state)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            err = 1 - acc
            spec = recall_score(y_test, y_pred, pos_label=0)

            accuracies.append(acc)
            errors.append(err)
            confusion_matrices.append(confusion_matrix(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            specificities.append(spec)

            if verbose:
                print(f"Fold {i + 1} accuracy: {accuracies[-1]:.5f} | error: {errors[-1]:.5f}")

        # The final confusion matrix is obtained by summing the matrices from all folds
        final_confusion_matrix = np.sum(confusion_matrices, axis=0)

        final_accuracy = np.mean(accuracies)
        final_error = np.mean(errors)
        final_recall = np.mean(recalls)
        final_precision = np.mean(precisions)
        final_f1 = np.mean(f1_scores)
        final_specificity = np.mean(specificities)

        return {
            'accuracy': final_accuracy,
            'error': final_error,
            'recall': final_recall,
            'precision': final_precision,
            'f1_score': final_f1,
            'specificity': final_specificity,
            'confusion_matrix': final_confusion_matrix
        }

    def run_grid_search(self):
        """
        Orchestrates the grid search to find the best hyperparameters.
        It returns the metrics for the best-performing model.
        """
        df = pd.read_csv(self.train_file_path)
        n_samples, n_features = df.drop('Label', axis=1).shape
        X, y = df.drop('Label', axis=1).values, df['Label']

        params = self.create_param_grid(n_features)

        all_results = []
        for ntree in params['n_estimators']:
            for mtry in params['max_features']:
                cv_metrics = self.k_fold_cross_validation(X, y, ntree, mtry, verbose=True)

                cv_metrics['n_estimators'] = ntree
                cv_metrics['max_features'] = mtry

                all_results.append(cv_metrics)

        results_df = pd.DataFrame(all_results)
        print(results_df)

        best_model_results = results_df.loc[results_df['accuracy'].idxmax()]
        print("\nBest Model Found:")
        print(best_model_results)

        return best_model_results, X, y, df

    def run_time_predict(self, final_best_model):
        """
        Acts as the RF run time engine to predict the class of new samples
        from the verification database.
        """
        verification_df = pd.read_csv('data/verification_db.csv')
        X_verif = verification_df.drop('Label', axis=1).values
        y_verif = verification_df['Label'].values

        predictions = final_best_model.predict(X_verif)
        probabilities = final_best_model.predict_proba(X_verif)

        return predictions, probabilities, y_verif


if __name__ == '__main__':
    """
    Main execution block that orchestrates the entire pipeline.
    """
    DATA_FILE = "data/original-training-db-e1-positive.csv"
    pipeline = RandomForestPipeline(orig_file_path=DATA_FILE)
    plotter = Plotter()

    # Step 1: Prepare the data
    pipeline.load_data()

    # Step 2: Run the grid search to find the best parameters
    best_results_series, X, y, df = pipeline.run_grid_search()

    # Step 3: Train the final best model using the optimal hyperparameters
    best_ntree = int(best_results_series['n_estimators'])
    best_mtry = int(best_results_series['max_features'])

    final_best_model = RandomForestClassifier(n_estimators=best_ntree,
                                              max_features=best_mtry,
                                              random_state=pipeline.random_state,
                                              oob_score=True)

    # Train this final model on ALL of the training data
    final_best_model.fit(X, y)

    # Step 4: Perform all final evaluations on the best model
    final_oob_score = final_best_model.oob_score_
    final_oob_error = 1 - final_oob_score
    print(f"\nFinal OOB Score: {final_oob_score:.5f}")
    print(f"Final OOB Error: {final_oob_error:.5f}")

    scores = cross_val_score(final_best_model, X, y, cv=5)
    mean_cv_accuracy = np.mean(scores)
    print(f"SciKit's built-in CV accuracy: {mean_cv_accuracy:.5f}")

    # Step 5: Generate all required plots and tables
    plotter.plot_best_parameters_table(best_n_estimators=best_ntree, best_max_features=best_mtry, cutoff_value=0.5)
    plotter.plot_accuracy_comparison_table(
        cv_acc=best_results_series['accuracy'],
        cv_err=best_results_series['error'],
        oob_acc=final_oob_score,
        oob_err=final_oob_error
    )
    plotter.plot_final_metrics_table(
        recall=best_results_series['recall'],
        precision=best_results_series['precision'],
        specificity=best_results_series['specificity'],
        f1_score=best_results_series['f1_score']
    )
    plotter.plot_confusion_matrix(final_confusion_matrix=best_results_series['confusion_matrix'])

    # Get feature names for plotting from the DataFrame
    feature_names = df.drop('Label', axis=1).columns
    plotter.plot_feature_rankings(model=final_best_model, feature_names=feature_names, n=30)

    # Step 6: Run final test on the verification data
    predictions, probabilities, y_verif = pipeline.run_time_predict(final_best_model)
    plotter.plot_verification_results(predictions, probabilities, y_verif)