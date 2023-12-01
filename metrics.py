"""
the Hter metric script
Author: Abdelkarim eljandoubi
date: Nov 2023
"""

import numpy as np

def calculate_hter(eval_pred):
    """
    Calculate the Half Total Error Rate (HTER) for binary classification.

    Returns:
    float: The calculated HTER.
    """

    # Extract true labels and the model predictions
    true_labels = eval_pred.label_ids
    predicted_labels = np.argmax(eval_pred.predictions, axis=1)

    # Calculate errors for each class
    errors_class_0 = np.sum((true_labels == 0) & (predicted_labels != 0))
    errors_class_1 = np.sum((true_labels == 1) & (predicted_labels != 1))

    # Calculate the total number of occurrences for each class
    total_class_0 = np.sum(true_labels == 0)
    total_class_1 = np.sum(true_labels == 1)

    # Handle the case where a class does not occur to avoid division by zero
    error_rate_0 = errors_class_0 / total_class_0 if total_class_0 > 0 else 0
    error_rate_1 = errors_class_1 / total_class_1 if total_class_1 > 0 else 0

    # Calculate HTER
    hter = 0.5 * (error_rate_0 + error_rate_1)

    return {"hter" : hter}
