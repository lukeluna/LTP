import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def confusion_matrix_metrics(dataset, expected='expected_label', actual='canonical'):
    # Get the unique labels from the expected labels
    labels = sorted(dataset[expected].unique())

    # Generate the confusion matrix with the specified labels
    cm = confusion_matrix(dataset[expected], dataset[actual], labels=labels)

    # Create a DataFrame for better visualization with seaborn
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Calculate per-class metrics
    report = classification_report(dataset[expected], dataset[actual], labels=labels, output_dict=True, zero_division=0)

    # Extract per-class metrics
    per_class_metrics = {label: {
        'precision': report[label]['precision'],
        'recall': report[label]['recall'],
        'f1-score': report[label]['f1-score'],
        'accuracy': cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    } for i, label in enumerate(labels)}

    # Convert per-class metrics to DataFrame
    metrics_df = pd.DataFrame(per_class_metrics).T

    # Print per-class metrics
    print("Per-class metrics:")
    print(metrics_df)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Actual Labels')
    plt.ylabel('Expected Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

    # Plot the per-class metrics
    plt.figure(figsize=(12, 6))
    sns.heatmap(metrics_df, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.title('Per-Class Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

# Load the dataset
dataset = pd.read_csv('results_canonical_spans/results-cot-mafalda-spans-mistral_canonical.csv')

# Display the confusion matrix with metrics
confusion_matrix_metrics(dataset)
