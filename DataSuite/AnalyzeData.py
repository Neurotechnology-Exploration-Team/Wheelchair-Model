import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def analyze_dataset(df, labelEncoder):
    # Ensure df is a DataFrame
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    # Assuming 'Label' is the name of the column containing encoded labels
    if 'Label' in df.columns:
        # Reverse-transform labels
        labels_encoded = df['Label'].values
        label_strings = labelEncoder.inverse_transform(labels_encoded)
        label_counts = pd.Series(label_strings).value_counts()

        print("Basic Dataset Info:")
        print(df.info())

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nLabel Distribution:")
        print("Label Counts:\n", label_counts)  # Explicitly print label counts
        label_counts.plot(kind='bar')
        plt.title('Label Distribution')
        plt.xlabel('Labels')
        plt.ylabel('Frequency')
        plt.show()

        # Add eeg data graphing?
    else:
        print("Label column not found in DataFrame.")

    # Data Distribution Analysis
    print("\nData Distribution Analysis:")
    df.describe().plot(kind='bar', figsize=(10, 8))
    plt.title('Data Distribution')
    plt.show()

    # Feature Correlation Matrix
    print("\nFeature Correlation Matrix:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()


#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
#
#
#def evaluate_dataset_and_model_performance(model, X_train, y_train, X_val, y_val):
#    # Assuming model is a trained PyTorch model and X_train, y_train, X_val, y_val are your datasets
#
#    # 1. Data Distribution
#    print("Visualizing data distribution...")
#    # Example for a single feature visualization
#    sns.histplot(X_train[:, 0])  # Adjust according to your dataset structure
#    plt.show()
#
#    # 2. Correlation Matrix
#    print("Correlation matrix:")
#    corr_matrix = np.corrcoef(X_train, rowvar=False)
#    sns.heatmap(corr_matrix)
#    plt.show()
#
#    # 3. Model Performance Metrics
#    # Convert PyTorch tensors to numpy arrays if necessary and make predictions
#    # For illustration, using sklearn metrics
#    y_pred_train = model.predict(X_train)
#    y_pred_val = model.predict(X_val)
#
#    print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
#    print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
#
#    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_val, average='binary')
#    print("Precision:", precision)
#    print("Recall:", recall)
#    print("F1 Score:", f1)
#
#    # 4. Learning Curves
#    # This part assumes you have recorded training and validation losses during model training
#    print("Plotting learning curves...")
#    plt.plot(training_losses, label='Training Loss')
#    plt.plot(validation_losses, label='Validation Loss')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.legend()
#    plt.title('Learning Curves')
#    plt.show()
#