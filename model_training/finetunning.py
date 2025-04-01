import os
import json
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization, Bidirectional, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from keras.utils import plot_model
from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping

data_directory = '../all_outputs/output_ssl_small/'

# list all the files in directories and sub directories in in data_direcotiry and store them list_list list
ft_file_names = []

for root, _, files in os.walk(data_directory):  # Walk through all directories and subdirectories
        for filename in files:
            ft_file_names.append(os.path.join(root, filename))

print(len(ft_file_names))  

all_data = []

pose_indices = [0, 15, 16, 17, 18, 19, 20]
hand_indices = [0, 4, 7, 8, 11, 12, 15, 16, 19, 20]

def load_landmarks(filenames, num_frames, glossIndex = 4):
    # Prepare storage for data and labels
    video_data = []
    labels = []
    
    for file in filenames:
        gloss = file.split('/')[glossIndex]
        with open(file, 'rb') as f:
                landmarks = pickle.load(f)
                    
                # Prepare frame storage for each video with fixed number of frames
                frames = []

                # Get frame sampling step (skip frames if necessary)
                total_frames = len(landmarks)
                step = max(1, total_frames // num_frames)
                    
                # Process frames
                for i in range(num_frames):
                    frame_index = i * step if total_frames >= num_frames else i
                    
                    if frame_index < total_frames:
                        frame = landmarks[frame_index]
                    else:
                        frame = None
                        
                    # Extract and flatten required points
                    pose_points = frame['pose_landmarks'] if frame and frame['pose_landmarks'] else [None] * 33
                    left_hand_points = frame['left_hand_landmarks'] if frame and frame['left_hand_landmarks'] else [None] * 21
                    right_hand_points = frame['right_hand_landmarks'] if frame and frame['right_hand_landmarks'] else [None] * 21
                        
                    # Collect only specified indices and flatten them
                    extracted_points = []

                    # Collect pose points first
                    for idx in pose_indices:
                        point = pose_points[idx] if pose_points[idx] is not None else {'x': 0, 'y': 0}
                        extracted_points.extend([point['x'], point['y']])
                        
                    for idx in hand_indices:
                        point = left_hand_points[idx] if left_hand_points[idx] is not None else {'x': 0, 'y': 0}
                        extracted_points.extend([point['x'], point['y']])

                    # Collect right hand points
                    for idx in hand_indices:
                        point = right_hand_points[idx] if right_hand_points[idx] is not None else {'x': 0, 'y': 0}
                        extracted_points.extend([point['x'], point['y']])
                        
                    frames.append(extracted_points)
                    
                # Add the processed video frames to the main array
                video_data.append(frames)
                labels.append(gloss)
                
    # Convert to numpy arrays
    video_data = np.array(video_data)
    labels = np.array(labels)
    
    return video_data, labels

# Example usage
num_frames = 30  # Desired fixed number of frames per video

fine_tune_video_data, fine_tune_labels = load_landmarks(ft_file_names, num_frames, 3)

# `video_data` has shape (number of videos, num_frames, number of points*2) where points*2 is the flattened landmark count
# `labels` contains the labels for each video

CUSTOM_NUM_CLASSES = 200
NUM_CLASSES = len(set(fine_tune_labels))
NUM_FT_CLASSES = NUM_CLASSES
INSTANCES_PER_CLASS = 2
N_RUNS = 5


# model_path = '../saved_models/book_8/pretrained_model_weights.h5'
model_path = '../saved_models/book_8/pretrained_model_weights_200_classes.h5'
# model_path = '../saved_models/book_8/pretrained_model_weights_80_with_overlapping_classes.h5'

# Count the occurrences of each class
class_counts = Counter(fine_tune_labels)

# Sort classes by their counts in descending order and select the top NUM_CLASSES classes
top_classes = [cls for cls, count in class_counts.most_common(NUM_FT_CLASSES)]

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(fine_tune_labels)
labels_one_hot = to_categorical(labels_encoded, num_classes=NUM_CLASSES)


# Custom split to ensure fixed instances per class in training set
train_indices = []
val_test_indices = []

for cls in range(NUM_CLASSES):
    # Get indices of samples for this class
    cls_indices = np.where(labels_encoded == cls)[0]
    np.random.shuffle(cls_indices)  # Randomize the order
    # Ensure there are enough samples for the class
    if len(cls_indices) < INSTANCES_PER_CLASS:
        raise ValueError(f"Class {cls} has only {len(cls_indices)} instances, but {INSTANCES_PER_CLASS} are required.")
    # Take fixed number of instances for training
    train_indices.extend(cls_indices[:INSTANCES_PER_CLASS])
    # Remaining instances go to validation/test
    val_test_indices.extend(cls_indices[INSTANCES_PER_CLASS:])

# Convert to numpy arrays
X_ft_train = fine_tune_video_data[train_indices]
y_ft_train = labels_one_hot[train_indices]
X_temp = fine_tune_video_data[val_test_indices]
y_temp = labels_one_hot[val_test_indices]

# Split remaining data into validation and test sets
X_ft_val, X_ft_test, y_ft_val, y_ft_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print train, val, test shapes
print(X_ft_train.shape, y_ft_train.shape)
print(X_ft_val.shape, y_ft_val.shape)
print(X_ft_test.shape, y_ft_test.shape)
print("Num Classes:", NUM_CLASSES)

# Print average number of instances per class in training set
class_counts = Counter(y_ft_train.argmax(axis=1))
print("Average number of instances per class in training set")
print(sum(class_counts.values()) / len(class_counts))

NUM_CLASSES = CUSTOM_NUM_CLASSES

class PositionEmbedding(layers.Layer):
    def __init__(self, config):
        super(PositionEmbedding, self).__init__()
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(config.max_position_embeddings, config.hidden_size),
            initializer="random_normal",
            trainable=True,
        )
    
    def call(self, x):
        seq_length = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = tf.gather(self.position_embeddings, positions)
        return x + position_embeddings

class Transformer(Model):
    def __init__(self, config, n_classes=50, freeze_pretrained=False):
        super(Transformer, self).__init__()
        self.l1 = layers.Dense(config.hidden_size, activation=None, kernel_regularizer=regularizers.l2(0.001))
        self.embedding = PositionEmbedding(config)
        
        # Transformer layers
        self.transformer_layers = [
            layers.MultiHeadAttention(
                num_heads=config.num_attention_heads, 
                key_dim=config.hidden_size // config.num_attention_heads,
                kernel_regularizer=regularizers.l2(0.001)
            )
            for _ in range(config.num_hidden_layers)
        ]
        
        # Layer normalization
        self.layer_norms = [
            layers.LayerNormalization(epsilon=1e-6)
            for _ in range(config.num_hidden_layers)
        ]
        
        # Final classification layer (will be replaced during fine-tuning)
        self.l2 = layers.Dense(n_classes, activation=None, kernel_regularizer=regularizers.l2(0.001))
        self.dropout = layers.Dropout(0.3)
        
        # Optionally freeze pretrained layers
        if freeze_pretrained:
            for layer in [self.l1, self.embedding] + self.transformer_layers + self.layer_norms:
                layer.trainable = False

    def call(self, x, training=False):
        x = self.l1(x)
        x = self.embedding(x)
        
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, x)  # Self-attention
            x = self.layer_norms[i](x)  # Apply LayerNormalization
        
        x = tf.reduce_max(x, axis=1)  # Global max pooling
        x = self.dropout(x, training=training)
        x = self.l2(x)
        return x

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,         # Stop if no improvement for 10 epochs
    restore_best_weights=True  # Restore weights from the best epoch
)

# Configuration
num_features = X_ft_train.shape[2]
print("Num features: ", num_features)

def augment_data(x):
    # Add random noise to the input
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.01)
    return x + noise

# Apply augmentation during training
X_train_augmented = augment_data(X_ft_train)

def fine_tune_and_evaluate_model(run_num):
    # Configuration for the new dataset
    class NewConfig:
        input_size = num_features
        hidden_size = 128
        num_hidden_layers = 4
        num_attention_heads = 8
        max_position_embeddings = 30
        num_classes = NUM_CLASSES  # Number of classes in the new dataset

    new_config = NewConfig()

    # Build the model with the original number of classes (to match the pretrained weights)
    fine_tuned_model = Transformer(new_config, n_classes=NUM_CLASSES)

    # Explicitly build the model to create variables
    fine_tuned_model.build(input_shape=(None, new_config.max_position_embeddings, new_config.input_size))

    # Load pretrained weights (including the original final layer)
    fine_tuned_model.load_weights(model_path)

    # Replace the final classification layer with a new one for the new dataset
    fine_tuned_model.l2 = layers.Dense(
        y_ft_test.shape[1],
        activation=None,
        kernel_initializer='he_normal',  # Use He initialization
        kernel_regularizer=regularizers.l2(0.001)
    )

    # Optionally, unfreeze pretrained layers and fine-tune the entire model
    for layer in fine_tuned_model.layers:
        layer.trainable = True

    # Compile the model with a lower learning rate for fine-tuning
    fine_tuned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy']
    )

    # Fine-tune the entire model
    fine_tuned_history = fine_tuned_model.fit(
        X_train_augmented, 
        y_ft_train, 
        validation_data=(X_ft_val, y_ft_val), 
        epochs=250,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Evaluate on the test set
    test_loss, test_accuracy = fine_tuned_model.evaluate(X_ft_test, y_ft_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate F1 score and other parameters
    y_pred = fine_tuned_model.predict(X_ft_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    f1 = f1_score(y_ft_test.argmax(axis=1), y_pred_classes, average='weighted')
    precision = precision_score(y_ft_test.argmax(axis=1), y_pred_classes, average='weighted')
    recall = recall_score(y_ft_test.argmax(axis=1), y_pred_classes, average='weighted')
    accuracy = accuracy_score(y_ft_test.argmax(axis=1), y_pred_classes)

    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)

    # Save plots for middle run only
    middle_run = N_RUNS // 2 + 1 if N_RUNS % 2 else N_RUNS // 2  # Calculate middle run (e.g., 2 for 3 runs)
    if run_num == middle_run:
        # Create temporary directory for plots
        temp_dir = './temp_plots'
        os.makedirs(temp_dir, exist_ok=True)

        # Plot and save accuracy
        plt.plot(fine_tuned_history.history['accuracy'], label='Train')
        plt.plot(fine_tuned_history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(temp_dir, 'accuracy_plot.png'))
        plt.close()

        # Plot and save confusion matrix
        cm = confusion_matrix(y_ft_test.argmax(axis=1), y_pred_classes)
        plt.figure(figsize=(30, 30))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(top_classes), yticklabels=set(top_classes))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(temp_dir, 'confusion_matrix.png'))
        plt.close()

    # Return metrics for aggregation
    return {
        'test_accuracy': test_accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'epochs': len(fine_tuned_history.history['accuracy'])
    }

# List to store metrics from each run
metrics_list = []

# Run the function N times
for run in range(N_RUNS):
    print(f"\nRun {run + 1}/{N_RUNS}:")
    print("-" * 50)
    metrics = fine_tune_and_evaluate_model(run + 1)
    metrics_list.append(metrics)

# Calculate and print average metrics
print("\nSummary of All Runs:")
print("=" * 50)

# Extract individual metrics into arrays
test_accuracies = [m['test_accuracy'] for m in metrics_list]
f1_scores = [m['f1'] for m in metrics_list]
precisions = [m['precision'] for m in metrics_list]
recalls = [m['recall'] for m in metrics_list]
accuracies = [m['accuracy'] for m in metrics_list]
epchocs = [m['epochs'] for m in metrics_list]

# Print individual run results
for i, metrics in enumerate(metrics_list):
    print(f"Run {i + 1}:")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Epcochs: {metrics['epochs']:.4f}")
    print()

# Print average results
print("Average Metrics Across All Runs:")
print(f"  Average Test Accuracy: {np.mean(test_accuracies):.4f} (±{np.std(test_accuracies):.4f})")
print(f"  Average F1 Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
print(f"  Average Precision: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
print(f"  Average Recall: {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
print(f"  Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
print(f"  Average Number of epochs: {np.mean(epchocs):.4f} (±{np.std(epchocs):.4f})")
print(f"  Number of base classes: {NUM_CLASSES}")
print(f"  Number of ft classes: {NUM_FT_CLASSES}")
print(f"  Number of instances per class: {INSTANCES_PER_CLASS}")

# Prompt for saving plots permanently
temp_dir = './temp_plots'
if os.path.exists(temp_dir):
    save_choice = input("\nDo you want to save the plots permanently? (y/n): ").lower()
    if save_choice == 'y':
        folder_name = input("Enter folder name for saving plots: ")
        result_dir = '../Results/script_results/'
        save_path = os.path.join(result_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Move plots from temporary to permanent location
        for plot_file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, plot_file), os.path.join(save_path, plot_file))
        print(f"Plots saved to {save_path}")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)