import os
import json
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import argparse
import sys
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
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping

data_directory = '../all_outputs/output_ssl_small/'

# list all the files in directories and sub directories in in data_direcotiry and store them list_list list
file_names = []

for root, _, files in os.walk(data_directory):  # Walk through all directories and subdirectories
        for filename in files:
            file_names.append(os.path.join(root, filename))

print(len(file_names))    

parser = argparse.ArgumentParser()
parser.add_argument('--runid', type=int, default=False)
parser.add_argument('--instances', type=int, required=True)
args = parser.parse_args()

INSTANCES_PER_CLASS = args.instances
RUN_ID = args.runid if args.runid is not False else False

try:
    INSTANCES_PER_CLASS = int(INSTANCES_PER_CLASS)
    if RUN_ID is not False:
        RUN_ID = int(RUN_ID)
except (TypeError, ValueError):
    print("❌ Error: INSTANCES_PER_CLASS and BASE_SIZE must be valid integers.")
    sys.exit(1)


all_data = []
N_RUNS = 1
TESTING_INSTANCES_START_INDEX = 6

pose_indices = [0, 15, 16, 17, 18, 19, 20]
hand_indices = [0, 4, 7, 8, 11, 12, 15, 16, 19, 20]

# pose_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
# hand_indices = [0, 1, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def load_landmarks(filenames, num_frames, glossIndex = 4):
    
    
    # Prepare storage for data and labels
    video_data = []
    labels = []
    
    for file in filenames:
     
        # print(file)
        gloss = file.split('/')[glossIndex]
        # print(gloss)


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
                        
                    # print(left_hand_points)
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

video_data, labels = load_landmarks(file_names, num_frames, 3)

# `video_data` has shape (number of videos, num_frames, number of points*2) where points*2 is the flattened landmark count
# `labels` contains the labels for each video


NUM_CLASSES = len(set(labels))



# Set the number of instances per class in the training set

CUSTOM_NUM_CLASSES = NUM_CLASSES  # Assuming NUM_CLASSES is defined elsewhere

# Count the occurrences of each class
class_counts = Counter(labels)

# Sort classes by their counts in descending order and select the top NUM_CLASSES classes
top_classes = [cls for cls, count in class_counts.most_common(CUSTOM_NUM_CLASSES)]

# Filter the dataset to keep only the samples belonging to the top classes
filtered_indices = [i for i, label in enumerate(labels) if label in top_classes]
video_data_filtered = video_data[filtered_indices]
labels_filtered = labels[filtered_indices]

# Encode labels after filtering
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_filtered)

# Custom split to ensure fixed instances per class in training set
train_indices = []
val_test_indices = []

for cls in range(len(top_classes)):
    # Get indices of samples for this class
    cls_indices = np.where(labels_encoded == cls)[0]
    np.random.shuffle(cls_indices)  # Randomize the order
    # Ensure there are enough samples for the class
    if len(cls_indices) < INSTANCES_PER_CLASS:
        raise ValueError(f"Class {cls} has only {len(cls_indices)} instances, but {INSTANCES_PER_CLASS} are required.")
    # Take fixed number of instances for training
    train_indices.extend(cls_indices[:INSTANCES_PER_CLASS])
    # Remaining instances go to validation/test
    val_test_indices.extend(cls_indices[TESTING_INSTANCES_START_INDEX:])

# Convert to numpy arrays
X_train = video_data_filtered[train_indices]
y_train = labels_encoded[train_indices]
X_temp = video_data_filtered[val_test_indices]
y_temp = labels_encoded[val_test_indices]

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(top_classes))
y_temp = to_categorical(y_temp, num_classes=len(top_classes))

# Split remaining data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Print train, val, test shapes
print("Train shapes:", X_train.shape, y_train.shape)
print("Validation shapes:", X_val.shape, y_val.shape)
print("Test shapes:", X_test.shape, y_test.shape)

# Verify and print number of instances per class in training set
y_train_indices = np.argmax(y_train, axis=1)
class_counts_train = Counter(y_train_indices)
print("Number of instances per class in training set:")
print(f"Total unique classes: {len(class_counts_train)}")
print(f"Average instances per class: {sum(class_counts_train.values()) / len(class_counts_train):.2f}")

NUM_CLASSES = CUSTOM_NUM_CLASSES


def augment_data(x):
    # Add random noise to the input
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.01)
    return x + noise

# Apply augmentation during training
X_train_augmented = augment_data(X_train)

print(X_train_augmented.shape, y_train.shape)




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
num_features = X_train_augmented.shape[2]
print("Num features: ", num_features)


# Assuming Transformer, num_features, X_train_augmented, y_train, X_val, y_val, X_test, y_test, early_stopping, and NUM_CLASSES are defined elsewhere

def create_train_evaluate_model(run_num):
    class Config:
        input_size = num_features
        hidden_size = 128
        num_hidden_layers = 4
        num_attention_heads = 8
        max_position_embeddings = 30  # Maximum number of frames
        num_classes = NUM_CLASSES  # Number of output classes

    config = Config()

    # Build the model
    model = Transformer(config, n_classes=config.num_classes)

    # Print the model summary
    model.build(input_shape=(None, config.max_position_embeddings, config.input_size))
    # model.summary()

    # Convert one-hot encoded labels to class indices
    y_train_indices = np.argmax(y_train, axis=1)
    y_val_indices = np.argmax(y_val, axis=1)
    y_test_indices = np.argmax(y_test, axis=1)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),  # Unique class indices
        y=y_train_indices                    # Class indices for training data
    )
    class_weights = dict(enumerate(class_weights))  # Convert to dictionary

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy']
    )

    # Train the model with class weights
    history = model.fit(
        X_train_augmented, 
        y_train, 
        validation_data=(X_val, y_val), 
        epochs=250, 
        batch_size=32, 
        callbacks=[early_stopping],
        class_weight=class_weights
    )

    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate F1 score and other parameters
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate metrics
    f1 = f1_score(y_test.argmax(axis=1), y_pred_classes, average='weighted')
    precision = precision_score(y_test.argmax(axis=1), y_pred_classes, average='weighted')
    recall = recall_score(y_test.argmax(axis=1), y_pred_classes, average='weighted')
    accuracy = accuracy_score(y_test.argmax(axis=1), y_pred_classes)

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
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(temp_dir, 'accuracy_plot.png'))
        plt.close()

        # Plot and save confusion matrix
        cm = confusion_matrix(y_test.argmax(axis=1), y_pred_classes)
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
        'epochs': len(history.history['accuracy'])
    }



# List to store metrics from each run
metrics_list = []

# Run the function N times
for run in range(N_RUNS):
    print(f"\nRun {run + 1}/{N_RUNS}:")
    print("-" * 50)
    metrics = create_train_evaluate_model(run + 1)
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
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  Number of instances per class: {INSTANCES_PER_CLASS}")

output_file = '../Results/script_results/logs/no_ft_' + str(NUM_CLASSES) + '_' + str(INSTANCES_PER_CLASS) + '.txt'

with open(output_file, "a") as f:
    f.write("# Individual Run Results\n")
    for i, metrics in enumerate(metrics_list):
        f.write(f"Run {i + 1}:\n")
        f.write(f"  Test Accuracy: {metrics['test_accuracy']:.4f}\n")
        f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Epochs: {metrics['epochs']:.4f}\n")
        f.write("\n")

    f.write("Average Metrics Across All Runs:\n")
    f.write(f"  Average Test Accuracy: {np.mean(test_accuracies):.4f} (±{np.std(test_accuracies):.4f})\n")
    f.write(f"  Average F1 Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})\n")
    f.write(f"  Average Precision: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})\n")
    f.write(f"  Average Recall: {np.mean(recalls):.4f} (±{np.std(recalls):.4f})\n")
    f.write(f"  Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})\n")
    f.write(f"  Average Number of Epochs: {np.mean(epchocs):.4f} (±{np.std(epchocs):.4f})\n")
    f.write(f"  Number of classes: {NUM_CLASSES}\n")
    f.write(f"  Number of instances per class: {INSTANCES_PER_CLASS}\n")
    f.write("="*50 + "\n\n")  # Separator between runs

# Prompt for saving plots permanently
temp_dir = './temp_plots'
if os.path.exists(temp_dir):
    # save_choice = input("\nDo you want to save the plots permanently? (y/n): ").lower()
    # if save_choice == 'y':
    if True:
        # folder_name = input("Enter folder name for saving plots: ")
        folder_name = f"no_ft_{NUM_CLASSES}_class_{INSTANCES_PER_CLASS}_instance"
        if RUN_ID is not False:
            folder_name += f"_{RUN_ID}"
        
        result_dir = '../Results/script_results/'
        save_path = os.path.join(result_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Move plots from temporary to permanent location
        for plot_file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, plot_file), os.path.join(save_path, plot_file))
        print(f"Plots saved to {save_path}")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)