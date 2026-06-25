import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # dont open GUI windows, just save to files
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from tqdm.keras import TqdmCallback

# paths (change these if u move stuff around)
CSV_PATH = 'ml_pipeline/exercise_dataset.csv'
MODEL_SAVE_PATH = 'ml_pipeline/exercise_model.keras'
LABEL_ENCODER_PATH = 'ml_pipeline/label_classes.npy'
SCALER_PATH = 'ml_pipeline/scaler.pkl'

def train_neural_network():
    # ok lets load this massive 305MB beast
    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns. holy moly thats a lot of math")

    # grab the labels and drop the stuff the AI shouldnt see
    # (we dont want it memorizing filenames lol)
    y_text = df['class_name']
    X_raw = df.drop(columns=['class_name', 'video_name', 'frame_number'])

    # turn text labels into numbers cuz neural nets only speak math
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text)
    np.save(LABEL_ENCODER_PATH, label_encoder.classes_)
    print(f"Found {len(label_encoder.classes_)} exercise classes. nice.")

    # 80/20 split - keep 20% hidden so the AI cant cheat
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # scale everything to be centered around 0
    # neural nets go brrrr when numbers are normalized
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # save the scaler so we can reuse it on live webcam data later
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    # -------------------------------------------------------
    # THE BRAIN ARCHITECTURE
    # 4 hidden layers with batch normalization
    # this is where the magic happens
    # -------------------------------------------------------
    print("\nBuilding the neural network...")

    num_features = X_train_scaled.shape[1]  # should be 132
    num_classes = len(label_encoder.classes_)  # should be 27

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(num_features,)),

        # layer 1 - the big boy, 512 neurons
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),  # kill 40% of connections to prevent memorization

        # layer 2
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        # layer 3
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # layer 4 - the narrow bottleneck
        tf.keras.layers.Dense(64, activation='relu'),

        # output - 27 neurons, one per exercise, softmax gives us percentages
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # -------------------------------------------------------
    # TRAINING TIME BABYYY
    # -------------------------------------------------------
    print("\nStarting training... grab some popcorn\n")

    # if the model stops improving for 10 epochs, just stop
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # if the model gets stuck, cut the learning rate in half
    # its like telling it "ok slow down and be more careful"
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # tqdm gives us that sweet progress bar
    tqdm_cb = TqdmCallback(verbose=0)

    history = model.fit(
        X_train_scaled, y_train,
        epochs=150,  # max 150 but early stopping will probably kick in way before
        batch_size=256,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stop, reduce_lr, tqdm_cb],
        verbose=0  # let tqdm handle the progress display
    )

    # -------------------------------------------------------
    # TIME TO SEE HOW SMART THIS THING ACTUALLY IS
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)

    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")

    # get predictions
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # top-5 accuracy (was the correct answer in the AIs top 5 guesses?)
    top5 = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    print(f"  Top-5 Accuracy: {top5 * 100:.2f}%")

    # the big detailed report - precision, recall, f1 for every single class
    print("\n--- PER-CLASS REPORT ---")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # -------------------------------------------------------
    # CONFUSION MATRIX - saved as a beautiful PNG
    # shows exactly where the AI gets confused
    # -------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Where Does The AI Get Confused?', fontsize=16)
    plt.ylabel('Actual Exercise', fontsize=13)
    plt.xlabel('What The AI Thought It Was', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('ml_pipeline/confusion_matrix.png', dpi=150)
    plt.close()
    print("  Confusion matrix saved -> ml_pipeline/confusion_matrix.png")

    # -------------------------------------------------------
    # TRAINING HISTORY - accuracy and loss over time
    # -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('ml_pipeline/training_history.png', dpi=150)
    plt.close()
    print("  Training history saved -> ml_pipeline/training_history.png")

    # -------------------------------------------------------
    # TOP 5 MOST CONFUSED PAIRS
    # aka "which exercises look identical to the AI"
    # -------------------------------------------------------
    print("\n--- TOP 5 MOST CONFUSED PAIRS ---")
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # ignore correct predictions
    flat_indices = np.argsort(cm_copy.ravel())[::-1]
    pairs = np.dstack(np.unravel_index(flat_indices, cm_copy.shape))[0]
    for i, (true_idx, pred_idx) in enumerate(pairs[:5]):
        count = cm_copy[true_idx, pred_idx]
        print(f"  {i+1}. {label_encoder.classes_[true_idx]:<25} mistaken for {label_encoder.classes_[pred_idx]:<25} ({count} times)")

    # -------------------------------------------------------
    # SAVE EVERYTHING
    # -------------------------------------------------------
    model.save(MODEL_SAVE_PATH)
    print(f"\n  Model saved  -> {MODEL_SAVE_PATH}")
    print(f"  Labels saved -> {LABEL_ENCODER_PATH}")
    print(f"  Scaler saved -> {SCALER_PATH}")
    print("\ndone!! the AI brain is ready to go")

if __name__ == "__main__":
    train_neural_network()