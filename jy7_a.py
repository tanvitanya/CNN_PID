#patience = 5
#AUTOTUNE shuffle 
# tf seed
#Save Model
# padding='same'
# Image deleted...

#jy6_a
#In conv2D : ,kernel_regularizer=regularizers.l2(1e-4)
#In Dense layer : ,    kernel_regularizer=regularizers.l2(1e-4)
#Replaced : x = layers.Flatten()(x) to x = layers.GlobalAveragePooling2D()(x)

#jy6_b
#Replace back : x = layers.GlobalAveragePooling2D()(x) to x = layers.Flatten()(x)

#jy7_a (from :  jy6_b.py)
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
#history = model.fit(train_ds, validation_data=test_ds, epochs=40, callbacks=[early_stop, reduce_lr])



import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ==========================================
# 1. SETUP & START LOGGER FIRST
# ==========================================
img_height, img_width = 100, 100
batch_size = 32
class_names = ['11', '211', '321', '2212']
data_dir = "8020" 
output_dir = "jy7_a" 
model_output_path = "Model_jy7_a.keras"

os.makedirs(output_dir, exist_ok=True) 

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout 
        self.log = open(filename, "w") 
        
    def write(self, message):
        self.terminal.write(message) 
        self.log.write(message)      
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# START LOGGING IMMEDIATELY
log_path = os.path.join(output_dir, "log.txt")
sys.stdout = DualLogger(log_path)
print(f"Directory created. Logging all terminal output to both screen and: {log_path}")

# ==========================================
# 2. EXTRACT & NORMALIZE DATA
# ==========================================
print("Parsing filenames and extracting momentum...")

paths = []
momentums = []
labels = []

for class_index, c_name in enumerate(class_names):
    search_pattern = os.path.join(data_dir, c_name, "*.png")
    files = glob.glob(search_pattern)
    
    for f in files:
        paths.append(f)
        labels.append(class_index) 
        
        basename = os.path.basename(f)
        
        try:
            mom = float(basename.split('_')[2])
            momentums.append(mom)
        except:
            momentums.append(0.0)

paths = np.array(paths)
#Why int32qqqqqqqqqqq
labels = np.array(labels, dtype=np.int32)
momentums = np.array(momentums, dtype=np.float32)

# : NORMALIZE MOMENTUM
max_mom = np.max(momentums)
if max_mom > 0:
    momentums = momentums / max_mom
print(f"Momentum normalized. Max raw value was {max_mom}.")

# ==========================================
# 3. SHUFFLE & SPLIT
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)
indices = np.arange(len(paths))
np.random.shuffle(indices)

paths, momentums, labels = paths[indices], momentums[indices], labels[indices]

split_idx = int(len(paths) * 0.8)

train_paths, test_paths = paths[:split_idx], paths[split_idx:]
train_moms, test_moms = momentums[:split_idx], momentums[split_idx:]
train_labels, test_labels = labels[:split_idx], labels[split_idx:]

print(f"Total images found: {len(paths)}")
print(f"Training on {len(train_paths)} images, testing on {len(test_paths)} images.")

# ==========================================
# 4. DATA PIPELINE
# ==========================================
def load_image_and_data(path, mom, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32) # Normalizes to 0.0 - 1.0
    img = tf.image.resize(img, [img_height, img_width])
    
    # ---> NEW: Convert to pure Black and White (Binary) <---
    # Any pixel brighter than 0.2 becomes 1.0 (white). Everything else becomes 0.0 (black).
    threshold = 0.2 
    img = tf.where(img > threshold, 1.0, 0.0) 
    
    return ({"image_input": img, "momentum_input": mom}, label)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_moms, train_labels))
    .shuffle(1000)
    .map(load_image_and_data, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_paths, test_moms, test_labels))
    .map(load_image_and_data, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# ==========================================
# 5. BUILD MODEL
# ==========================================
image_input = layers.Input(shape=(img_height, img_width, 1), name="image_input")
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(1e-4))(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
    kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Flatten()(x)

cnn_output = layers.Dense(64, activation='relu',
    kernel_regularizer=regularizers.l2(1e-4))(x)
cnn_output = layers.Dropout(0.2)(cnn_output) 

momentum_input = layers.Input(shape=(1,), name="momentum_input")
combined = layers.Concatenate()([cnn_output, momentum_input])

z = layers.Dense(32, activation='relu',
    kernel_regularizer=regularizers.l2(1e-4))(combined)
z = layers.Dropout(0.2)(z) 
final_output = layers.Dense(len(class_names), activation='softmax')(z)

model = models.Model(inputs=[image_input, momentum_input], outputs=final_output)

# ---> CRITICAL FIX 2: LOWER LEARNING RATE <--
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# ==========================================
# 6. TRAIN
# ==========================================
print("\nStarting Training...")
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(train_ds, validation_data=test_ds, epochs=100, callbacks=[early_stop, reduce_lr])

#Save the  trained model

model_path = os.path.join(output_dir, model_output_path)
model.save(model_path)
print(f"Model saved to {model_path}")

# ==========================================
# 7. VISUALIZE & SAVE
# ==========================================
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model Accuracy (With Mom info)')

accuracy_path = os.path.join(output_dir, 'training_accuracy_plot.png')
plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
print(f"\nSaved accuracy plot to {accuracy_path}")
plt.close() 

print("Generating Confusion Matrices...")
test_preds = model.predict(test_ds)
predicted_labels = np.argmax(test_preds, axis=1)

cm_global = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_global, annot=True, fmt='d', cmap='Purples', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Global Confusion Matrix With Mom info (All Mom)')
plt.ylabel('True Particle ID')
plt.xlabel('Predicted Particle ID')

global_cm_path = os.path.join(output_dir, 'global_confusion_matrix.png')
plt.savefig(global_cm_path, dpi=300, bbox_inches='tight')
print(f"Saved global confusion matrix to {global_cm_path}")
plt.close()

# Re-multiply by max_mom so the charts show real physics values
unique_moms = np.unique(test_moms)
unique_moms.sort()

# ---> NEW CODE: 1. Set up empty lists before the loop starts <---
moments_list = []
accuracies_list = []

for norm_mom in unique_moms:
    mask = (test_moms == norm_mom)
    true_binned = test_labels[mask]
    pred_binned = predicted_labels[mask]
    
    if len(true_binned) == 0:
        continue
        
    cm_binned = confusion_matrix(true_binned, pred_binned, labels=[0, 1, 2, 3])
    real_mom = norm_mom * max_mom
    
    # ---> NEW CODE: 2. Calculate accuracy and save to lists <---
    acc = np.trace(cm_binned) / np.sum(cm_binned)
    moments_list.append(real_mom)
    accuracies_list.append(acc)
    # -----------------------------------------------------------
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_binned, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Momentum = {real_mom:.1f} GeV/c)')
    plt.ylabel('True Particle ID')
    plt.xlabel('Predicted Particle ID')
    
    filename = f'cm_mom_{real_mom:.1f}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
print(f"Saved {len(unique_moms)} individual momentum confusion matrices to the '{output_dir}' folder!")

# ---> NEW CODE: 3. Draw the final summary graph after the loop finishes <---
plt.figure(figsize=(10, 6))
plt.plot(moments_list, accuracies_list, marker='o', linestyle='-', color='darkred', linewidth=2)
plt.title('PID Accuracy vs. Momentum')
plt.xlabel('Momentum (GeV/c)')
plt.ylabel('PID Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0, 1.05]) # Keeps y-axis between 0 and 100%

# Save and Show the summary graph
summary_path = os.path.join(output_dir, 'accuracy_vs_mom.png')
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"Saved summary accuracy plot to {summary_path}")
#plt.show()
# --------------------------------------------------------------------------
