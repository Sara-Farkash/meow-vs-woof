import tensorflow as tf
import os

# מגדירים משתנים
batch_size = 32
img_height = 180
img_width = 180

# טוענים את התמונות (אימון)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# טוענים את התמונות (וולידציה)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# אופטימיזציה: הופך את הטעינה למהירה
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# בונים את המודל
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)  # כי יש לנו 2 קטגוריות: חתול או כלב
])

# קומפילציה (הגדרת איך המודל ילמד)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# הצגת סיכום של המודל
model.summary()


epochs = 10  # כמה פעמים נעבור על הדאטה
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# מוודא שהתיקייה קיימת, ואם לא – יוצר אותה
os.makedirs('saved_model', exist_ok=True)

# עכשיו שמור את המודל
model.save('saved_model/my_catdog_model.keras')


# ציור הגרפים!
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()