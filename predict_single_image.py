import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# טוענים את המודל ששמרנו
model = tf.keras.models.load_model('saved_model/my_catdog_model.keras')

# טוענים תמונה לניבוי
img_path = 'predict/my_test_image.jpg'  # את יכולה לשים שם כל תמונה!
img = image.load_img(img_path, target_size=(180, 180))
img_array = image.img_to_array(img) / 255.0  # מנרמלים בדיוק כמו באימון
img_array = np.expand_dims(img_array, axis=0)  # הופכים ל-batch של 1

# מבצעים ניבוי
prediction = model.predict(img_array)[0][0]

# מדפיסים את התוצאה
print("תוצאה גולמית:", prediction)
if prediction < 0.5:
    label = 'Cat'
else:
    label = 'Dog'

print(f'נראה לי שזה: {label}')

# מציגים את התמונה
plt.imshow(img)
plt.title(f'Prediction: {label}')
plt.axis('off')
plt.show()
