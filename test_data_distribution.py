import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

# מגדירים את התיקיה שממנה נציג תמונה לדוגמה
sample_dir = r'D:\machine learning cats\data\train\dogs'  # לדוגמה: חתולים מאימון

# מביאים קובץ רנדומלי מתוך התיקיה
sample_images = os.listdir(sample_dir)
random_image = random.choice(sample_images)

# טוענים את התמונה
img_path = os.path.join(sample_dir, random_image)
img = mpimg.imread(img_path)

# מציגים את התמונה
plt.imshow(img)
plt.title(f"Random Cat Image - {random_image}")
plt.axis('off')
plt.show()
