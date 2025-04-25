import os
import shutil
import random

# מיקום התמונות המקוריות
original_dataset_dir = "D:\machine learning cats\Asirra_ cat vs dogs"  # <-- תשני לנתיב שלך

# מיקום בסיסי לנתונים החדשים
base_dir = "data"
os.makedirs(base_dir, exist_ok=True)

# יוצרים את תיקיות היעד
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

train_cats_dir = os.path.join(train_dir, 'cats')
os.makedirs(train_cats_dir, exist_ok=True)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.makedirs(train_dogs_dir, exist_ok=True)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.makedirs(validation_cats_dir, exist_ok=True)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.makedirs(validation_dogs_dir, exist_ok=True)

# כל התמונות
all_filenames = os.listdir(original_dataset_dir)

# ערבוב רנדומלי
random.shuffle(all_filenames)

# מחלקים ל-80% אימון ו-20% ולידציה
split_index = int(len(all_filenames) * 0.8)
train_filenames = all_filenames[:split_index]
validation_filenames = all_filenames[split_index:]

# פונקציה שמעבירה קבצים
def copy_images(filenames, target_dir_cats, target_dir_dogs):
    for filename in filenames:
        if 'cat' in filename.lower():
            shutil.copy(os.path.join(original_dataset_dir, filename), os.path.join(target_dir_cats, filename))
        elif 'dog' in filename.lower():
            shutil.copy(os.path.join(original_dataset_dir, filename), os.path.join(target_dir_dogs, filename))

# מעתיקים
copy_images(train_filenames, train_cats_dir, train_dogs_dir)
copy_images(validation_filenames, validation_cats_dir, validation_dogs_dir)

print("Finished copying files!")
