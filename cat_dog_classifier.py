import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (100, 100))  # Resize image to 100x100
            images.append(img.flatten())  # Flatten image
            label = folder.split('/')[-1]  # Extract label from folder name
            labels.append(label)
    return images, labels

# Load images
cat_images, cat_labels = load_images_from_folder('train/cats')
dog_images, dog_labels = load_images_from_folder('train/dogs')

# Combine datasets
X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
