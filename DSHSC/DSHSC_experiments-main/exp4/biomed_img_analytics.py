import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Dataset paths
train_path = r"D:\DS Practice\DSHSC\DSHSC_experiments-main\exp4\dataset\testing"
test_path = r"D:\DS Practice\DSHSC\DSHSC_experiments-main\exp4\dataset\testing"

classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Extract HOG features
def extract_features(folder):
    X = []
    y = []

    for label, class_name in enumerate(classes):
        class_path = os.path.join(folder, class_name)

        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)

            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (64, 64))

            features = hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2)
            )

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


# Load training and testing data
X_train, y_train = extract_features(train_path)
X_test, y_test = extract_features(test_path)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Feature size:", X_train.shape[1])

# Train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Brain MRI Classification - Confusion Matrix")
plt.tight_layout()
plt.show()