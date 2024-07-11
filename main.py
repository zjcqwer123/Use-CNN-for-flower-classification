import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
from glob import glob
from PIL import Image
import scipy.io
# 数据集路径
dataset_path = 'jpg'

# 获取所有图像文件路径
image_paths = glob(os.path.join(dataset_path, '*.jpg'))

import scipy.io

# 标签文件路径
labels_path = 'imagelabels.mat'

# 检查文件是否存在
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"File not found: {labels_path}")

# 读取标签文件
labels_data = scipy.io.loadmat(labels_path)
labels = labels_data['labels'][0]  # 假设标签存储在 'labels' 键中

# 将标签转换为整数
labels = [int(label) for label in labels]

# 打印前10个标签以验证
print(labels[:10])

# 加载图像并调整大小
images = []
for image_path in image_paths:
    img = Image.open(image_path)
    img = img.resize((128, 128))  # 调整图像大小
    images.append(np.array(img))

# 转换为numpy数组
images = np.array(images)
labels = np.array(labels)

# 归一化图像数据
images = images / 255.0

# 将标签进行One-Hot编码
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
# 数据增强
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(102, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)
# 定义学习率调度器
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# 训练模型并使用学习率调度器
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=60, validation_data=(X_test, y_test), callbacks=[reduce_lr])
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
model.save('flower_classification_model.h5')

# 展示分类正确和错误的图片实例
def display_examples(model, X_test, y_test, lb):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    correct_indices = np.where(predicted_labels == true_labels)[0]
    incorrect_indices = np.where(predicted_labels != true_labels)[0]

    # 展示分类正确的图片
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(correct_indices[:5]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx])
        plt.title(f"True: {lb.classes_[true_labels[idx]]}\nPred: {lb.classes_[predicted_labels[idx]]}")
        plt.axis('off')

    # 展示分类错误的图片
    for i, idx in enumerate(incorrect_indices[:5]):
        plt.subplot(2, 5, i + 6)
        plt.imshow(X_test[idx])
        plt.title(f"True: {lb.classes_[true_labels[idx]]}\nPred: {lb.classes_[predicted_labels[idx]]}")
        plt.axis('off')

    plt.show()

display_examples(model, X_test, y_test, lb)