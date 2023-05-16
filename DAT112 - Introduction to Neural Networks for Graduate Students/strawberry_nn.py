import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Load the strawberries data
data = np.load('strawberries.npz')
x = data['x']
y = data['y']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert categorical labels to begin at 0
y_train = y_train - 1
y_test = y_test - 1

# Define the image dimensions
image_height, image_width = x_train.shape[1], x_train.shape[2]

# Define the neural network architecture
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model:
#    --> the ADAM optimizer is an adaptive learning rate optimization algorithm
#    --> the loss function is sparse_categorical_crossentropy, which converts integer categories to one-hot-encoding
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model; 10 epochs appears sufficient for full minimization of the loss function
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on training data
train_loss, train_acc = model.evaluate(x_train, y_train)
print('The training score is [{:.4f}, {:.4f}]'.format(train_loss, train_acc))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('The test score is [{:.4f}, {:.4f}]'.format(test_loss, test_acc))