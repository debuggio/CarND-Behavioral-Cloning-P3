import csv
import cv2
import numpy as np

from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

import sklearn
from sklearn.model_selection import train_test_split


def get_data_lines():
	lines = []
	with open('data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	return lines


def process_row(data_row):
	images, steering_angles = [], []

	path = "./data/IMG/"
	correction = 0.25

	steering_center = float(data_row[3])
	#steering_left = steering_center + correction
	#steering_right = steering_center - correction

	steering_center_flipped = steering_center * -1.
	#steering_left_flipped = steering_left * -1.
	#steering_right_flipped = steering_right * -1.

	img_center = preprocess_image(cv2.imread(path + data_row[0].split('\\')[-1]))
	#img_left = preprocess_image(cv2.imread(path + data_row[1].split('\\')[-1]))
	#img_right = preprocess_image(cv2.imread(path + data_row[2].split('\\')[-1]))

	img_center_flipped = flip_image(img_center)
	#img_left_flipped = flip_image(img_left)
	#img_right_flipped = flip_image(img_right)  

	#images.extend([img_center, img_left, img_right])
	images.extend([img_center])
	#steering_angles.extend([steering_center, steering_left, steering_right])
	steering_angles.extend([steering_center])

	return images, steering_angles


def preprocess_image(image):
	return image[50:140,:,:]


def flip_image(image):
	return cv2.flip(image, 1)


def generator(data_rows, batch_size=32):
	num_rows = len(data_rows)
	while True:
		np.random.shuffle(data_rows)
		for offset in range(0, num_rows, batch_size):
			current_data_rows = data_rows[offset : offset+batch_size]

			current_images = []
			current_steering_angles = []
			for current_data_row in current_data_rows:
				images, steering_angles = process_row(current_data_row)

				current_images.extend(images)
				current_steering_angles.extend(steering_angles)

			X_train = np.array(current_images)
			Y_train = np.array(current_steering_angles)

			yield sklearn.utils.shuffle(X_train, Y_train)


def build_model(input_width, input_height):
	model = Sequential()
	model.add(Lambda(lambda x: x / 255. - .5, input_shape=(input_height, input_width, 3)))
	model.add(Conv2D(24, (5 ,5), activation="relu", strides=(2, 2)))
	model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
	model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation="relu"))
	model.add(Conv2D(64, (3, 3), activation="relu"))
	model.add(Dropout(.5))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	return model


def train_and_save_model(model, train_generator, train_data_length, validation_generator, validation_data_length):
	model.compile(loss='mse', optimizer='adam')
	model.fit_generator(train_generator, samples_per_epoch=train_data_length, validation_data=validation_generator, nb_val_samples=validation_data_length, nb_epoch=3)

	model.save('model.h5')


def main():
	target_image_height = 90
	target_image_width = 320

	data_lines = get_data_lines()

	train_data, validation_data = train_test_split(data_lines, test_size=0.2)

	train_generator = generator(train_data, batch_size=128)
	validation_generator = generator(validation_data, batch_size=128)

	model = build_model(target_image_width, target_image_height)
	train_and_save_model(model, train_generator, len(train_data), validation_generator, len(validation_data))


if __name__ == '__main__':
	main()