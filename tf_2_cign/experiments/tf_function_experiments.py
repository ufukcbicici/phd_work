import numpy as np
import tensorflow as tf

temperature = 25.0


# def calculate_regressor(x, y):
#     x_tempered = x / temperature
#     y_hat = tf.keras.layers.Dense(units=1, activation=None)(x_tempered)
#     y_hat = tf.squeeze(y_hat)
#     mse = tf.losses.mean_squared_error(y, y_hat)
#     return mse


batch_size = 250
input_dim = 128
input_x = tf.keras.Input(shape=(input_dim, ))
input_y = tf.keras.Input(shape=())
temp_tf = tf.keras.Input(shape=())

x_tempered = input_x / temp_tf
y_hat = tf.keras.layers.Dense(units=1, activation=None)(x_tempered)
y_hat = tf.squeeze(y_hat)
mse = tf.losses.mean_squared_error(input_y, y_hat)
# loss = calculate_regressor(input_x, input_y)

model = tf.keras.Model(inputs={"input_x": input_x, "input_y": input_y, "temp_tf": temp_tf},
                       outputs=[mse, input_x, x_tempered])
x_arr = np.random.uniform(size=(batch_size, input_dim))
y_arr = 5.0 * np.ones_like(x_arr[:, 0])

l9 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 0.1}, training=True)
l8 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 0.25}, training=True)
l7 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 0.5}, training=True)
l6 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 1.0}, training=True)
l0 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 2.5}, training=True)
l1 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 5.0}, training=True)
l2 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 10.0}, training=True)
l3 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 15.0}, training=True)
l4 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 20.0}, training=True)
l5 = model({"input_y": y_arr, "input_x": x_arr, "temp_tf": 25.0}, training=True)
# l2 = model({"input_y": y_arr, "zfoo": x_arr}, training=True)
# l2 = model([x_arr, y_arr], training=True)
# l3 = model([x_arr, y_arr], training=True)
# temperature = 5.0
# l4 = model({"input_x": x_arr, "input_y": y_arr}, training=True)
print("X")




