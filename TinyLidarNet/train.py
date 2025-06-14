#Requirement Library
import os
import sys
print(sys.executable)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # If you want utilize GPU, uncomment this line
from sklearn.utils import shuffle
import rosbag
import time
import subprocess
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.losses import huber
from tensorflow.keras.optimizers import Adam

# Check GPU availability - You don't need a gpu to train this model
# gpu_available = tf.test.is_gpu_available()
gpu_available = tf.config.list_physical_devices('GPU')
gpu_available = len(gpu_available) > 0
print('GPU AVAILABLE:', gpu_available)

#========================================================
# Functions
#========================================================
#Linear maping
def linear_map(x, x_min, x_max, y_min, y_max):
    """Linear mapping function."""
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

#Huber Loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    mean_loss = np.mean(loss)
    return mean_loss
#========================================================
# Global Data
#========================================================

# Initialize lists for data
lidar = []
servo = []
speed = []
test_lidar = []
test_servo = []
test_speed = []
model_name = 'TLN'
# model_files = [
#     './Models/'+model_name+'_noquantized.tflite',
#     './Models/'+model_name+'_int8.tflite'
# ]
# dataset_path = [
#     './Dataset/out.bag', 
#     './Dataset/f2.bag', 
#     './Dataset/f4.bag',
# ]
# loss_figure_path = './Figures/loss_curve.png'
model_files = [
    './Models/'+model_name+'_trackdata_noquantized.tflite',
    './Models/'+model_name+'_trackdata_int8.tflite'
]

# dataset_path = [
    # './Dataset/Speilberg/gapfollower_gym_data2(6,100).csv', 
    # './Dataset/Speilberg/keyboard_data4(5).csv',
    #  './Dataset/Speilberg/Gapfollower_data3(6,100).csv',
    # './Dataset/Sochi/Gapfollower_data2(6,100).csv',
    # './Dataset/Silverstone/Gapfollower_data2(6,100).csv',
    # './Dataset/Oschersleben/Gapfollower_data2(6,200).csv',
    # './Dataset/Oschersleben/keyboard_data2(4).csv',
    # './Dataset/Silverstone/keyboard_data2(4).csv',
    # './Dataset/Sochi/keyboard_data2(4).csv',
    # './Dataset/real_car_data.csv'
    
# ]
dataset_path = [
    './f1tenth_ws/rosbag/stony5_scan.csv',
    './f1tenth_ws/rosbag/stony5_teleop.csv',
    # './f1tenth_ws/rosbag/stony1.bag'
]
print("-----------------------------------")
print(f"dataset_path: {dataset_path}")
loss_figure_path = './Figures/gym_loss_curve.png'

down_sample_param = 2 # Down-sample Lidar data
lr = 5e-5
loss_function = 'huber'
batch_size = 64
num_epochs = 20
hz = 40

# Initialize variables for min and max speed
# max_speed = 0
max_speed = 9
min_speed = 0

#========================================================
# Get Dataset
#========================================================
i=0
# Iterate through bag files
for pth in dataset_path:
#     if not os.path.exists(pth):
#         print("out.bag doesn't exist in ", pth)
#         exit(0)
#     good_bag = rosbag.Bag(pth)

    lidar_data = []
    servo_data = []
    speed_data = []

#     # Read messages from bag file
#     for topic, msg, t in good_bag.read_messages():
#         # if topic == 'Lidar':
#         if topic == '/scan':
#             ranges = msg.ranges[::down_sample_param]
#             lidar_data.append(ranges)
#         # if topic == 'Ackermann':
#         if topic == '/vesc/low_level/ackermann_cmd_mux/input/teleop':
#             data = msg.drive.steering_angle
#             s_data = msg.drive.speed
            
#             servo_data.append(data)
#             if s_data > max_speed:
#                 max_speed = s_data
#             speed_data.append(s_data)
    # Read messages from bag file

    df = pd.read_csv(pth)
    # print(type(df['lidar']))
    # print(df['lidar'])

    # for lidar_msg in df['lidar']:
        # print(type(lidar_msg))
        # print(lidar_msg)
    # columns = [col for col in df.columns]
    # print(columns)

    if 'field.ranges0' in df.columns:
        print("run1")
        range_columns = [col for col in df.columns if col.startswith("field.ranges")]

        for _, row in df.iterrows():
            lidar_msg = row[range_columns].values.tolist()

        # lidar_msg = lidar_msg.replace('[', '').replace(']', '').split(',')
        # print(f"lidar msg:\n{lidar_msg}")
        # lidar_msg = [float(value) for value in lidar_msg]
            ranges = lidar_msg[::down_sample_param]
            lidar_data.append(ranges)
    
    if 'field.drive.steering_angle' in df.columns: 
        print("run")
        for servo_msg in df['field.drive.steering_angle']:
            servo_data.append(servo_msg)
        # print(df['drive.steering_angle'])

    # print(df['drive.speed'])
    if 'field.drive.speed' in df.columns: 
        for speed_msg in df['field.drive.speed']:        
            if speed_msg > max_speed:
                speed_msg = max_speed  
            speed_data.append(float(speed_msg))

 
        # lidar_msg = lidar_msg.replace('[', '').replace(']', '').split(',')
        # print(f"lidar msg:\n{lidar_msg}")
        # lidar_msg = [float(value) for value in lidar_msg]
        # ranges = lidar_msg[::down_sample_param]
        # lidar_data.append(ranges)

#     print("run")
#     for servo_msg in df['steering_angle']:
#         servo_data.append(servo_msg)
#     # print(df['drive.steering_angle'])

# # print(df['drive.speed'])
#     for speed_msg in df['speed']:        
#         if speed_msg > max_speed:
#             speed_msg = max_speed  
#         speed_data.append(float(speed_msg))
    
    # print("-----------------------------lidar\n")
    # print(lidar_data)
    # print("-----------------------------speed\n")
    # print(speed_data)
    # print("-----------------------------servo\n")
    # print(servo_data)

    # Convert data to arrays
    lidar_data = np.array(lidar_data) 
    servo_data = np.array(servo_data)
    speed_data = np.array(speed_data)
    # print(f"lidar_data: {len(lidar_data)}")
    # print(f"speed_data: {len(speed_data)}")
    print("lidar.shape: ", lidar_data.shape)


    # Shuffle data
    shuffled_data = shuffle(np.concatenate((servo_data[:, np.newaxis], speed_data[:, np.newaxis]), axis=1), random_state=62)
    shuffled_lidar_data = shuffle(lidar_data, random_state=62)
    print(f"shuffled_lidar_data: {len(shuffled_lidar_data)}")
    print(f"shuffled_data: {len(shuffled_data)}")

    # Split data into train and test sets
    train_ratio = 0.85
    if len(shuffled_lidar_data) != 0:
        train_samples = int(train_ratio * len(shuffled_lidar_data))
    else:
        train_samples = int(train_ratio * len(shuffled_data))
    x_train_bag, x_test_bag = shuffled_lidar_data[:train_samples], shuffled_lidar_data[train_samples:]
    print(f"x_train_bag(lidar): {len(x_train_bag)}")
    print(f"x_test_bag(lidar): {len(x_test_bag)}")

    # Extract servo and speed values
    y_train_bag = shuffled_data[:train_samples]
    y_test_bag = shuffled_data[train_samples:]
    print(f"y_train_bag(servo, speed): {len(y_train_bag)}")
    print(f"y_test_bag(servo, speed): {len(y_test_bag)}")
    # print(f"x_train_bag format: {type(x_train_bag)}")
    # print(f"y_train_bag(servo) format: {type(y_train_bag[:, 0])}")
    # print(f"y_train_bag(speed) format: {type(y_train_bag[:, 1])}")

    # Extend lists with train and test data
    lidar.extend(x_train_bag)
    servo.extend(y_train_bag[:, 0])
    speed.extend(y_train_bag[:, 1])
    # print(f"lidar format: {type(lidar)}")
    # print(f"servo format: {type(servo)}")
    # print(f"speed format: {type(speed)}")

    test_lidar.extend(x_test_bag)
    test_servo.extend(y_test_bag[:, 0])
    test_speed.extend(y_test_bag[:, 1])
    print(f"num{i}")
    i+=1
    print(f'\nData in {pth}:')
    print(f'Shape of Train Data --- Lidar: {len(lidar)}, Servo: {len(servo)}, Speed: {len(speed)}')
    print(f'Shape of Test Data --- Lidar: {len(test_lidar)}, Servo: {len(test_servo)}, Speed: {len(test_speed)}')

# Calculate total number of samples
total_number_samples = len(lidar)

print(f'Overall Samples = {total_number_samples}')
lidar = np.asarray(lidar)
servo = np.asarray(servo)
speed = np.asarray(speed)

speed = linear_map(speed, min_speed, max_speed, 0, 1)
test_lidar = np.asarray(test_lidar)
test_servo = np.asarray(test_servo)
test_speed = np.asarray(test_speed)
test_speed = linear_map(test_speed, min_speed, max_speed, 0, 1)

print(f'Min_speed: {min_speed}')
print(f'Max_speed: {max_speed}')
print(f'Loaded {len(lidar)} Training samples ---- {(len(lidar)/total_number_samples)*100:0.2f}% of overall')
print(f'Loaded {len(test_lidar)} Testing samples ---- {(len(test_lidar)/total_number_samples)*100:0.2f}% of overall\n')

# Check array shapes
assert len(lidar) == len(servo) == len(speed)
assert len(test_lidar) == len(test_servo) == len(test_speed)

#======================================================
# Split Dataset
#======================================================

print('Splitting Data into Train/Test')
train_data = np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1)
test_data =  np.concatenate((test_servo[:, np.newaxis], test_speed[:, np.newaxis]), axis=1)
# Check array shapes
print(f'Train Data(lidar): {lidar.shape}')       #Train Data(lidar): (4040, 541)
print(f'Train Data(servo, speed): {servo.shape}, {speed.shape}')
print(f'Test Data(lidar): {test_lidar.shape}')
print(f'Test Data(servo, speed): {test_servo.shape}, {test_speed.shape}')

#======================================================
# DNN Arch
#======================================================

num_lidar_range_values = len(lidar[0]) 
print(f'num_lidar_range_values: {num_lidar_range_values}')     #num_lidar_range_values: 541

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='tanh')
    # tf.keras.layers.Dense(2, activation='linear')
])
# print("linear")
#======================================================
# Model Compilation
#======================================================

optimizer = Adam(lr)
model.compile(optimizer=optimizer, loss=loss_function)
print(model.summary())

#======================================================
# Model Fit
#======================================================
# class DebugCallback(tf.keras.callbacks.Callback):
#     def on_train_batch_end(self, batch, logs=None):
#         if logs is not None and tf.math.is_nan(logs.get('loss', 0)):
#             print(f"Nan loss detected at batch {batch}")
#             self.model.stop_training = True

# debug_callback = DebugCallback()
start_time = time.time()
# history = model.fit(lidar, np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1),
#                     epochs=num_epochs, batch_size=batch_size, validation_data=(test_lidar, test_data), callbacks=[debug_callback])
history = model.fit(lidar, np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1),
                    epochs=num_epochs, batch_size=batch_size, validation_data=(test_lidar, test_data))

print(f'=============>{int(time.time() - start_time)} seconds<=============')

# Plot training and validation losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(loss_figure_path)
plt.close()

#======================================================
# Model Evaluation
#======================================================

print("==========================================")
print("Model Evaluation")
print("==========================================")

# Evaluate test loss
test_loss = model.evaluate(test_lidar, test_data)
print(f'Overall Test Loss = {test_loss}')

# Calculate and print overall evaluation
y_pred = model.predict(test_lidar)
hl = huber_loss(test_data, y_pred)
print('\nOverall Evaluation:')
print('Overall Huber Loss: {:.3f}'.format(hl))


# Calculate and print speed evaluation
speed_y_pred = model.predict(test_lidar)[:, 1]
speed_test_loss = huber_loss(test_data[:, 1], speed_y_pred)
print("\nSpeed Evaluation:")
print("Speed Test Loss: ", speed_test_loss)

# Calculate and print servo evaluation
servo_y_pred = model.predict(test_lidar)[:, 0]
servo_test_loss = huber_loss(test_data[:, 0], servo_y_pred)
print("\nServo Evaluation:")
print("Servo Test Loss: ",servo_test_loss)

#======================================================
# Save Model
#======================================================
# Save non-quantized model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# tflite_model_path = './Models/' + model_name + "_noquantized.tflite"
tflite_model_path = './Models/' + model_name + "_trackdata_noquantized.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
    print(f"{model_name}_noquantized.tflite is saved.")

# Save int8 quantized model
rep_32 = lidar.astype(np.float32)
rep_32 = np.expand_dims(rep_32, -1)
dataset = tf.data.Dataset.from_tensor_slices(rep_32)

def representative_data_gen():
    for input_value in dataset.batch(len(lidar)).take(rep_32.shape[0]):
        yield [input_value]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
quantized_tflite_model = converter.convert()

# tflite_model_path = './Models/' + model_name + "_int8.tflite"
tflite_model_path = './Models/' + model_name + "_trackdata_int8.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(quantized_tflite_model)
    print(f"{model_name}_int8.tflite is saved.")

print('Tf_lite Models also saved')

#======================================================
# Evaluated TfLite Model
#======================================================

def evaluate_model(model_path, test_lidar, test_data):
    """Evaluate TfLite model."""
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_details = interpreter.get_output_details()

    output_lidar = test_lidar
    output_servo = []
    output_speed = []

    period = 1.0 / hz

    # Initialize a list to store inference times in microseconds
    inference_times_micros = []

    # Iterate through the lidar data
    for lidar_data in output_lidar:
        # Preprocess lidar data for inference
        lidar_data = np.expand_dims(lidar_data, axis=-1).astype(np.float32)
        lidar_data = np.expand_dims(lidar_data, axis=0)

        # Check for empty lidar data
        if lidar_data is None:
            continue

        # Measure inference time
        ts = time.time()
        interpreter.set_tensor(input_index, lidar_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        dur = time.time() - ts

        # Convert inference time to microseconds
        inference_time_micros = dur * 1e6
        inference_times_micros.append(inference_time_micros)

        # Print inference time information
        if dur > period:
            print("%.3f: took %.2f microseconds - deadline miss." % (dur, int(dur * 1000000)))

        # Extract servo and speed output from the model
        servo = output[0, 0]
        speed = output[0, 1]

        # Append output servo and speed
        output_servo.append(servo)
        output_speed.append(speed)

    output_lidar = np.asarray(output_lidar)
    output_servo = np.asarray(output_servo)
    output_speed = np.asarray(output_speed)
    assert len(output_lidar) == len(output_servo) == len(output_speed)
    output = np.concatenate((output_servo[:, np.newaxis], output_speed[:, np.newaxis]), axis=1)
    y_pred = output

    # Calculate average and maximum inference times in microseconds
    arr = np.array(inference_times_micros)
    perc99 = np.percentile(arr, 99)
    arr = arr[arr < perc99]
    average_inference_time_micros = np.mean(arr)
    max_inference_time_micros = np.max(arr)

    # Print inference time statistics
    print("Model: ", model_path)
    print("Average Inference Time: %.2f microseconds" % average_inference_time_micros)
    print("Maximum Inference Time: %.2f microseconds" % max_inference_time_micros)

    return y_pred, inference_times_micros

# Initialize empty lists to store results for each model
all_inference_times_micros = []
for model_name in model_files:
    y_pred, inference_times_micros = evaluate_model(model_name, test_lidar, test_data)
    all_inference_times_micros.append(inference_times_micros)
    
    print(f'Huber Loss for {model_name}: {huber_loss(test_data, y_pred)}\n')

# Plot inference times
plt.figure()
for inference_times_micros in all_inference_times_micros:
    arr = np.array(inference_times_micros)
    perc99 = np.percentile(arr, 99)
    arr = arr[arr < perc99]
    plt.plot(arr)
plt.xlabel('Inference Iteration')
plt.ylabel('Inference Time (microseconds)')
plt.title('Inference Time per Iteration')
plt.legend(model_files)

print('End')
