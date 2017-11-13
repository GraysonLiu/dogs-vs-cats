import numpy as np
import os

import yaml

from model_wrapper import ModelWrapper
import process_data

with open('path.yml') as data_path_file:
    data_path_yaml = yaml.load(data_path_file)
    train_set_raw_data_dir = data_path_yaml['train_set_raw_data_dir']
    test_set_raw_data_dir = data_path_yaml['test_set_raw_data_dir']
    processed_data_dir = data_path_yaml['processed_data_dir']
    tensorboard_dir = data_path_yaml['tensorboard_dir']
    model_dir = data_path_yaml['model_dir']
    learning_rate = float(data_path_yaml['learning_rate'])
    img_size = int(data_path_yaml['img_size'])

if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

train_set_processed_data_path = os.path.join(processed_data_dir, 'train_{}.npy'.format(img_size))
if os.path.exists(train_set_processed_data_path):
    train_set_data = np.load(train_set_processed_data_path)
    print('Data loaded!')
else:
    train_set_data = process_data.process_train_set_data(img_size, train_set_raw_data_dir,
                                                         train_set_processed_data_path)
    print('Data processed!')

model_wrapper = ModelWrapper(learning_rate, img_size, tensorboard_dir=tensorboard_dir)
model = model_wrapper.model

model_path = os.path.join(model_dir, model_wrapper.name)
if os.path.exists(model_path):
    model.load(os.path.join(model_path, model_wrapper.name))
    print('Model loaded!')

train_set_data, validation_set_data = train_set_data[:-500], train_set_data[-500:]

train_x = np.array([i[0] for i in train_set_data]).reshape(-1, img_size, img_size, 1)
train_y = [i[1] for i in train_set_data]

validation_x = np.array([i[0] for i in validation_set_data]).reshape(-1, img_size, img_size, 1)
validation_y = [i[1] for i in validation_set_data]

model.fit({'input': train_x}, {'targets': train_y}, n_epoch=5,
          validation_set=({'input': validation_x}, {'targets': validation_y}),
          snapshot_step=500, show_metric=True, run_id=model_wrapper.name)

model.save(os.path.join(model_dir, model_wrapper.name, model_wrapper.name))

# tensorboard --logdir=foo:A:\Code\Python\tensorflow\dogs-vs-cats\log
