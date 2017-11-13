import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
from tqdm import tqdm

from model_wrapper import ModelWrapper

import process_data

with open('path.yml') as data_path_file:
    data_path_yaml = yaml.load(data_path_file)
    # train_set_raw_data_dir = data_path_yaml['train_set_raw_data_dir']
    test_set_raw_data_dir = data_path_yaml['test_set_raw_data_dir']
    processed_data_dir = data_path_yaml['processed_data_dir']
    # tensorboard_dir = data_path_yaml['tensorboard_dir']
    model_dir = data_path_yaml['model_dir']
    # learning_rate = float(data_path_yaml['learning_rate'])
    # img_size = int(data_path_yaml['img_size'])

model_list = os.listdir(model_dir)
print('\nFind model:')

for num, model in enumerate(model_list):
    print(num, model)

model_name = model_list[int(input('\nChoose a model (enter a number): '))]
print('Load Model {}'.format(model_name))

model_path = os.path.join(model_dir, model_name)

learning_rate = float(model_name.split('-')[2])
img_size = int(model_name.split('-')[3])

model = ModelWrapper(learning_rate, img_size).model
model.load(os.path.join(model_path, model_name))
print('Model loaded!')

test_set_processed_data_path = os.path.join(processed_data_dir, 'test_{}.npy'.format(img_size))
if os.path.exists(test_set_processed_data_path):
    test_set_data = np.load(test_set_processed_data_path)
    print('Data loaded!')
else:
    test_set_data = process_data.process_test_set_data(img_size, test_set_raw_data_dir,
                                                       test_set_processed_data_path)
    print('Data processed!')

fig = plt.figure()

for num, data in enumerate(test_set_data[:12]):
    # cat: [1,0]
    # dog: [0,1]

    img_data = data[0]
    img_num = data[1]

    y = fig.add_subplot(3, 4, num + 1)

    model_out = model.predict([img_data.reshape(img_size, img_size, 1)])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(img_data, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

with open('submission-file.csv', 'w') as f:
    f.write('id,label\n')

with open('submission-file.csv', 'a') as f:
    for data in tqdm(test_set_data):
        img_data = data[0]
        img_num = data[1]
        model_out = model.predict([img_data.reshape(img_size, img_size, 1)])[0]
        f.write('{},{}\n'.format(img_num, model_out[1]))
