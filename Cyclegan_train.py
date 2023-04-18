import os
import matplotlib.pyplot as plt
import numpy as np

from models.cycleGAN import CycleGAN
from utils.loaders import DataLoader

import cv2
from glob import glob
########################################*PARAMS*########################################################################
# run params
SECTION = 'paint'
RUN_ID = '0001'
DATA_NAME = 'cezanne2photo'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])


if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
    os.mkdir(os.path.join(RUN_FOLDER, 'translated_imgs'))


#mode =  'build' # 'build' #
mode = 'test' # 'test' #

IMAGE_SIZE = 128


BATCH_SIZE = 1
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 10

TEST_A_FILE = '00080.jpg'
TEST_B_FILE = "2014-08-05 03:00:51.jpg"

TEST_TYPE = 'testB'
########################################################################################################################
# Data
data_loader = DataLoader(dataset_name=DATA_NAME, img_res=(IMAGE_SIZE, IMAGE_SIZE))

# Architecture
gan = CycleGAN(
    input_dim = (IMAGE_SIZE,IMAGE_SIZE,3)
    ,learning_rate = 0.0002
    , buffer_max_length = 50
    , lambda_validation = 1
    , lambda_reconstr = 10
    , lambda_id = 2
    , generator_type = 'resnet'
    , gen_n_filters = 32
    , disc_n_filters = 32
    )
if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

gan.g_BA.summary()

gan.g_AB.summary()

gan.d_A.summary()

gan.d_B.summary()

# Plot models
gan.plot_model(RUN_FOLDER)

# Train
if mode != 'test':
    gan.train(data_loader
            , run_folder = RUN_FOLDER
            , epochs=EPOCHS
            , test_A_file = TEST_A_FILE
            , test_B_file = TEST_B_FILE
            , batch_size=BATCH_SIZE
            , sample_interval=PRINT_EVERY_N_BATCHES)

# Loss
fig = plt.figure(figsize=(20,10))

plt.plot([x[1] for x in gan.g_losses], color='green', linewidth=0.1) #DISCRIM LOSS
# plt.plot([x[2] for x in gan.g_losses], color='orange', linewidth=0.1)
plt.plot([x[3] for x in gan.g_losses], color='blue', linewidth=0.1) #CYCLE LOSS
# plt.plot([x[4] for x in gan.g_losses], color='orange', linewidth=0.25)
plt.plot([x[5] for x in gan.g_losses], color='red', linewidth=0.25) #ID LOSS
# plt.plot([x[6] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.plot([x[0] for x in gan.g_losses], color='black', linewidth=0.25)

# plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.ylim(0, 5)

plt.show()



# Test image read
im_path = glob('./data/%s/%s/%s' % (data_loader.dataset_name, TEST_TYPE, TEST_B_FILE))

im_path = ' '.join(map(str, im_path))

base_image = cv2.imread(im_path)

trans_impath = os.path.join(RUN_FOLDER, 'translated_imgs', 'res_trans_img.png')

translated_image = data_loader.write_img(gan.g_BA.predict(data_loader.load_img(im_path)), trans_impath)

translated_image = cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB)

cv2.imshow("Base image", base_image)
cv2.imshow("Translated image", translated_image)

cv2.waitKey(0)