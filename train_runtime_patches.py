import numpy as np
import os, sys, csv, cv2, time, random, keras, h5py
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.externals import joblib 
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, concatenate 
from keras.optimizers import Adam 


def _show_patch(real_patch,blackdotted_patch):
    f, ax = plt.subplots(1,2)
    ax1, ax2 = ax.flatten()
    ax1.imshow(real_patch)
    ax1.set_title('real')
    ax2.imshow(blackdotted_patch)
    ax2.set_title('blackdotted')
    plt.show()

def sample_centered_patch(train_set,patch_size):
    train_index = random.randrange(len(train_set))
    real_filename = 'input/cropped/{}'.format(train_set[train_index])
    blackdotted_filename = 'input/black_white_dots/{}'.format(train_set[train_index])
    
    real_image = cv2.imread(real_filename)
    blackdotted_image = cv2.imread(blackdotted_filename,0)
    n_points = np.sum(blackdotted_image > 245)  # jpeg is lossy, so white dots may be less than 255
    
    #print('Sampling from image {} with shape {}'.format(train_set[train_index],blackdotted_image.shape))
    
    # split between empty images and images with dots  
    if n_points>0:
        good_points = np.where(blackdotted_image > 245)
        r_good = good_points[0]
        c_good = good_points[1]
            
        # select a random dot to extract a patch roughly centered on that dot
        patch_index = random.randrange(len(r_good))
        
        r_center = r_good[patch_index]
        c_center = c_good[patch_index]
   
        # check bounds for row coord
        if r_center < patch_size-1:
            r_start = 0
            r_end = patch_size - 1
        elif (blackdotted_image.shape[0]-1)-r_center < patch_size:
            r_end = blackdotted_image.shape[0] - 1
            r_start = r_end - patch_size + 1 
        else:
            r_offset = random.randrange(patch_size - 1)
            r_start = r_center - r_offset
            r_end = r_start + patch_size - 1   
        # check bounds for col coord
        if c_center < patch_size-1:
            c_start = 0
            c_end = patch_size - 1
        elif (blackdotted_image.shape[1]-1)-c_center < patch_size:
            c_end = blackdotted_image.shape[1] - 1
            c_start = c_end - patch_size + 1
        else:
            c_offset = random.randrange(patch_size - 1)
            c_start = c_center - c_offset
            c_end = c_start + patch_size - 1

        real_patch = real_image[r_start:r_end+1,c_start:c_end+1]
        blackdotted_patch = blackdotted_image[r_start:r_end+1,c_start:c_end+1]
        label = np.sum(blackdotted_patch > 245)
    
    else:
        # select a random patch that won't contain any dot (becouse the image has no dots) 
        r_start = random.randrange(real_image.shape[0] - patch_size - 1)
        c_start = random.randrange(real_image.shape[1] - patch_size - 1)

        real_patch = real_image[r_start:r_start+patch_size,c_start:c_start+patch_size]
        blackdotted_patch = blackdotted_image[r_start:r_start+patch_size,c_start:c_start+patch_size]
        label = 0
              
    #print('Shape of Real Patch : {}'.format(real_patch.shape))
    #print('Shape of Blackdotted Patch : {}'.format(blackdotted_patch.shape))
    #print('Label : {}'.format(label))
    return (real_patch/255).astype('float32'), np.array(label).reshape(1,1,1), (train_index,r_start,c_start) 

def sample_random_patch(train_set,patch_size):
    train_index = random.randrange(len(train_set))
    real_filename = 'input/cropped/{}'.format(train_set[train_index])
    blackdotted_filename = 'input/black_white_dots/{}'.format(train_set[train_index])
    
    real_image = cv2.imread(real_filename)
    blackdotted_image = cv2.imread(blackdotted_filename,0)
    
    #print('Sampling from image {} with shape {}'.format(train_set[train_index],blackdotted_image.shape))
    
    # select a random patch 
    r_start = random.randrange(real_image.shape[0] - patch_size - 1)
    c_start = random.randrange(real_image.shape[1] - patch_size - 1)

    real_patch = real_image[r_start:r_start+patch_size,c_start:c_start+patch_size]
    blackdotted_patch = blackdotted_image[r_start:r_start+patch_size,c_start:c_start+patch_size]
    label = 0
              
    #print('Shape of Real Patch : {}'.format(real_patch.shape))
    #print('Shape of Blackdotted Patch : {}'.format(blackdotted_patch.shape))
    #print('Label : {}'.format(label))
    return (real_patch/255).astype('float32'), np.array(label).reshape(1,1,1), (train_index,r_start,c_start)

def get_batch(file_names,batch_size,patch_size):
    x = []
    y = []
    count = 0
    while count<batch_size:
        if count<batch_size*0.7:
            patch, label, coords = sample_centered_patch(file_names,patch_size)
        else:
            patch, label, coords = sample_random_patch(file_names,patch_size)    
        x.append(patch)
        y.append(label)
        count += 1 
    x=np.array(x)
    y=np.array(y)
    return x,y

def train_generator(file_names,batch_size,patch_size):
    print('\nTRAIN generator was initiated..')
    idf = 0
    while True:
        start_time = time.time()
        print('\nExtracting TRAIN Batch with id={}'.format(idf))
        x,y = get_batch(file_names,batch_size,patch_size)
        elapsed = time.time() - start_time
        #print('\nCounts in batch with id={} batch: {} +/- {}'.format(idf,y.mean(),y.std()))
        print('\nDone TRAIN batch with id={} in {} seconds...'.format(idf,elapsed))
        idf+=1
        yield (x,y)

def valid_generator(file_names,batch_size,patch_size):
    print('\nVALIDATION generator was initiated..')
    idf = 0
    while True:
        start_time = time.time()
        print('\nExtracting VALIDATION Batch with id={}'.format(idf))
        x,y = get_batch(file_names,batch_size,patch_size)
        elapsed = time.time() - start_time
        #print('\nCounts in batch with id={} batch: {} +/- {}'.format(idf,y.mean(),y.std()))
        print('\nDone VALIDATION batch with id={} in {} seconds...'.format(idf,elapsed))
        idf+=1
        yield (x,y)
        
def build_model_patch32():
    
    input_layer = Input(shape=(32,32,3))
    
    net = Conv2D(64, (3,3), padding='valid', activation='relu')(input_layer)
    net = BatchNormalization()(net)
    
    stack_1 = Conv2D(16, (3,3), padding='same', activation='relu')(net)
    stack_2 = Conv2D(16, (1,1), padding='same', activation='relu')(net)
    net = concatenate([stack_1,stack_2])
    net = BatchNormalization()(net)

    stack_3 = Conv2D(32, (3,3), padding='same', activation='relu')(net)
    stack_4 = Conv2D(16, (1,1), padding='same', activation='relu')(net)
    net = concatenate([stack_3,stack_4])
    net = BatchNormalization()(net)  

    net = Conv2D(16, (14,14), padding='valid', activation='relu')(net)
    net = BatchNormalization()(net)

    stack_5 = Conv2D(48, (3,3), padding='same', activation='relu')(net)
    stack_6 = Conv2D(112, (1,1), padding='same', activation='relu')(net)
    net = concatenate([stack_5,stack_6])
    net = BatchNormalization()(net)  

    stack_7 = Conv2D(40, (3,3), padding='same', activation='relu')(net)
    stack_8 = Conv2D(40, (1,1), padding='same', activation='relu')(net)
    net = concatenate([stack_7,stack_8])
    net = BatchNormalization()(net) 

    stack_9 = Conv2D(96, (3,3), padding='same', activation='relu')(net)
    stack_10 = Conv2D(32, (1,1), padding='same', activation='relu')(net)
    net = concatenate([stack_9,stack_10])
    net = BatchNormalization()(net) 

    net = Conv2D(64, (17,17), padding='valid', activation='relu')(net)
    net = BatchNormalization()(net)     

    net = Conv2D(64, (1,1), padding='valid', activation='relu')(net)
    net = BatchNormalization()(net)  
    
    net = Conv2D(1, (1,1), padding='valid', activation='relu')(net)
    net = BatchNormalization()(net)
    
    model = Model(inputs=input_layer, outputs=net)
    
    #model.summary()
    
    #print('*'*30)
    #for l in model.layers:
    #    print('-'*30)
    #    print(l.input_shape)
    #    print(l.output_shape)
    #    print('-'*30)
    #print('*'*30)
    
    optimizer = Adam(lr=0.005)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model    
    
    
if __name__ == "__main__":
    start_time = time.time()  
    
    
    file_names = os.listdir('input/cropped')
    file_names = sorted(file_names, key=lambda 
                        item: int(item.partition('.')[0]) if item[0].isdigit() else float('inf'))
    
    # TODO
    # At the moment: ugly train/test split
    train_size = int(len(file_names)*0.6)
    validation_size = int((len(file_names) - train_size)/2)
    test_size = len(file_names) - train_size - validation_size
    
    train_file_names = file_names[:train_size]
    validation_file_names = file_names[train_size:train_size+validation_size]
    test_file_names = file_names[train_size+validation_size:]
    
    batch_size = 80
    patch_size = 32
    
    modelid = time.strftime('%Y%m%d%H%M%S')

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50),
        keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/model_checkpoint_best_{}.h5'.format(modelid),
            monitor='val_loss',
            save_best_only=True),
        keras.callbacks.TensorBoard(
            log_dir='./logs/{}'.format(modelid),
            histogram_freq=0, write_graph=False, write_images=False)
    ]

    model = build_model_patch32()
    history=model.fit_generator(
                        generator=train_generator(train_file_names, batch_size, patch_size),
                        use_multiprocessing=False,
                        workers=1,
                        steps_per_epoch=50, 
                        epochs=1000, 
                        verbose=1,
                        validation_data=valid_generator(validation_file_names, batch_size, patch_size),
                        validation_steps=5,
                        callbacks=callbacks_list)
        
    elapsed = (time.time() - start_time)/60
    print()
    print('Total Time: {}'.format(elapsed)) 
    
    
    
    
