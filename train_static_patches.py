import numpy as np
import os, sys, csv, cv2, time, random, keras, h5py
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.externals import joblib 
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, concatenate
from keras.optimizers import Adam, RMSprop 


def _show_patch(real_patch,blackdotted_patch):
    f, ax = plt.subplots(1,2)
    ax1, ax2 = ax.flatten()
    ax1.imshow(real_patch)
    ax1.set_title('real')
    ax2.imshow(blackdotted_patch)
    ax2.set_title('blackdotted')
    plt.show()

def sample_patch(file_names,initial_path,labels_dict,patch_type):
    rnd_index = random.randrange(len(file_names))
    filename = file_names[rnd_index]
    path = initial_path+'/'+patch_type+'/'+filename
    patch = cv2.imread(path)
    label = labels_dict[filename]
    #print('Selected patch {} of type {} with shape {}'.format(filename,patch_type,patch.shape))
    return (patch/255).astype('float32'), np.array(label).reshape(1,1,1)
    
def get_batch(file_names_empty,file_names_full,initial_path,labels_dict,batch_size):
    x = []
    y = []
    count = 0
    while count<batch_size:
        if count<batch_size*0.5:
            patch, label = sample_patch(file_names_full,initial_path,labels_dict,'full')
        else:
            patch, label = sample_patch(file_names_empty,initial_path,labels_dict,'empty')    
        x.append(patch)
        y.append(label)
        count += 1 
    x=np.array(x)
    y=np.array(y)
    return x,y
    
def my_generator(file_names_empty,file_names_full,labels_dict,batch_size,initial_path):
    print('\nA generator has been allocated!')
    while True:
        #start_time = time.time()
        #print('\nExtracting Batch...')
        x,y = get_batch(file_names_empty,file_names_full,initial_path,labels_dict,batch_size)
        #elapsed = time.time() - start_time
        #print('\nCounts in batch: {} +/- {}'.format(y.mean(),y.std()))
        #print('\nDone in {} seconds...'.format(elapsed))
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
    
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=[])
    return model    

    
if __name__ == "__main__":
    start_time = time.time()  
    
    train_file_names_full = os.listdir('input/patches32_train/full')
    train_file_names_empty = os.listdir('input/patches32_train/empty')
    validation_file_names_full = os.listdir('input/patches32_validation/full')
    validation_file_names_empty = os.listdir('input/patches32_validation/empty')
    
    #print(len(train_file_names_empty))
    #print(len(train_file_names_full))
    #print(len(validation_file_names_empty))
    #print(len(validation_file_names_full))
    #print(validation_file_names_full[0])
    
    train_labels_dict = joblib.load('input/patches32_train_total_counts.pkl')
    validation_labels_dict = joblib.load('input/patches32_validation_total_counts.pkl')
      
    #print(len(train_labels_dict))
    #print(len(validation_labels_dict))
    
    batch_size = 32
    
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
                        generator=my_generator(train_file_names_empty, train_file_names_full, train_labels_dict, batch_size,'input/patches32_train/'),
                        use_multiprocessing=False,
                        workers=1,
                        steps_per_epoch=500, 
                        epochs=1000, 
                        verbose=1,
                        validation_data=my_generator(validation_file_names_empty, validation_file_names_full, validation_labels_dict, batch_size,'input/patches32_validation/'),
                        validation_steps=500,
                        callbacks=callbacks_list)
        
    elapsed = (time.time() - start_time)/60
    print()
    print('Total Time: {}'.format(elapsed)) 
    
    
    
    
