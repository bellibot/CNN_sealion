import numpy as np
import os, sys, csv, cv2, time, random
import matplotlib.pyplot as plt
from sklearn.externals import joblib 
from skimage.util import view_as_windows
from keras.models import load_model

def _show_countmaps(pred,true,diff):
    f, ax = plt.subplots(1,3)
    ax1, ax2, ax3 = ax.flatten()
    ax1.imshow(pred)
    ax1.set_title('pred')
    ax2.imshow(true)
    ax2.set_title('true')
    ax3.imshow(diff)
    ax3.set_title('diff')
    plt.show()       
  
def sample_centered_patch(clean_image,blackdotted_image,patch_size):
    n_points = np.sum(blackdotted_image == 255)
    if n_points>0:
        good_points = np.where(blackdotted_image == 255)
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

        clean_patch = clean_image[r_start:r_end+1,c_start:c_end+1]
        blackdotted_patch = blackdotted_image[r_start:r_end+1,c_start:c_end+1]
    else:
        clean_patch = clean_patch[0,patch_size,:]
        blackdotted_patch = blackdotted_patch[0,patch_size]  
    return clean_patch, blackdotted_patch   
        
if __name__ == "__main__":
    start_time = time.time()  
    
    file_names = os.listdir('input/cropped')
    file_names = sorted(file_names, key=lambda 
                        item: int(item.partition('.')[0]) if item[0].isdigit() else float('inf'))
    
    train_size = int(len(file_names)*0.6)
    validation_size = int((len(file_names) - train_size)/2)
    test_size = len(file_names) - train_size - validation_size
    
    train_file_names = file_names[:train_size]
    validation_file_names = file_names[train_size:train_size+validation_size]
    test_file_names = file_names[train_size+validation_size:]
    
    #print(len(train_file_names))
    #print(len(validation_file_names))
    #print(len(test_file_names))
    #sys.exit()
    
    file_names = test_file_names
    
    model = load_model('checkpoints/new.h5') #32adam0.0001
    patch_size = int(model.inputs[0].shape[1]) 
    true_counts = []
    pred_counts = []
    for filename in file_names:
        print('Processing {}'.format(filename))
        image_time = time.time()
            
        clean_filename = 'input/cropped/{}'.format(filename)
        blackdotted_filename = 'input/black_white_dots/{}'.format(filename)
               
        clean_image = cv2.imread(clean_filename)      
        blackdotted_image = cv2.imread(blackdotted_filename,0)            
        blackdotted_image = cv2.threshold(blackdotted_image, 245, 255, cv2.THRESH_BINARY)[1]
        blackdotted_image[np.where(blackdotted_image < 255)] = 0      
        
        clean_image, blackdotted_image = sample_centered_patch(clean_image,blackdotted_image,216)
   
        total_counts = np.sum(blackdotted_image == 255)
        
        print('PRE PADDING CLEAN {}'.format(clean_image.shape))   
        print('PRE PADDING BLACKDOTTED {}'.format(blackdotted_image.shape)) 
        print()        
                      
        clean_image = np.pad(clean_image,((patch_size-1,patch_size-1),(patch_size-1,patch_size-1),(0,0)),'constant',constant_values=(0,0))
        clean_image = clean_image/255                    
        blackdotted_image = np.pad(blackdotted_image,((patch_size-1,patch_size-1),(patch_size-1,patch_size-1)),'constant',constant_values=(0,0))
        blackdotted_image = blackdotted_image/255

        print('POST PADDING CLEAN {}'.format(clean_image.shape)) 
        print('POST PADDING BLACKDOTTED {}'.format(blackdotted_image.shape))
        print() 
        
        clean_patches = view_as_windows(clean_image,(patch_size,patch_size,3))
        clean_patches = clean_patches.reshape((clean_patches.shape[0],clean_patches.shape[1],patch_size,patch_size,3))
        blackdotted_patches = view_as_windows(blackdotted_image,(patch_size,patch_size))
           
        print('NUMBER OF CLEAN PATCHES {}'.format(clean_patches.shape))
        print('NUMBER OF BLACKDOTTED PATCHES {}'.format(blackdotted_patches.shape))
        print()
      
        # with many windows the gpu goes out of memory
        #predictions = []
        #for i in range(clean_patches.shape[0]):
            #preds = model.predict(clean_patches[i,:,:,:,:])
            #predictions.extend(list(preds.reshape(-1))) 
            #print('DONE I={}'.format(i))      
         
        clean_patches = clean_patches.reshape(clean_patches.shape[0]*clean_patches.shape[1],clean_patches.shape[2],clean_patches.shape[3],clean_patches.shape[4])
        pred_countmap = model.predict(clean_patches).reshape((blackdotted_patches.shape[0],blackdotted_patches.shape[1]))
        pred_countmap = pred_countmap/(patch_size*patch_size)   
        
        true_countmap = np.zeros((blackdotted_patches.shape[0],blackdotted_patches.shape[1])) 
        for r in range(blackdotted_patches.shape[0]):
            for c in range(blackdotted_patches.shape[1]):
                true_countmap[r,c] = np.sum(blackdotted_patches[r,c,:,:])             
        true_countmap = true_countmap/(patch_size*patch_size)
        
        print('PRED COUNTMAP SHAPE {}'.format(pred_countmap.shape))
        print('TRUE COUNTMAP SHAPE {}'.format(true_countmap.shape))
        print()
        
        true_count = np.sum(true_countmap)
        pred_count = np.sum(pred_countmap) 
                      
        print('TOTAL COUNTS IN IMAGE {}'.format(total_counts))
        print('TRUE_COUNTMAP: TOTAL {} | MAX {} | MIN {}'.format(true_count,true_countmap.max(),true_countmap.min()))
        print('PRED_COUNTMAP: TOTAL {} | MAX {} | MIN {}'.format(pred_count,pred_countmap.max(),pred_countmap.min()))
        
        #absdiff = np.abs(pred_countmap - true_countmap)
        #_show_countmaps(pred_countmap,true_countmap,absdiff)
        
        true_counts.append(true_count)
        pred_counts.append(pred_count)
        
        elapsed = (time.time() - image_time)/60
        print('Image {} Processed in {} Minutes'.format(filename,elapsed))
        print()
        
    joblib.dump((true_counts,pred_counts),'input/test_true_pred_counts.pkl')     
    elapsed = (time.time() - start_time)/60
    print()
    print('Total Time: {}'.format(elapsed))
    
    
    
    
