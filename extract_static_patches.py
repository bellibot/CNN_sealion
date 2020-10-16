import numpy as np
import os, sys, csv, cv2, time, random
import matplotlib.pyplot as plt
from sklearn.externals import joblib 
from skimage.util import view_as_blocks, view_as_windows

def _show_patch(real_patch,blackdotted_patch):
    f, ax = plt.subplots(1,2)
    ax1, ax2 = ax.flatten()
    ax1.imshow(real_patch)
    ax1.set_title('real')
    ax2.imshow(blackdotted_patch)
    ax2.set_title('blackdotted')
    plt.show()

    
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
    
    mode = 'validation' #'train' #'validation'
    
    if mode=='train':
        file_names = train_file_names
    elif mode=='validation':
        file_names = validation_file_names
    else:
        print('Wrong Mode...')
        sys.exit()     
        
    patch_size = 32
    total_counts_dict = {}
    for filename in file_names:
        image_time = time.time()
            
        clean_filename = 'input/cropped/{}'.format(filename)
        blackdotted_filename = 'input/black_white_dots/{}'.format(filename)
               
        clean_image = cv2.imread(clean_filename)  
        blackdotted_image = cv2.imread(blackdotted_filename,0)             
        blackdotted_image = cv2.threshold(blackdotted_image, 245, 255, cv2.THRESH_BINARY)[1]
        blackdotted_image[np.where(blackdotted_image < 255)] = 0 
        
        #print(clean_image.shape)
        #print(blackdotted_image.shape)
        #print(clean_image.dtype)
        #print(blackdotted_image.dtype)
        #print()
        
        BLOCKS = True
        if BLOCKS:
            clean_patches = view_as_blocks(clean_image,(patch_size,patch_size,3))
            blackdotted_patches = view_as_blocks(blackdotted_image,(patch_size,patch_size))
            #print(len(clean_patches))
            #print(len(blackdotted_patches))
        else:    
            clean_patches = view_as_windows(clean_image,(patch_size,patch_size,3))
            blackdotted_patches = view_as_windows(blackdotted_image,(patch_size,patch_size))
            #print(len(clean_patches))
            #print(len(blackdotted_patches))
        
        clean_patches = clean_patches.reshape((clean_patches.shape[0],clean_patches.shape[1],patch_size,patch_size,3))
        
        n_meaningful_patches = 0  
        #extract all meaningful patches   
        for r in range(clean_patches.shape[0]):
            for c in range(clean_patches.shape[1]):
                n_dots_in_patch = np.sum(blackdotted_patches[r,c,:,:] == 255)
                if  n_dots_in_patch > 0:
                    patchname = filename.partition('.')[0]+'_'+str(r)+'_'+str(c)+'.jpg'
                    cv2.imwrite('input/patches32_{}/full/{}'.format(mode,patchname),clean_patches[r,c,:,:,:])
                    cv2.imwrite('input/patches32_{}/blackdotted/{}'.format(mode,patchname),blackdotted_patches[r,c,:,:]) 
                    total_counts_dict[patchname] = n_dots_in_patch
                    n_meaningful_patches += 1
               
        #extract an equal number of empty patches
        for r in range(clean_patches.shape[0]):
            for c in range(clean_patches.shape[1]):
                if n_meaningful_patches>0:            
                    n_dots_in_patch = np.sum(blackdotted_patches[r,c,:,:] == 255)
                    rnd = random.random()
                    if rnd>=0.25:
                        accept_flag = False
                    else:
                        accept_flag = True 
                    if n_dots_in_patch==0 and accept_flag:    
                        patchname = filename.partition('.')[0]+'_'+str(r)+'_'+str(c)+'.jpg'
                        cv2.imwrite('input/patches32_{}/empty/{}'.format(mode,patchname),clean_patches[r,c,:,:,:])
                        total_counts_dict[patchname] = n_dots_in_patch
                        n_meaningful_patches -= 1 
        
        elapsed = (time.time() - image_time)
        print('Image {} Processed in {} seconds'.format(filename,elapsed))
        print()
        
    joblib.dump(total_counts_dict,'input/patches32_{}_total_counts.pkl'.format(mode))   
    elapsed = (time.time() - start_time)/60
    print()
    print('Total Time: {}'.format(elapsed)) 
    
    
    
    
