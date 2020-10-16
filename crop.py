import numpy as np
import os, sys, cv2, time
import skimage.feature

if __name__ == "__main__":
    start_time = time.time()    
    file_names = os.listdir('input/Train')
    file_names = sorted(file_names, key=lambda 
                        item: int(item.partition('.')[0]) if item[0].isdigit() else float('inf')) 

    r_list = []
    c_list = []
    for filename in file_names:
        image = cv2.imread('input/Train/' + filename)
        r_list.append(image.shape[0])
        c_list.append(image.shape[1])

    min_r = min(r_list)
    min_c = min(c_list)
    print(min_r)
    print(min_c)

    for filename in file_names:
        
        # read the Train and Train Dotted images
        image_dot = cv2.imread('input/TrainDotted/{}'.format(filename))
        image_nodot = cv2.imread('input/Train/{}'.format(filename))
        
        r_diff = image_dot.shape[0] - min_r
        c_diff = image_dot.shape[1] - min_c
        
        if r_diff % 2 == 0:
            r_begin = int(r_diff/2)
            r_end = int(r_diff/2)
        else:
            r_begin = int((r_diff-1)/2)
            r_end = int((r_diff-1)/2 +1)    

        if c_diff % 2 == 0:
            c_begin = int(c_diff/2)
            c_end = int(c_diff/2)
        else:
            c_begin = int((c_diff-1)/2)
            c_end = int((c_diff-1)/2 +1)     
        
        cv2.imwrite('input/cropped_dotted/{}'.format(filename), image_dot[r_begin:image_dot.shape[0]-r_end,c_begin:image_dot.shape[1]-c_end])
        cv2.imwrite('input/cropped/{}'.format(filename), image_nodot[r_begin:image_nodot.shape[0]-r_end,c_begin:image_nodot.shape[1]-c_end])
        

    elapsed = (time.time() - start_time)/60
    print()
    print('Total Time: {}'.format(elapsed)) 

