import numpy as np
import os, sys, csv, cv2, time
import skimage.feature
from sklearn.externals import joblib 


if __name__ == "__main__":
    start_time = time.time()    
    file_names = os.listdir('input/cropped')
    file_names = sorted(file_names, key=lambda 
                        item: int(item.partition('.')[0]) if item[0].isdigit() else float('inf')) 
    
    classes = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']
    errors_dict = {filename:0 for filename in file_names}
    total_counts_dict = {}
    class_counts_dict = {}

    for filename in file_names:    
        print('Filename: {}'.format(filename))
        image_time = time.time()
        
        # read the clean and dotted images
        image_1 = cv2.imread('input/cropped_dotted/{}'.format(filename))
        image_2 = cv2.imread('input/cropped/{}'.format(filename))
        
        # absolute difference between the two
        image_3 = cv2.absdiff(image_1,image_2)
        
        # mask out blackened regions from dotted
        mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        mask_1[mask_1 < 20] = 0
        mask_1[mask_1 > 0] = 255
       
        mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        mask_2[mask_2 < 20] = 0
        mask_2[mask_2 > 0] = 255
        
        image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
        image_5 = cv2.bitwise_or(image_4, image_4, mask=mask_2) 
        
        # convert to grayscale to be accepted by skimage.feature.blob_log
        image_6 = cv2.cvtColor(image_5, cv2.COLOR_BGR2GRAY)
        
        # detect blobs
        blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
        print('Found {} blobs'.format(len(blobs)))
         
        # output black image with a white pixel in the center of each blob
        output = np.zeros_like(image_6)
        classes_dict = {key:0 for key in classes}
        for blob in blobs:
            # decision tree to pick the class of the blob by looking at the color in the dotted image 
            y, x, s = blob
            b,g,r = image_1[int(y)][int(x)][:]                   
            BAD_BLOB = False
                   
            if r > 200 and b < 50 and g < 50: # RED
                classes_dict['adult_males'] += 1         
            elif r > 200 and b > 200 and g < 50: # MAGENTA
                classes_dict['subadult_males'] += 1
            elif r < 100 and b < 100 and 150 < g < 200: # GREEN
                classes_dict['pups'] += 1
            elif r < 100 and  100 < b and g < 100: # BLUE
                classes_dict['juveniles'] += 1 
            elif r < 150 and b < 50 and g < 100:  # BROWN
                classes_dict['adult_females'] += 1
            else:
                BAD_BLOB = True
                errors_dict[filename] += 1     
            if not BAD_BLOB:
                output[int(y)][int(x)] = 255      
                       
        total_counts_dict[filename] = np.sum(output == 255)
        class_counts_dict[filename] = classes_dict
        
        #print(class_counts_dict[filename])
        #print(total_counts_dict[filename])
        print('Number of bad blobs: {}'.format(errors_dict[filename]))
        elapsed = (time.time() - image_time)
        print('Processed in: {} seconds'.format(elapsed))
        print()
        
        cv2.imwrite('input/black_white_dots/{}'.format(filename), output)
        
    joblib.dump(class_counts_dict,'input/whole_total_counts.pkl')

    elapsed = (time.time() - start_time)/60
    print()
    print('Total Time: {}'.format(elapsed)) 




