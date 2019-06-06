import os

import random
import shutil

ROOT_DIR = "/home/ubuntu/data/processed_tuSimple_dataset"
ORIGINAL_ROOT_DIR = ROOT_DIR + "/original_image/"
ORIGINAL_TRAIN_DIR = ROOT_DIR + "/original_train/"
ORIGINAL_DEV_DIR = ROOT_DIR + "/original_dev/"
ORIGINAL_TEST_DIR = ROOT_DIR + "/original_test/"
LABEL_ROOT_DIR = ROOT_DIR + "/label_image/"
LABEL_TRAIN_DIR = ROOT_DIR + "/label_train/"
LABEL_DEV_DIR = ROOT_DIR + "/label_dev/"
LABEL_TEST_DIR = ROOT_DIR + "/label_test/"

def loop_directory_and_find_images(directory):
    test_data = []
    rest = []
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            loop_directory(os.path.join(directory, filename))
        elif filename.endswith(".png"):
            if (filename.startswith("clips_0313-1")):
                no_suffix = filename.split('.')[0]
                frame = no_suffix.split('_')[2]
                if (int(frame) / 60 % 10 == 0):
                    test_data.append(filename)
#                     print filename
                else:
                    rest.append(filename)
            elif (filename.startswith("clips_0313-2")):
                no_suffix = filename.split('.')[0]
                frame = no_suffix.split('_')[2]
                if (int(frame) <= 1800):
                    if (int(frame) / 5 % 10 == 0):
#                         print filename
                        test_data.append(filename)
                    else:
                        rest.append(filename)
                else:
                    if (int(frame) / 60 % 10 == 0):
#                         print filename
                        test_data.append(filename)
                    else:
                        rest.append(filename)

    
    return test_data, rest

def loop_directory_and_copy_images(directory, new_directory, data_set):
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            loop_directory(os.path.join(directory, filename))
        elif filename in data_set:
            shutil.copy2(os.path.join(directory, filename), new_directory)
            print("Copied file ", filename, " to new directory ", new_directory)
            
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Directory " , directory ,  " Created ")
    else:    
        print("Directory " , directory ,  " already exists")
        
if __name__ == '__main__':
    test_data, rest = loop_directory_and_find_images(ORIGINAL_ROOT_DIR)
    print (len(test_data))
    print (len(rest))

    SEED = 1024
    random.seed(SEED)
    random.shuffle(rest)
    dev_data = rest[:281]
    train_data = rest[281:]
    
    create_directory_if_not_exists(ORIGINAL_TRAIN_DIR)
    create_directory_if_not_exists(ORIGINAL_DEV_DIR)
    create_directory_if_not_exists(ORIGINAL_TEST_DIR)
    
    loop_directory_and_copy_images(ORIGINAL_ROOT_DIR, ORIGINAL_TRAIN_DIR, train_data)
    loop_directory_and_copy_images(ORIGINAL_ROOT_DIR, ORIGINAL_DEV_DIR, dev_data)
    loop_directory_and_copy_images(ORIGINAL_ROOT_DIR, ORIGINAL_TEST_DIR, test_data)
    
    create_directory_if_not_exists(LABEL_TRAIN_DIR)
    create_directory_if_not_exists(LABEL_DEV_DIR)
    create_directory_if_not_exists(LABEL_TEST_DIR)
    
    loop_directory_and_copy_images(LABEL_ROOT_DIR, LABEL_TRAIN_DIR, train_data)
    loop_directory_and_copy_images(LABEL_ROOT_DIR, LABEL_DEV_DIR, dev_data)
    loop_directory_and_copy_images(LABEL_ROOT_DIR, LABEL_TEST_DIR, test_data)
    
    
