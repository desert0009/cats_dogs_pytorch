'''
Please manually download dataset from Kaggle:
    https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data#_=_
This script will copy all cat images(cat.<id>.jpg) to cats folder and 
all dog images(dog.<id>.jpg) to dogs folder. We will also set apart val set.

This is train/val/test directory structure:
    train/
        dogs/
            dog.0.jpg
        cats/
            cat.0.jpg
    val/
        dogs/
            dog.1.jpg
        cats/
            cat.0.jpg
    test/
        0.jpg (is dog)
        1.jpg (is cat)
'''
from argparse import ArgumentParser
import os
import numpy as np
from shutil import copyfile

def dog_cat_list(input_path):
    listOfFiles = os.listdir(input_path)
    dogs, cats= [], []
    for file in listOfFiles:
        if file.startswith('dog'):
            dogs.append(file)
        else:
            cats.append(file)
    return dogs, cats

def setup_train_val(dogs, cats):
    val_dogs = set(np.random.choice(dogs, int(len(dogs)*0.04), replace=False))
    val_cats = set(np.random.choice(cats, int(len(cats)*0.04), replace=False))
    train_dogs = set(dogs) - val_dogs
    train_cats = set(cats) - val_cats
    return train_dogs, train_cats, val_dogs, val_cats

def copy_fils(from_path, to_path, train_dogs, train_cats, val_dogs, val_cats):
    for file in train_dogs:
        copyfile(os.path.join(from_path, file), os.path.join(to_path, 'train/dogs', file))
    for file in train_cats:
        copyfile(os.path.join(from_path, file), os.path.join(to_path, 'train/cats', file))
    for file in val_dogs:
        copyfile(os.path.join(from_path, file), os.path.join(to_path, 'val/dogs', file))
    for file in val_cats:
        copyfile(os.path.join(from_path, file), os.path.join(to_path, 'val/cats', file))
    
# python setup_train.py --from_path '../../coursera/dataset/train' --to_path '../data/dogs_cats'
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--from_path', type=str, default='', \
                        help='The path of mixed training images')
    parser.add_argument('--to_path', type=str, default='')
    args = parser.parse_args()
    
    print('Start copy training images from {} to {}'.format(\
          args.from_path, args.to_path))
    
    dogs, cats = dog_cat_list(args.from_path)
    train_dogs, train_cats, val_dogs, val_cats = setup_train_val(dogs, cats)
    print('Size of train_dogs:{}, Size of train_cats:{}, Size of val_dogs:{},\
           Size of val_cats:{}'.format(len(train_dogs), len(train_cats), \
          len(val_dogs), len(val_cats)))
    
    copy_fils(args.from_path, args.to_path, train_dogs, train_cats, val_dogs, val_cats)
    print('Done.')