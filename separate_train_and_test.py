from glob import glob
import shutil
import os
import random

dataset = "dataset"
test = "test"
train = "train"

os.chdir(dataset)

for test_file in random.choices(glob("*.jpg"), k=100):
    shutil.move(os.path.join(test_file),
                os.path.join(test, test_file))

for train_file in glob("*.jpg"):
    shutil.move(os.path.join(train_file),
                os.path.join(train, train_file))
