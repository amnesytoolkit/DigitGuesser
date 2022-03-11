import os, shutil
from keras_preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15)
super_dir = os.getcwd()

os.chdir("./datasets")
datasets_dir = os.getcwd()
os.chdir("./data")
data_dir = os.getcwd()

moved_files = []
images_per_class = 15

# move all dirs in dataset
for file in os.listdir():
        shutil.move(file, "../")
        moved_files.append(file)

if not os.path.exists(super_dir + '/datasets/augmented_data/'):
        os.mkdir(super_dir + '/datasets/augmented_data/')

for file in moved_files:
        shutil.move(datasets_dir + "/" + file, "./")
        i = 0
        if not os.path.exists(super_dir + '/datasets/augmented_data/' + file):
                os.mkdir(super_dir + '/datasets/augmented_data/' + file)
        for batch in datagen.flow_from_directory(super_dir + '/datasets/data', batch_size=128, target_size=(28, 28),
                                                 save_to_dir=super_dir + '/datasets/augmented_data/' + file, save_format='png'):
                i += 1
                if i > images_per_class:
                        break
        shutil.move(file, "../")

for file in moved_files:
        shutil.move(datasets_dir + "/" + file, "./data")
