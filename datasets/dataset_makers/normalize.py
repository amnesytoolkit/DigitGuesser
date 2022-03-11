import csv, os
import shutil

# dataset maker for https://www.kaggle.com/dhruvildave/english-handwritten-characters-dataset

os.chdir("../data/")
with open('english.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[0] == 'image':
            continue
        # if directory does not exist, create it
        if not os.path.exists(row[1]):
            os.makedirs(row[1])
        shutil.move("./" + row[0], "./" + row[1])

os.remove('english.csv')
shutil.rmtree('./Img')