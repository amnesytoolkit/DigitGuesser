import os, shutil

os.chdir("./data")
for dir in os.listdir():
    os.chdir(dir)
    for file in os.listdir():
        path = os.getcwd()  + "/"
        shutil.move(path + file, path + "moved_" + file)
    os.chdir("../")