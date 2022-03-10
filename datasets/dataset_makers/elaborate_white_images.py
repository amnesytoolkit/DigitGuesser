import os, cv2

os.chdir("../data")
for dir in os.listdir():
    os.chdir(dir)
    for file in os.listdir():
        path = "/home/amnesy/Desktop/sharedWAll/Scuola/" \
               "stage-2021/DigitGuesser/datasets/data/" + \
               dir + "/" + file
        data = cv2.imread(path)
        grey = cv2.cvtColor(data.copy(), cv2.COLOR_BGR2GRAY)
        a, thresh = cv2.threshold(grey, 70, 255, cv2.THRESH_BINARY_INV)
        resized_digit = cv2.resize(thresh, (28, 28))
        os.remove(path)
        cv2.imwrite(path, resized_digit)
    os.chdir("../")