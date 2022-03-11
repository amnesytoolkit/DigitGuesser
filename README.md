# DigitGuesser
Draw a digit and the AI will try to recognise it. Made by Kabir Paolo Singh Bertolani (amnesy) && Luca Bertocchi

Note: Trained with MNIST data (and/or data from other sources) - you should try to train the AI with your own data

Instruction to deploy this project:

         - First, you should run the train.py file, that should create the mnist_model directory containing the model
         - Then, just run main.py and browse to localhost:5000
         
Note: Every digit created in the canvas and submitted will be saved in static/

Steps To replicate the model (88% accuracy):
- Download the dataset from [Kaggle](https://www.kaggle.com/dhruvildave/english-handwritten-characters-dataset)
- Extract and put all the data in the dataset folder
- Rename the extracted folder "archive" to "data" and delete the old zip archive
- Run dataset_makers/normalize.py
- Delete (If the script is not fully working) the folders "datasets/data/Img" and the file english.csv
- I removed (for training purposes) all the folders of lowercase letters, so this model will only work with 36 characters. To change this, change the number of neurons in the final layer of the train.py file and adjust the label list in the main.py file.
- You should have 36 folders in datasets/data
- Run elaborate_white_images.py
- Run augment_data.py
- Run train.py

----> This project was just an exercise.