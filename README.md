# volDNN

I haven't been able to find many thorough tutorials for building volumetric DNN models so I thought I would share one. volDNN can help you train a volumetric DNN (e.g., using MRI data) in a repeated, nested, k-fold cross-validation scheme using pytorch. 
    
Here nesting means: split training data in outer folds into more inner folds that are used to select hyperparameters. Then collapse inner folds and repartition data into train and validation sets for retraining final network. Finally, test on test data left out in the initial outer folds.
            
            
Some features built into volDNN: 

* it will use random search and can test different DNN architectures during tuning. 
* it can plot training data for you as it trains. 
* it will do some early stopping based on F1, accuracy and/or loss independently or combined.
* it can prioritize loss or f1 scores during tuning and independently during testing (or accuracy with a slight tweak
* it can load in a cv scheme from a previous run of volDNN for fairer model comparisons


How to use:
1. Check dependencies. Either pip install -r requirements.txt or conda env create -f conda_requirements.yaml (note, matplotlib is in the list but is not strictly necessary to run volDNN.py)
2. Open volDNN.py and read about the assumptions it makes.
3. Alter the code in "settings and hyperparameters" section of volDNN.py to suit your needs
4. Alter the code in "import data and preprocess" to load in your own labels and predictor matrix 
    (arranged in a 3d images x subjects matrix)
5. Run the script
        
    * pro tips for mac users: do not use default python os install or hide osx GUI elements in system preferences (matplotlib compatibility issues)
