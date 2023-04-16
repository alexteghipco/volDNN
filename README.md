# volDNN: Relatively flexible examples for building DNNs with volumetric MRI data in pycharm. 

I haven't been able to find many thorough tutorials for building volumetric DNN models so I thought I would share one. volDNN can help you train a volumetric DNN in a repeated, nested, k-fold cross-validation scheme using pytorch. 
    
        * HERE, NESTING = split training data in outer folds into more inner folds that are used to select hyperparameters.
            Then collapse inner folds and repartition data into train and validation sets for retraining final network.
            Finally, test on test data left out in the initial outer folds.
            
Some features built into volDNN: 
        * IT WILL USE RANDOM SEARCH AND CAN TEST DIFFERENT DNN ARCHITECTURES DURING TUNING. 
        * IT CAN PLOT TRAINING DATA FOR YOU AS IT TRAINS. 
        * IT WILL DO SOME EARLY STOPPING BASED ON F1, ACCURACY AND/OR LOSS INDEPENDENTLY OR COMBINED.
        * IT CAN PRIORITIZE LOSS OR F1 SCORES DURING TUNING AND INDEPENDENTLY DURING TESTING 
            (OR ACCURACY WITH A SLIGHT TWEAK)
        * IT CAN LOAD IN A CV SCHEME FROM A PREVIOUS RUN OF VOLDNN FOR FAIRER MODEL COMPARISONS
    
    ------------------------
    HOW TO USE:     
        1. Open volDNN.py and read about the assumptions it makes.
        2. Alter the code in "settings and hyperparameters" section of volDNN.py to suit your needs
        3. Alter the code in "import data and preprocess" to load in your own labels and predictor matrix 
            (arranged in a 3d images x subjects matrix)
        3. Run the script
        
    * pro tips for mac users: do not use default python os install or hide osx GUI elements in system preferences (matplotlib compatibility issues)
