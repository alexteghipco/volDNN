# volDNN

This package might help you train volumetric DNN models (think trained on 3D MRI data) in a repeated, nested, k-fold cross-validation scheme using pytorch. It contains code used for this preprint, where we showed 3D CNNs improve classification of chronic stroke patients with severe language deficits by identifying unique spatially dependent patterns unavailable to classical machine learning methods: https://www.researchsquare.com/article/rs-3126126/v1
    
Here nesting means: split training data in outer folds into more inner folds that are used to select hyperparameters. Then collapse inner folds and repartition data into train and validation sets for retraining final network. Finally, test on test data left out in the initial outer folds.
            
Some features built into volDNN: 
* it will use random search and can test different DNN architectures during tuning (not just different CNNs, but note you may have to edit some of the code depending on what you choose as this is not a fully implemented feature yet)
* it can plot training data for you as it trains
* it will do some early stopping based on F1, accuracy, correlation and/or loss independently or combined
* it can prioritize loss or f1 scores during tuning and independently during testing (or accuracy with a slight tweak)
* it can load in a cv scheme from a previous run of volDNN for fairer model comparisons
* it can generate some gradCam++ maps from automatically saved .pkl files(see gradCAMer.py)

How to use:
1. Check dependencies. Either pip install -r requirements.txt or conda env create -f conda_requirements.yaml (note, matplotlib is in the list but is not strictly necessary to run volDNN.py)
2. Open volDNN.py and read about the assumptions it makes.
3. Alter the code in "settings and hyperparameters" section of volDNN.py to suit your needs
4. Alter the code in "import data and preprocess" to load in your own labels and predictor matrix 
    (arranged in a 3d images x subjects matrix)
5. Run the script
        
    * pro tips for mac users: do not use default python os install or hide osx GUI elements in system preferences (matplotlib compatibility issues)

## For replication of preprint results

You may have to change some of the default settings to be completely consistent with what we used in the preprint. Training a deep learning model in a repeated nested fashion like this takes substantial time. We confirmed in our data that using the following truncated set of parameters works nearly as well: 
- highest drop rate (~0.8) of range in settings
- lowest L2 penalty (~0.001) of range in settings
- most complex architecture (3) of range in settings
- highest learn rate (~0.0001) of range in settings
