import random
import numpy as np
import torch
from torch import nn, cuda
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from generate_model import getDNN
from collections import defaultdict
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.use('tkagg')

"""""""""""""""""""""""""""""
SUMMARY OF SCRIPT
"""""""""""""""""""""""""""""
"""
    
    THIS SCRIPT CAN HELP YOU TRAIN A VOLUMETRIC DNN IN A REPEATED, NESTED, K-FOLD CV SCHEME. MAKE SURE YOU UNDERSTAND 
    THE ASSUMPTIONS THAT IT MAKES AS DOCUMENTED IN THE NEXT SECTION.
    
        * NESTING = split training data in outer folds into more inner folds that are used to select hyperparameters.
            Then collapse inner folds and repartition data into train and validation sets for retraining final network.
            Finally, test on test data.
        * IT WILL USE RANDOM SEARCH AND CAN TEST DIFFERENT DNN ARCHITECTURES DURING TUNING. 
        * IT CAN PLOT TRAINING DATA FOR YOU AS IT TRAINS. 
        * IT WILL DO SOME EARLY STOPPING BASED ON F1, ACCURACY AND/OR LOSS INDEPENDENTLY OR COMBINED.
        * IT CAN PRIORITIZE LOSS OR F1 SCORES DURING TUNING AND INDEPENDENTLY DURING TESTING 
            (or accuracy with a slight tweak)
        * IT CAN LOAD IN CV SCHEME FROM A PREVIOUS RUN FOR FAIRER MODEL COMPARISONS
    
    ------------------------
    HOW TO USE:     
        1. Alter the code in "settings and hyperparameters" to suit your needs
        2. Alter the code in "import data and preprocess" to load in your own labels and predictor matrix 
            (arranged in a 3d images x subjects matrix)
        3. Run the script
    * IF ON MAC, DO NOT USE DEFAULT OS INSTALL OF PYTHON AND DO NOT HIDE OSX UI ELEMENTS IN SETTINGS (MATPLOTLIB)
         
"""

"""""""""""""""""""""""""""""
!!! DANGER: ASSUMPTIONS !!!
"""""""""""""""""""""""""""""
"""
    * IF GPU IS AVAILABLE WILL USE THAT.
    
    * IF TUNING DNN ARCHITECUTRE OR COMPLEXITY LOOK AT CODE IN ./MODELS TO MAKE ANY CHANGES THAT MAY BE NECESSARY FOR 
        YOUR PURPOSE/DATA/LEVEL OF COMPLEXITY.  
        * TUNING NETWORK ARCHITECTURE WILL REQUIRE A CONDITIONAL STATEMENT IF USING SOMETHING OTHER THAN VGG OR RESNET 
    
    * THIS SCRIPT WILL RUN A (MANUAL) RANDOM SEARCH. 
        * YOUR HYPERPARAMETER SPACE MUST BE LARGE ENOUGH AND NUMBER OF SEARCH ITERATIONS SMALL ENOUGH TO 
            FIND UNIQUE SOLUTIONS BY FORCE. OTHERWISE YOU WILL GET STUCK IN A WHILE LOOP THAT SEARCHES FOR
            NEW SOLUTIONS THAT YOU HAVE NOT RANDOMLY SELECTED IN PREVIOUS ITERATIONS OF THE SEARCH. 
        
    * NOT TESTED WITH MULTICLASS BUT SHOULD SUPPORT IT.
        * MATRIX EXPANSION FOR x IN DATA EXTRACTOR SECTION MAY NEED TO BE EDITED BASED ON YOUR NEEDS
        * RESHAPING X IN DATA PREPROCESSING SECTION MAY HAVE TO BE EDITED
        * LOSS FUNCTION MAY NEED TO BE UPDATED IN MAIN CV LOOP (CURRENTLY USING ONLY BCEWITHLOGITS)
    
    * OPTIMIZER IS ADAM BUT WE USE ADAMW BECAUSE WE ASSUME YOU WANT TO TUNE L2. CHANGE IF NECESSARY
        * FOR ADAMW, WE MITIGATE LOSS OSCILLATIONS BY UPDATING EXPONENTIAL DECAY RATE TO INCREASE WEIGHT FOR PAST 
            GRADIENT ESTIMATE AND REDUCE WEIGHT FOR CURRENT ESTIMATE (USING BETA1 = BETA1*0.999).
    
    * WE ASSUME YOU WANT TO SAVE OUT ALMOST EVERYTHING (MOST SETTINGS, OUTER MODELS, TRAINING/TUNING CURVES FOR OUTER 
        FOLDS, BEST INNER FOLD PERFORMANCE). EDIT DICTIONARY AT THE END OF THE SCRIPT IF YOU DON'T.
        
    * IF PERFORMING REPEATED CV YOU CAN REPLICATE THE CV PARTITIONS LATER USING THE RANDOMLY GENERATED 
        SEEDS STORED/SAVED IN SDS WITH STRATIFIEDKFOLD (SEE CODE IN SETTING UP CV).
        
    * IF YOU ARE ON A MAC AND GETTING ERRORS, IT'S PROBABLY MATPLOTLIB. ENSURE YOU ARE NOT USING DEFAULT OS PYTHON. E.G.
        USE ANACONDA INSTALL OF PYTHON INSTEAD. IT'S DEFINITELY MATPLOTLIB IF ALL YOU SEE IS A BLACK FIGURE WHEN 
        TRAINING. IF YOU ARE HIDING YOUR TASKBAR, YOU MAY GET WARNINGS OR ERRORS (COMPATIBILITY ISSUES WITH MATPLOTLIB).
        
"""

"""""""""""""""""""""""""""""
SETTINGS AND HYPERPARAMETERS FOR EVALUATION
"""""""""""""""""""""""""""""
# START USER-DEFINED VARIABLES
# FOR CV/GENERAL
gpu = True # If false will avoid using gpu though it may be available
ldCV = False # if true, will load cv scheme from file oF and overwrite all the other variables in oF after training (so make a copy!)
kfo = 6 # number of outer folds for testing
kfi = 6 # number of inner folds for tuning
valH = 0.2 # validation holdout for final retraining of CNN (i.e., after tuning we collapse inner folds and repartition training data into train and validation sets for retraining the network)
hiter = 50 #40 # number of random search iterations over hyperparameter space
reps = 20 # repeats of entire CV scheme
inc = 1 # number of channels in network (e.g., how many modalities of images for each subject?)

# FOR HYPERPARAMETER SPACE
lr_space = [0.000001, 0.000005, 0.000007, 0.00001, 0.00003, 0.00005, 0.00007, 0.0001, 0.001, 0.01] #np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]) # learning rates to evaluate
drp_space = [0.8, 0.7, 0.6, 0.4, 0.2, 0.1] #np.array([0.8, 0.7, 0.6, 0.4]) # drop rates to evaluate (only applicable for VGG at the moment)
l2_space = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005] #np.array([0.1, 0.05, 0.005, 0.0005]) # l2 norm penalties to add to optimizer
dpth_space = [1,2,3,4] #np.array([1,2,3,4]) # network depths to test (see relevant network code in ./models; for vgg and resnet we use relative complexity but for other networks this value should be set to the number of layers in the network)
network_space = ['vgg'] # which networks to evaluate? #['vgg', 'cnn', 'C3DNet', 'resnet', 'ResNetV2', 'ResNeXt', 'ResNeXtV2', 'WideResNet', 'PreActResNet','EfficientNet', 'DenseNet', 'ShuffleNet', 'ShuffleNetV2', 'SqueezeNet', 'MobileNet', 'MobileNetV2']

# FOR TRAINING
num_epochs = 200 #400 #400 #600 # number of epochs for tuning (inner folds training)
num_epochs_test = 300 #500 #800 #1200 # number of epochs for testing (outer folds training)
mbs = 128 # mini batch size; larger is better for GPU
sT = 'standard' # can be standard or balanced -- this is for mini batch 'stratification'. If balanced, no loss function weighting and instead we bootstrap samples for balance in the dataloader. If standard then loss function weighting. If stratification, mini batches will simply be balanced by class (this last option is still in development, do not use)
priorityMetricTune = 'F1' # should we tune model based on best loss for hyperparameter sets or best F1?
priorityMetricTest = 'F1' # should we retain model based on best loss or best F1 for final test data (i.e., which model to pluck during training: best based on F1 or loss for validation data?)
tuneMetricCeil = 0.9 # what should the max metric be while tuning (i.e., if a hyperparameter set is higher than this value we will ignore it because it's probably overfitting)

# FOR STOPPING
perfThresh = 1 # threshold for how many 100% accuracy epochs can occur in a row before training termination (accuracy for training data in inner folds)
patThresh = 50 #150 # validation patience threshold (based on loss *or* F1). After this many epochs, if loss or F1 has not improved, we terminate. (also for training data in inner folds)
perfThreshTest = 50 #5 # same as perfThresh above but applied only to testing data (i.e., when retraining the model for final outer loop predictions)
patThreshTest = 200 # same as patThresh above but applied only to testing data (i.e., when retraining the model for final outer loop predictions)
perfThreshVal = 0.92 # value that perfThresh and perfThreshTest looks for (i.e., it can be something other than 100%)

# FOR FEEDBACK/OUTPUT
verbose = 1 #2 # 1 is give me the important bits, 2 is give me everything as printed text, 3 is give me most things as text and plot inner training instead of printing, 0 is give me nothing (only applies to inner fold loop)
verboseTest = 3 # same as above but only applies for outer fold loop
oF = 'FirstTry_CV.pkl' # save the outputs to this directory/file
# END USER-DEFINED VARIABLES

"""""""""""""""""""""""""""""
IMPORT DATA AND PREPROCESS
"""""""""""""""""""""""""""""
# START USER-DEFINED VARIABLES
# load data -- !INSERT YOUR OWN CODE HERE!
inmat = loadmat('PyTorchInput_withFolds.mat') # this is my data, load yours however
yT = inmat['wabClassi'].flatten() # put your classes in a variable called yT. Here I flatten it, you may not have to. Check if your data has more than one dimension. It shouldn't.
#yF = yT
yF = inmat['wabClass2i'].flatten() # for your data, make this identical to yT like the above line 130. This is the vector on which stratitfication will be based. I was experimenting with using more granular classes for stratification while still using yT for testing
x = inmat['data'] # this is a 4D predictor matrix. It needs to be shaped as such: (image dimension 1, image dimension 2, image dimension 3, subjects)
del(inmat)
# END USER-DEFINED VARIABLES

# reshape input data and rescale if necessary
xr = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2], x.shape[3]))
xr = (xr - xr.min()) / (xr.max() - xr.min()) * 2 - 1 #our data has only 4 possible integers so we just rescale from -1 to 1 *EACH IMAGE SEPARATELY*; you could also try normalization or something simple like this: xr = xr / xr.max()

"""""""""""""""""""""""""""""
CUSTOM SAMPLER FOR STRATIFICATION OF MINI BATCHES (NOT FULLY FIXED YET)
"""""""""""""""""""""""""""""
#this requires some debugging...
class stratSamp(Sampler):
    def __init__(self, labels, batch_size, drop_last=False):
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

    def __iter__(self):
        batch = []
        for label in cycle(self.label_to_indices.keys()):
            indices = self.label_to_indices[label]
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if self.drop_last and len(batch) > 0:
                break

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        else:
            return (len(self.labels) + self.batch_size - 1) // self.batch_size


"""""""""""""""""""""""""""""
DATA EXTRACTOR
"""""""""""""""""""""""""""""
def torchDataExtractor(id, xr, y, x, mbs, sampType='stratified'):
    """
    EXTRACT INDICES FROM RESHAPED BRAIN DATA (TO 2D MATRIX), CONVERT THE RESULT BACK TO 4D MATRIX AND PREPROCESS MATRIX FOR DNN IN PYTORCH
        - I.E., DO SOME MATRIX RESHAPING AND SETUP TENSORDATASET, SAMPLER, AND DATALOADER
        - ASSUMES IN_CHANNELS IS 1

    id is a set of indices for xr. xr is a predictor matrix where rows are samples (for id), cols are subjects/images.
    x is the *size* of the 4d matrix of original images that were reshaped into xr. Rows in xr map onto 1st, 2nd and 3rd dims of x that were collapsed.
    yT are samples
    mbs is the mini batch size

    output: tensor dataset for pytorch
    """
    tmp = xr[:, id]
    tmp = np.reshape(tmp, (x[0], x[1], x[2], tmp.shape[1]))
    tmp = np.transpose(tmp, (3, 0, 1, 2))
    tmp = torch.from_numpy(np.expand_dims(tmp, axis=1))

    tmp2 = torch.from_numpy(np.transpose(y[id])).to(torch.uint8)
    tmp3 = tmp2.long()

    if sampType == 'balanced':
        # this implements a sampler with replacement to have balanced minibatches
        cw = torch.tensor([1 / class_count for class_count in torch.bincount(tmp3)])
        cw = cw.view(1,-1)
        sw = cw[:,tmp3].squeeze()
        samp = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
        shuf = False
        drp = False

    elif sampType == 'stratified':
        # this implements a sampler with stratification but no class imblanace
        samp = stratSamp(tmp2,mbs,drop_last=True)
        shuf = False
        drp = True

    elif sampType == 'standard':
        samp = None
        shuf = True
        drp = False

    o = DataLoader(TensorDataset(tmp.float(), tmp2.float()), batch_size=mbs, sampler=samp, shuffle=shuf, drop_last=drp)
    return o

"""""""""""""""""""""""""""""
PERFORMANCE EVALUATOR
"""""""""""""""""""""""""""""
def getPerf(tei, model, lossFn=None, model_state=None, multiClass=False):
    """
    tei is a test dataset, model is a CNN model, lossFn is a prespecified loss function, model_state is a model state.
    Set mutliClass to true to evaluate macro F1, accuracy, etc across classes.

    Returns accuracy, loss, precision, recall, and F1 score. Also returns predictions and actual labels
    (since you will probably be shuffling them in the dataloader)

    """
    if model_state is not None:
        model.load_state_dict(model_state)

    model.eval()
    loss = 0
    yh = []
    y = []
    val_correct = 0
    val_tp = 0
    val_fp = 0
    val_fn = 0
    with torch.no_grad():
        for val_idx, (data2, target2) in enumerate(tei):
            if next(model.parameters()).is_cuda:
                data2 = data2.to('cuda')
                output2 = model(data2).squeeze()
                output2 = output2.to('cpu')
            else:
                output2 = model(data2).squeeze()

            if lossFn is not None:
                loss += lossFn(output2, target2.squeeze())

            if not multiClass:
                prob = torch.sigmoid(output2)
                pred = (prob > 0.5).float()
                val_correct += pred.eq(target2.view_as(pred)).sum().item()
                val_tp += ((pred == 1) & (target2.view_as(pred) == 1)).sum().item()
                val_fp += ((pred == 1) & (target2.view_as(pred) == 0)).sum().item()
                val_fn += ((pred == 0) & (target2.view_as(pred) == 1)).sum().item()
            else:
                prob = torch.softmax(output2, dim=1)
                pred = prob.argmax(dim=1)
            if pred.numel() == 1: # if batch is 1, yh will not work with tolist as its a tensor scalar that may also be just a 0 (class)
                yh.append(pred)
            else:
                yh.extend(pred.tolist())

            #yh.extend(pred.tolist())
            y.extend(target2.tolist())

    if lossFn is not None:
        loss /= len(tei)
    else:
        loss = float('nan')

    if not multiClass:
        acc = val_correct / len(tei.dataset)
        prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        rec = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    else:
        acc = accuracy_score(y, yh)
        prec, rec, f1, _ = precision_recall_fscore_support(y, yh, average='weighted', zero_division=0)

    return loss, f1, rec, prec, acc, yh, y


"""""""""""""""""""""""""""""
MODEL TRAINER
"""""""""""""""""""""""""""""
def dnnTrainer(tri, tei, inc, lossFn, num_epochs, lr, drp, l2, dpth, patThresh, perfThresh, perfThreshVal, verbose, cf=True, title=None):
    """
    tri is a training dataset, tei is a test dataset, lossFn is a loss function, etc...see all of these parameters in
    the parameter section for more info.

    If cf is true, it will close the figure (if you're plotting performance across epochs)
    If a title is passed in figures will be given the title passed in
    (to keep track of e.g., outer folds, repeats, search iterations, etc)
    """
    # hard coded variables
    beta1 = 0.9
    bestF1v = 0 # winning F1 score
    bestLossv = 10000000000000000 # winning loss
    lossPat = 0 # counter for loss increasing
    f1Pat = 0 # counter for f1 decreasing
    perf = 0 # counter for 100 percent accuracy
    lid = 0
    fid = 0

    # temporary tracking of performance metrics within epoch
    f1vm = np.zeros(num_epochs)
    accvm = np.zeros(num_epochs)
    acctm = np.zeros(num_epochs)
    lossvm = np.zeros(num_epochs)
    losstm = np.zeros(num_epochs)

    # tracking of performance metrics across epochs
    losst = torch.zeros(len(tri))
    lossv = torch.zeros(len(tri))
    accv = torch.zeros(len(tri))
    f1v = torch.zeros(len(tri))
    acct = torch.zeros(len(tri))

    # setup model
    model = getDNN(cnn_name='vgg', model_depth=dpth, n_classes=1, in_channels=inc, sample_size=len(tri.dataset), drop=drp, cuda=gpu) # 13 works BUT ALSO, 19 with drop = 0.7
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=l2, amsgrad=False, eps=1e-08) # 0.000005 lr[lri]

    # track model state
    wmodel = None
    wmodel2 = None

    if verbose >= 3:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for e in range(num_epochs):
        if verbose >= 2:
            print('-----Epoch:', e+1)
        for batch_idx, (data, target) in enumerate(tri):

            # train model...
            model.train()
            if next(model.parameters()).is_cuda: # we already check for cuda when building the model, now just carry over data if necessary
                data = data.to('cuda')
            optimizer.zero_grad()
            output = model(data).squeeze()
            if next(model.parameters()).is_cuda:
                output = output.to('cpu')
            prob1 = torch.sigmoid(output)
            pred1 = (prob1 > 0.5).float()
            acct[batch_idx] = pred1.eq(target.view_as(pred1)).sum().item() / len(target)

            loss = lossFn(output, target)
            loss.backward()
            losst[batch_idx] = loss.item()
            optimizer.step()
            beta1 = beta1 * 0.999

            # now eval model...
            lossv[batch_idx], f1v[batch_idx], _, _, accv[batch_idx], _, _ = getPerf(tei, model, lossFn, model_state=None, multiClass=False)

        f1vm[e] = np.mean(f1v[:].numpy())
        lossvm[e] = np.mean(lossv[:].numpy())
        accvm[e] = np.mean(accv[:].numpy())
        losstm[e] = np.mean(losst[:].numpy())
        acctm[e] = np.mean(acct[:].numpy())

        if f1vm[e] > bestF1v:
            bestF1v = f1vm[e]
            f1Pat = 0
            wmodel = model.state_dict()
            fid = e
        else:
            f1Pat += 1

        if lossvm[e] < bestLossv:
            bestLossv = lossvm[e]
            wmodel2 = model.state_dict()
            lossPat = 0
            lid = e
        else:
            lossPat += 1

        if verbose == 2:
            print('Validation set-- BEST F1: {:.4f}, Average loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}'.format(
                    bestF1v, lossvm[e], accvm[e], f1vm[e]))
            print('Training set-- Loss: {:.4f}, Accuracy: {:.4f}'.format(
                    losstm[e], acctm[e]))
        elif verbose == 3:
            plt.rcParams.update({'font.size': 12, 'axes.titlesize': 16})
            ax1.spines['bottom'].set_linewidth(4)
            ax1.spines['top'].set_linewidth(4)
            ax1.spines['left'].set_linewidth(4)
            ax1.spines['right'].set_linewidth(4)
            ax1.clear()
            ax1.plot(range(1, e+2), losstm[:e+1], color = (0.415686, 0.172549, 0.439216), label='Training Loss',linewidth=3.5)
            ax1.plot(range(1, e+2), lossvm[:e+1], color = (0.721569, 0.231373, 0.368627), label='Validation Loss',linewidth=6)
            ax1.grid(axis='y',linewidth=2,linestyle='--')
            ax1.scatter(lid + 1, bestLossv, s=250, c='r', marker='o', zorder=3)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()

            ax2.clear()
            ax2.spines['bottom'].set_linewidth(4)
            ax2.spines['top'].set_linewidth(4)
            ax2.spines['left'].set_linewidth(4)
            ax2.spines['right'].set_linewidth(4)
            ax2.plot(range(1, e+2), acctm[:e+1], color = (0.415686, 0.172549, 0.439216), label='Training Accuracy',linewidth=3.5)
            ax2.plot(range(1, e+2), accvm[:e+1], color = (0.721569, 0.231373, 0.368627), label='Validation Accuracy',linewidth=3.5)
            ax2.plot(range(1, e+2), f1vm[:e+1], color = (0.976471, 0.929412, 0.411765), label='Validation F1 Score',linewidth=6)
            ax2.grid(axis='y',linewidth=2,linestyle='--')
            ax2.scatter(fid + 1, bestF1v, s=250, c='r', marker='o', zorder=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy and F1')
            ax2.legend()
            if title is not None:
                ax1.set_title(title, fontsize=16)

            plt.pause(0.01)

        if acctm[e] >= perfThreshVal:
            perf += 1
        else:
            perf = 0

        if f1Pat >= patThresh or lossPat >= patThresh:
            if verbose >= 2:
                print(f"Early stopping on epoch {e} triggered: Validation F1 score or loss did not improve for {patThresh} consecutive epochs.")
            break

        if perf >= perfThresh:
            if verbose >= 2:
                print(f"Early stopping on epoch {e} triggered: Number of times training accuracy hit {perfThreshVal*100} % exceeded: {perfThresh} ")
            break

    if verbose == 3 and cf:
        plt.close(fig)

    return f1vm, lossvm, accvm, acctm, losstm, bestF1v, bestLossv, wmodel, wmodel2, model

"""""""""""""""""""""""""""""
INITIALIZE DATA TRACKERS (ALL WILL BE PICKL'D)
"""""""""""""""""""""""""""""
hypeTrackerF1o2 = []
hypeTrackerLosso2 = []
valF1Trackero2 = []
testF1Trackero2 = []
trainF1Trackero2 = []
modelTracker = []
modelStateTracker = []
modelStateTracker2 = []

trainLossTrackero2 = []
vallossTrackero2 = []
testlossTrackero2 = []
valtrlossALLTrackero2 = []
valtelossALLTrackero2 = []
valteF1ALLTrackero2 = []
valtrACCALLTrackero2 = []
valteACCALLTrackero2 = []

valaccTrackero2 = []
testaccTrackero2 = []
finpredTrackerv = []
finpredTrackerYh = []
finpredTrackerY = []
finpredTrackervYh = []
finpredTrackervY = []

"""""""""""""""""""""""""""""
START CV
"""""""""""""""""""""""""""""
# we will use random seed integers to preallocate folds...
if ldCV:
    with open(oF, 'rb') as f:
        sds = pickle.load(f)['sds']
else:
    sds = [random.randint(0, 4294967295) for _ in range(reps)] #that number is the max possible allowed by random

for r in range(reps):
    if verbose >= 1:
        print('-------------------------------------------------------------------------------------------------------Repeat is:', r+1)

    xr = np.transpose(xr) # need to transpose for sklearn
    outer_cv = StratifiedKFold(n_splits=kfo, shuffle=True, random_state=sds[r])
    inner_cv = StratifiedKFold(n_splits=kfi, shuffle=True, random_state=sds[r])

    # create inner/outer folds
    nested_indices = []
    for outer_train_idx, outer_test_idx in outer_cv.split(xr, yF):
        inner_indices = []
        for inner_train_idx, inner_val_idx in inner_cv.split(xr[outer_train_idx,:], yF[outer_train_idx]):
            inner_train_idx_global = [outer_train_idx[idx] for idx in inner_train_idx]
            inner_val_idx_global = [outer_train_idx[idx] for idx in inner_val_idx]
            inner_indices.append((inner_train_idx_global, inner_val_idx_global))
        nested_indices.append((outer_train_idx, outer_test_idx, inner_indices))

    # Create new train/validation sets from the outer training for retraining the CNN after tuning
    outer_train_val = []
    for outer_train_idx, outer_test_idx, inner_indices in nested_indices:
        new_train_idx, val_idx = train_test_split(outer_train_idx, test_size=valH, stratify=yF[outer_train_idx], shuffle=True, random_state=sds[r])
        outer_train_val.append((new_train_idx, val_idx))
    xr = np.transpose(xr) # at this point xr should be features x samples

    for o in range(kfo):
        best_score_F1 = 0 # track best score you get during tuning FOR F1
        best_score_loss = 1000000000000000000000000000000 # track best score you get during tuning
        best_hyperparams_F1 = None # track best hyperparams you get when tuning
        best_hyperparams_loss = None # track best hyperparams you get when tuning
        hyperparameters_chosen = [] # track hyperparams so we do not choose repeats in random search
        if verbose >= 1:
            print(f"----------------------------------------------------------------------Outer fold is {o+1} of {kfo}")
        for h in range(hiter):
            if verbose >= 1:
                print(f"---------------------------------------Random subset is: {h+1} of {hiter}")

            while True:
                # Choose hyperparameters at random to test
                lr = random.choice(lr_space)
                drp = random.choice(drp_space)
                l2 = random.choice(l2_space)
                dpth = random.choice(dpth_space)
                netType = random.choice(network_space)
                hyperparams = (lr, drp, l2, dpth, netType)

                # Check if the hyperparameters have already been chosen
                if hyperparams not in hyperparameters_chosen:
                    hyperparameters_chosen.append(hyperparams)
                break

            avg_inner_score_F1 = 0
            avg_inner_score_loss = 0

            if verbose >= 1:
                print(f"----> Testing hyperparameters: {hyperparams}")

            for i in range(kfi):
                if verbose >= 2:
                    print(f"----------------Inner fold is: {i+1}  of {kfi}")
                if sT == 'standard':
                    # weighted loss function
                    tmp = np.bincount(yT[nested_indices[o][2][i][0]]) # get inverse class weights for loss function
                    w = len(nested_indices[o][2][i][0]) / tmp.astype(float)
                    lossFn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w[1]))
                elif sT == 'balanced':
                    # no weights because we will bootstrap the undersampled class
                    lossFn = nn.BCEWithLogitsLoss()

                tri = torchDataExtractor(nested_indices[o][2][i][0], xr, yT, x.shape, mbs, sampType=sT)
                tei = torchDataExtractor(nested_indices[o][2][i][1], xr, yT, x.shape, mbs, sampType='standard') # this has to be standard because you should not be bootstrapping validation data. makes no sense.

                # train the model
                f1vm, lossvm, accvm, acctm, losstm, bestF1v, bestLossv_val, _, _, _ = dnnTrainer(tri, tei, inc, lossFn, num_epochs, lr, drp, l2, dpth, patThresh, perfThresh, perfThreshVal, verbose, title=f"Repeat {r+1} / {reps}. Test fold: {o+1} / {kfo}, Inner fold: {i+1} / {kfi} and random search: {h+1} / {hiter}")

                avg_inner_score_loss += bestLossv_val / (kfi - 1) # average score for hyperparameter tuning within inner fold CHANGE TO bestF1v to select based on f1 and bestLossv_val to select based on loss
                avg_inner_score_F1 += bestF1v / (kfi - 1)

            # keep score and params if it is the best we've seen yet (based on F1 or loss)
            if avg_inner_score_F1 > best_score_F1:
                if priorityMetricTune == 'F1' and avg_inner_score_F1 < tuneMetricCeil or priorityMetricTune != 'F1':
                    best_score_F1 = avg_inner_score_F1
                    best_hyperparams_F1 = (lr, drp, l2, dpth, netType)
                    if verbose >= 1:
                        print('******************* Best F1 score is now updated to :', avg_inner_score_F1)
                        print(f"******************* HYPERPARAMETERS: {hyperparams}")
                else:
                    print(f"******************* Found better F1 score {avg_inner_score_F1} but it's above ceiling:{tuneMetricCeil}")

            if avg_inner_score_loss < best_score_loss:
                if (priorityMetricTune == 'loss' and avg_inner_score_loss > tuneMetricCeil) or priorityMetricTune != 'loss':
                    best_score_loss = avg_inner_score_loss
                    best_hyperparams_loss = (lr, drp, l2, dpth, netType)
                    if verbose >= 1:
                        print('******************* Best loss is now updated to :', avg_inner_score_loss)
                        print(f"******************* HYPERPARAMETERS: {hyperparams}")
                else:
                    print(f"******************* Found better loss {avg_inner_score_F1} but it's below floor:{tuneMetricCeil}")


        # get best hyperparams for the outer fold
        if priorityMetricTune == 'F1':
            if best_hyperparams_F1 is not None: # we need this in case you are doing something dumb (e.g., while debugging) and the best score is as bad as what we initialized (i.e., worst possible score)
                lr, drp, l2, dpth, netType = best_hyperparams_F1
        elif priorityMetricTune == 'loss':
            if best_hyperparams_loss is not None: # we need this in case you are doing something dumb (e.g., while debugging) and the best score is as bad as what we initialized (i.e., worst possible score)
                lr, drp, l2, dpth, netType = best_hyperparams_loss

        tri = torchDataExtractor(outer_train_val[o][0], xr, yT, x.shape, mbs, sampType=sT)
        vali = torchDataExtractor(outer_train_val[o][1], xr, yT, x.shape, mbs, sampType='standard') # again, would make no sense to bootstrap undersampled class here
        tei = torchDataExtractor(nested_indices[o][1], xr, yT, x.shape, mbs, sampType='standard') # same...don't touch the test data

        if sT == 'standard':
            tmp = np.bincount(yT[outer_train_val[o][0]]) # get inverse class weights for loss function...again (outer folds can be slightly different)
            w = len(outer_train_val[o][0]) / tmp.astype(float)
            lossFn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w[1]))
        elif sT == 'balanced':
            lossFn = nn.BCEWithLogitsLoss()

        # retrain best network and get its performance
        f1i_val, lossi_val, acci_val, accit_val, lossit_val, bestF1_val, bestLossv_val, wmodel, wmodel2, model = dnnTrainer(tri, vali, inc, lossFn, num_epochs_test, lr, drp, l2, dpth, patThreshTest, perfThreshTest, perfThreshVal, verboseTest, cf=False, title=f"Test fold: {o+1} / {kfo} for repeat {r+1} / {reps}")

        if priorityMetricTest == 'F1':
            lossval, f1val, recval, precval, accval, yhval, ysval = getPerf(vali, model, lossFn, wmodel, multiClass=False) # change this to wmodel to select best model based on F1, change this to wmodel2 to select best model based on loss
            loss, f1, rec, prec, acc, yh, ys = getPerf(tei, model, lossFn, wmodel, multiClass=False) # change this to wmodel to select best model based on F1, change this to wmodel2 to select best model based on loss
            if verbose >= 1:
                print(f"Inner validation F1: {best_score_F1}, Outer validation F1: {f1val}, Test F1: {f1}, Outer validation loss: {lossval}, Test loss: {loss}")
        elif priorityMetricTest == 'loss':
            lossval, f1val, recval, precval, accval, yhval, ysval = getPerf(vali, model, lossFn, wmodel2, multiClass=False) # change this to wmodel to select best model based on F1, change this to wmodel2 to select best model based on loss
            loss, f1, rec, prec, acc, yh, ys = getPerf(tei, model, lossFn, wmodel2, multiClass=False) # change this to wmodel to select best model based on F1, change this to wmodel2 to select best model based on loss
            if verbose >= 1:
                print(f"Inner validation loss: {best_score_loss}, Outer validation loss: {lossval}, Test loss: {loss}, Outer validation F1: {f1val}, Test f1: {f1}")

        if verbose >= 1:
            print(f"----> BEST HYPERPARAMETERS FOR F1: {best_hyperparams_F1}")
            print(f"----> BEST HYPERPARAMETERS FOR LOSS: {best_hyperparams_loss}")

        # save everything....
        hypeTrackerF1o2.append(best_hyperparams_F1)
        hypeTrackerLosso2.append(best_hyperparams_loss)

        valF1Trackero2.append(f1val)
        testF1Trackero2.append(f1)

        valaccTrackero2.append(accval)
        testaccTrackero2.append(acc)

        trainF1Trackero2.append(best_score_F1)
        trainLossTrackero2.append(best_score_loss)

        vallossTrackero2.append(lossval)
        testlossTrackero2.append(loss)

        valtrlossALLTrackero2.append(lossi_val)
        valtelossALLTrackero2.append(lossit_val)

        valteF1ALLTrackero2.append(f1i_val)

        valtrACCALLTrackero2.append(acci_val)
        valteACCALLTrackero2.append(accit_val)

        modelStateTracker.append(wmodel)
        modelStateTracker2.append(wmodel2)
        modelTracker.append(model)

        finpredTrackerYh.append(yh)
        finpredTrackerY.append(ys)

        finpredTrackervYh.append(yhval)
        finpredTrackervY.append(ysval)

    # save outputs on each repeat to be safe...
    dd = {
        'trainLossTrackero2': trainLossTrackero2,
        'vallossTrackero2': vallossTrackero2,
        'testlossTrackero2' : testlossTrackero2,
        'valtrlossALLTrackero2' :valtrlossALLTrackero2,
        'valtelossALLTrackero2' : valtelossALLTrackero2,
        'valteF1ALLTrackero2' : valteF1ALLTrackero2,
        'valtrACCALLTrackero2' : valtrACCALLTrackero2,
        'valteACCALLTrackero2' : valteACCALLTrackero2,
        'hypeTrackerF1o2' : hypeTrackerF1o2,
        'hypeTrackerLosso2' : hypeTrackerLosso2,
        'valF1Trackero2': valF1Trackero2,
        'testF1Trackero2': testF1Trackero2,
        'trainF1Trackero2': trainF1Trackero2,
        'valaccTrackero2': valaccTrackero2,
        'testaccTrackero2': testaccTrackero2,
        'modelStateTracker': modelStateTracker,
        'modelStateTracker2': modelStateTracker2,
        'modelTracker': modelTracker,
        'finpredTrackerYh': finpredTrackerYh,
        'finpredTrackerY': finpredTrackerY,
        'finpredTrackervYh': finpredTrackervYh,
        'finpredTrackervY': finpredTrackervY,
        'sds': sds,
        'kfo': kfo,
        'kfi': kfi,
        'valH': valH,
        'hiter': hiter,
        'reps': reps,
        'lr_space': lr_space,
        'drp_space': drp_space,
        'l2_space': l2_space,
        'dpth_space': dpth_space,
        'network_space': network_space,
        'num_epochs': num_epochs,
        'num_epochs_test': num_epochs_test,
        'mbs': mbs,
        'sT': sT,
        'perfThresh': perfThresh,
        'patThresh': patThresh,
        'perfThreshTest': perfThreshTest,
        'patThreshTest': patThreshTest,
        'perfThreshVal': perfThreshVal
    }

    with open(oF, 'wb') as f:
        pickle.dump(dd, f)
