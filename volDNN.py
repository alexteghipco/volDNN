import random
import numpy as np
import torch
from torch import nn, cuda
from torch.nn.parallel import DataParallel
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from generate_model import getDNN
from collections import defaultdict
from itertools import cycle
import pickle
import copy
import pandas

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
        * IT CAN USE COSINE ANNEALING WITH WARM RESTARTS FOR SCHEDULING LEARNING RATE
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
    * IF YOU ASK FOR FIGURE PLOTS W/MATPLOTLIB EXPECT SOME DELAYS

"""

"""""""""""""""""""""""""""""
!!! DANGER: ASSUMPTIONS !!!
"""""""""""""""""""""""""""""
"""
    * IF GPU IS AVAILABLE WILL USE THAT.

    * IF TUNING DNN ARCHITECUTRE OR COMPLEXITY LOOK AT CODE IN ./MODELS TO MAKE ANY CHANGES THAT MAY BE NECESSARY FOR 
        YOUR PURPOSE/DATA/LEVEL OF COMPLEXITY.  
        * TUNING NETWORK ARCHITECTURE WILL REQUIRE INSERTING A CONDITIONAL STATEMENT IF USING SOMETHING OTHER THAN VGG 
        OR RESNET .

    * THIS SCRIPT WILL RUN A (MANUAL) RANDOM SEARCH IF THERE IS MORE THAN 1 SET OF HYPERPARAMETERS.

    * NOT TESTED WITH MULTICLASS BUT HAS SOME SUPPORT FOR IT.
        * MATRIX EXPANSION FOR x IN DATA EXTRACTOR SECTION MAY NEED TO BE EDITED BASED ON YOUR NEEDS
        * RESHAPING X IN DATA PREPROCESSING SECTION MAY HAVE TO BE EDITED
        * LOSS FUNCTION MAY NEED TO BE UPDATED IN MAIN CV LOOP (CURRENTLY USING ONLY BCEWITHLOGITS)

    * OPTIMIZER IS ADAM BUT WE USE ADAMW BECAUSE WE ASSUME YOU WANT TO TUNE L2. CHANGE IF NECESSARY.
    
    * WE ASSUME YOU WANT TO SAVE OUT ALMOST EVERYTHING (MOST SETTINGS, OUTER MODELS, TRAINING/TUNING CURVES FOR OUTER 
        FOLDS, BEST INNER FOLD PERFORMANCE). EDIT DICTIONARY AT THE END OF THE SCRIPT IF YOU DON'T. NOTABLY, WE DO NOT 
        SAVE THE SCHEDULER.

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
gpu = True  # If false will avoid using gpu though it may be available
para = False  # Should we parallelize? Yes if True. Good especially for CPU
nm = 0  # Number of workers to use for dataloader
ldCV = []  # path to file that will be used to load in random seeds to make folds (e.g., 'FirstTry_CV_Rep_1.pkl'). If empty will auto generate seeds. (file just needs to be pickled. Script will search for a variable called sds which should contain random seeds. Will also make reps equal to sds length)
kfo = 6  # number of outer folds for testing
kfi = 8  # number of inner folds for tuning
valH = 0.3  # validation holdout for final retraining of CNN (i.e., after tuning we collapse inner folds and repartition training data into train and validation sets for retraining the network)
hiter = 1 # number of random search iterations over hyperparameter space
reps = 20  # repeats of entire CV scheme
inc = 1  # number of channels in network (e.g., how many modalities of images for each subject?)

# FOR HYPERPARAMETER SPACE
lr_space = [0.00001, 0.00004, 0.00008, 0.0001] # learning rates to test
adaptLR = True # If true will use CosineAnnealingWarmRestarts for scheduling learning rate over multiple cycles/restarts.
eta_min = 1e-10  # minimum learning rate we will allow scheduler to use
T_mult = 2  # 2 # Number of epochs between restarts/cycles is increased by this factor (i.e., how much slower each subsequent cycle will be)
T_0 = 25  # affects number of iterations. T_0 is multiplied by the length of the dataloader object (i.e., # of mini batches in training data) to determine the total number of iterations.

drp_space = [0.8, 0.7, 0.6, 0.4] # drop rate frequencies to evaluate inside network (only applicable for VGG at the moment)
l2_space = [0.001, 0.01] # l2 norm penalties to add to optimizer
dpth_space = [2, 3] # network depths to test (see relevant network code in ./models; for vgg and resnet we use relative complexity but for other networks this value should be set to the number of layers in the network)
network_space = ['vgg']  # which networks to evaluate? #['vgg', 'cnn', 'C3DNet', 'resnet', 'ResNetV2', 'ResNeXt', 'ResNeXtV2', 'WideResNet', 'PreActResNet','EfficientNet', 'DenseNet', 'ShuffleNet', 'ShuffleNetV2', 'SqueezeNet', 'MobileNet', 'MobileNetV2']

# FOR TRAINING
num_epochs = 500 # number of epochs for tuning (inner folds training)
num_epochs_test = 600 # number of epochs for testing (outer folds training)
mbs = 128  # mini batch size; larger is better for GPU
sT = 'standard'  # can be standard or balanced -- this is for mini batch 'stratification'. If balanced, no loss function weighting and instead we bootstrap samples for balance in the dataloader. If standard then loss function weighting. If stratification, mini batches will simply be balanced by class (this last option is still in development, do not use)
priorityMetricTune = 'F1'  # should we tune model based on best loss for hyperparameter sets or best F1?
priorityMetricTest = 'loss'  # should we retain model based on best loss or best F1 for final test data (i.e., which model to pluck during training: best based on F1 or loss for validation data?)
tuneMetricCeil = 1  # what should the max metric be while tuning (i.e., if a hyperparameter set is higher than this value we will ignore it because it's probably overfitting). Will need to set to 0 for loss if no ceiling (i.e., technically a floor for loss)

# FOR STOPPING
perfThresh = 1  # threshold for how many 100% accuracy epochs can occur in a row before training termination (accuracy for training data in inner folds)
patThresh = 50  # validation patience threshold. After this many epochs, if loss and/or F1 has not improved, we terminate. (also for training data in inner folds)
perfThreshTest = 1 # same as perfThresh above but applied only to testing data (i.e., when retraining the model for final outer loop predictions)
patThreshTest = 90 # same as patThresh above but applied only to testing data (i.e., when retraining the model for final outer loop predictions)
perfThreshVal = 1 # value that perfThresh and perfThreshTest looks for (i.e., it can be something other than 100%)
patMetric = 'loss'  # should patience threshold apply only to 'loss', 'f1', or 'both'
corPat = True  # should we independently track correlation between train/val loss and use that as a stopping criteria? (applies to train and test data)
ws = 30  # sliding window for computing correlation over epochs
cThresh = -0.35  # value below which correlation is flagged as being too low
cNum = 60  # number of times correlation has to be below cThresh to trigger stopping
minMetric = 0.72  # min F1 value model must hit before stop conditions are triggered (correlation stopping is independent) (applies to train and test data)
maxMetric = 0.8 # max F1 score after which we immediately stop training (i.e., to avoid any overfitting) (applies to train and test data)

# FOR FEEDBACK/OUTPUT
verbose = 1  # 1 is give me the important bits, 2 is give me everything as printed text, 3 is give me most things as text and plot inner training instead of printing, 0 is give me nothing (only applies to inner fold loop)
verboseTest = 1  # same as above but only applies for outer fold loop

oF = 'FirstTry_CV.pkl'  # save the outputs to this directory/file
# END USER-DEFINED VARIABLES

if verbose >= 3 or verboseTest >= 3:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('tkagg')
if len(lr_space) * len(drp_space) * len(l2_space) * len(dpth_space) * len(
        network_space) < hiter:  # check if you have just one parameter set you're passing in
    hiter = len(lr_space) * len(drp_space) * len(l2_space) * len(dpth_space) * len(network_space)
    print('WARNING: you have selected to perform more hyperparameter searches than possible...fixing to max possible')

"""""""""""""""""""""""""""""
IMPORT DATA AND PREPROCESS
"""""""""""""""""""""""""""""
# START USER-DEFINED VARIABLES
# load data -- !INSERT YOUR OWN CODE HERE!
inmat = loadmat('PyTorchInput_withFolds.mat')  # this is my data, load yours however
yT = inmat['wabClassi'].flatten()  # put your classes in a variable called yT. Here I flatten it, you may not have to. Check if your data has more than one dimension. It shouldn't.
yF = inmat['wabClass2i'].flatten()  # for your data, make this identical to yT like the above line 130. This is the vector on which stratitfication will be based. I was experimenting with using more granular classes for stratification while still using yT for testing
x = inmat['data']  # this is a 4D predictor matrix. It needs to be shaped as such: (image dimension 1, image dimension 2, image dimension 3, subjects)

x = np.delete(x,[35,37,73,74,160],axis=3) # here i am deleting some problem subjects manually for my own data(mystery reslicing issue)
yF = np.delete(yF,[35,37,73,74,160],axis=0)
yT = np.delete(yT,[35,37,73,74,160],axis=0)
del (inmat)

# reshape input data and rescale if necessary -- DO NOT ALTER UNLESS YOU HAVE MULTIPLE CHANNELS
xr = np.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

# this is some code peculiar to my data that you should delete
lsz = np.sum(xr == 4, axis=0) # stratify by lesion type as well so compute lesion size
dc = pandas.qcut(lsz,q=3,labels=False) # convert continuous size measures to quartiles
yC = yF * 10 + dc # combine lesion size and categories in yF to produce one larger set of categories
un = np.unique(yC, return_counts=True) # get category counts
id = yC==11 # the below categories have fewer than the number of inner folds we have, so we have to merge them
yC[id]=12
id = yC==30
yC[id]=31
id = yC==40
yC[id]=42
id = yC==41
yC[id]=42
yF = yC # make yF these new categories (this is the variable startification will be based on)

# do not delete. These lines this is code for preprocessing xr. Here we rescale -1 to 1 each subject's data independently
sc = MinMaxScaler(feature_range=(-1,1))
sc.fit(xr)
xr = sc.transform(xr)
#xr = xr / xr.max()

# END USER-DEFINED VARIABLES
"""""""""""""""""""""""""""""
CUSTOM SAMPLER FOR STRATIFICATION OF MINI BATCHES (NOT FULLY FIXED YET)
"""""""""""""""""""""""""""""
# this requires some debugging...
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
def torchDataExtractor(id, xr, y, x, mbs, nm, sampType='stratified', shuf=None):
    """
    EXTRACT INDICES FROM RESHAPED BRAIN DATA (TO 2D MATRIX), CONVERT THE RESULT BACK TO 4D MATRIX AND PREPROCESS MATRIX FOR DNN IN PYTORCH
        - I.E., DO SOME MATRIX RESHAPING AND SETUP TENSORDATASET, SAMPLER, AND DATALOADER
        - ASSUMES IN_CHANNELS IS 1
        - nm is number of workers for data loader
        - shuf determines whether to shuffle data within data loader

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
        cw = cw.view(1, -1)
        sw = cw[:, tmp3].squeeze()
        samp = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
        if shuf == 'none':
            shuf = False
        drp = False

    elif sampType == 'stratified':
        # this implements a sampler with stratification but no class imblanace
        samp = stratSamp(tmp2, mbs, drop_last=True)
        if shuf == 'none':
            shuf = False
        drp = True

    elif sampType == 'standard':
        samp = None
        if shuf == 'none':
            shuf = True
        drp = False
    o = DataLoader(TensorDataset(tmp.float(), tmp2.float()), batch_size=mbs, sampler=samp, shuffle=shuf, drop_last=drp,
                   num_workers=nm, pin_memory=True)
    return o


"""""""""""""""""""""""""""""
PERFORMANCE EVALUATOR
"""""""""""""""""""""""""""""

def getPerf(tei, model, optimizer, scheduler, lossFn=None, model_state=None, optimizer_state=None, scheduler_state=None,
            multiClass=False):
    """
    tei is a test dataset, model is a CNN model, lossFn is a prespecified loss function, model_state is a model state,
    etc. Set mutliClass to true to evaluate macro F1, accuracy, etc across classes.

    Returns accuracy, loss, precision, recall, and F1 score. Also returns predictions and actual labels
    (since you may be shuffling them in the dataloader)

    """
    if model_state is not None:
        model.load_state_dict(model_state)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    if scheduler_state is not None and scheduler is not None:
        scheduler.load_state_dict(scheduler_state)

    model.eval()
    loss = 0
    yh = []
    y = []
    yhp = []
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
            if pred.numel() == 1:  # if batch is 1, yh will not work with tolist as its a tensor scalar that may also be just a 0 (class)
                yh.append(pred)
                yhp.append(prob)
            else:
                yh.extend(pred.tolist())
                yhp.append(prob)

            # yh.extend(pred.tolist())
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

    return loss, f1, rec, prec, acc, yh, y, yhp


"""""""""""""""""""""""""""""
MODEL TRAINER
"""""""""""""""""""""""""""""

def dnnTrainer(tri, tei, inc, lossFn, num_epochs, lr, drp, l2, dpth, ws, cThresh, cNum, minMetric, maxMetric, patThresh,
               perfThresh, perfThreshVal, verbose, patMetric, eta_min, T_0, T_mult, adaptLR=True, para=False, cf=True,
               title=None):
    """
    tri is a training dataset, tei is a test dataset, lossFn is a loss function, etc...see all of these parameters in
    the parameter section for more info.

    If cf is true, it will close the figure (if you're plotting performance across epochs)
    If a title is passed in figures will be given the title passed in
    (to keep track of e.g., outer folds, repeats, search iterations, etc)
    """
    # hard coded variables
    # beta1 = 0.9
    bestF1v = 0  # winning F1 score
    bestLossv = 10000000000000000  # winning loss
    lossPat = 0  # counter for loss increasing
    f1Pat = 0  # counter for f1 decreasing
    perf = 0  # counter for 100 percent accuracy
    lid = 0
    fid = 0
    cc = 0

    # temporary tracking of performance metrics within epoch
    f1vm = np.zeros(num_epochs)
    f1tm = np.zeros(num_epochs)
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
    model = getDNN(cnn_name='vgg', model_depth=dpth, n_classes=1, in_channels=inc, sample_size=len(tri.dataset),
                   drop=drp, cuda=gpu)
    if para:
        model = DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=l2, amsgrad=True,
                                  eps=1e-08)

    if adaptLR:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0 * len(tri), T_mult=T_mult, eta_min=eta_min)  # T_max=T_max, eta_min=eta_min)

    # track model state
    wmodel = None
    wmodel2 = None
    wmodelo = None
    wmodel2o = None
    wmodels = None
    wmodel2s = None
    if verbose >= 3:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ic = 0
    for e in range(num_epochs):
        if verbose >= 4:
            print('-----Epoch:', e + 1)
        # train model...
        model.train()

        for batch_idx, (data, target) in enumerate(tri):
            if next(model.parameters()).is_cuda:  # we already check for cuda when building the model, now just carry over data if necessary
                data = data.to('cuda')
            optimizer.zero_grad()
            output = model(data).squeeze()
            if next(model.parameters()).is_cuda:
                output = output.to('cpu')

            loss = lossFn(output, target)
            loss.backward()
            optimizer.step()

            if adaptLR:
                scheduler.step(ic)
                ic += 1

        if adaptLR and verbose >= 4:
            currentlr = scheduler.get_last_lr()[0]
            print(f"LR is: {currentlr}")

        # now eval model...
        losstm[e], f1tm[e], _, _, acctm[e], _, _, _ = getPerf(tri, model, optimizer, scheduler, lossFn,
                                                              model_state=None, optimizer_state=None,
                                                              scheduler_state=None, multiClass=False)
        lossvm[e], f1vm[e], _, _, accvm[e], _, _, _ = getPerf(tei, model, optimizer, scheduler, lossFn,
                                                              model_state=None, optimizer_state=None,
                                                              scheduler_state=None, multiClass=False)

        if f1vm[e] > bestF1v + 0.05 and f1vm[e] <= maxMetric:
            bestF1v = f1vm[e]
            f1Pat = 0
            wmodel = copy.deepcopy(model.state_dict())
            wmodelo = copy.deepcopy(optimizer.state_dict())
            wmodels = copy.deepcopy(scheduler.state_dict())
            fid = e
        else:
            f1Pat += 1

        if lossvm[e] < bestLossv and f1vm[e] <= maxMetric:
            bestLossv = lossvm[e]
            wmodel2 = copy.deepcopy(model.state_dict())
            wmodel2o = copy.deepcopy(optimizer.state_dict())
            wmodel2s = copy.deepcopy(scheduler.state_dict())
            lossPat = 0
            lid = e
        else:
            lossPat += 1

        if verbose == 2:
            print('Validation set-- BEST F1: {:.4f}, Average loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}'.format(
                bestF1v, lossvm[e], accvm[e], f1vm[e]))
            print('Training set-- Loss: {:.4f}, Accuracy: {:.4f}'.format(
                losstm[e], acctm[e]))
        elif verbose >= 3:
            plt.rcParams.update({'font.size': 12, 'axes.titlesize': 16})
            ax1.spines['bottom'].set_linewidth(4)
            ax1.spines['top'].set_linewidth(4)
            ax1.spines['left'].set_linewidth(4)
            ax1.spines['right'].set_linewidth(4)
            ax1.clear()
            ax1.plot(range(1, e + 2), losstm[:e + 1], color=(0.415686, 0.172549, 0.439216), label='Training Loss',
                     linewidth=3.5)
            ax1.plot(range(1, e + 2), lossvm[:e + 1], color=(0.721569, 0.231373, 0.368627), label='Validation Loss',
                     linewidth=6)
            ax1.grid(axis='y', linewidth=2, linestyle='--')
            ax1.scatter(lid + 1, bestLossv, s=250, c='r', marker='o', zorder=3)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()

            ax2.clear()
            ax2.spines['bottom'].set_linewidth(4)
            ax2.spines['top'].set_linewidth(4)
            ax2.spines['left'].set_linewidth(4)
            ax2.spines['right'].set_linewidth(4)
            ax2.plot(range(1, e + 2), acctm[:e + 1], color=(0.415686, 0.172549, 0.439216), label='Training Accuracy',
                     linewidth=3.5)
            ax2.plot(range(1, e + 2), accvm[:e + 1], color=(0.721569, 0.231373, 0.368627), label='Validation Accuracy',
                     linewidth=3.5)
            ax2.plot(range(1, e + 2), f1vm[:e + 1], color=(0.976471, 0.929412, 0.411765), label='Validation F1 Score',
                     linewidth=6)
            ax2.grid(axis='y', linewidth=2, linestyle='--')
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

        if e >= ws - 1:
            # cr = np.corrcoef(losstm[e-ws+1:e], lossvm[e-ws+1:e])
            cr = np.corrcoef(acctm[e - ws + 1:e], accvm[e - ws + 1:e])
            if cr[0, 1] < cThresh:
                cc += 1
            else:
                cc = 0
        if patMetric == 'loss':
            if lossPat >= patThresh and bestF1v > minMetric:  # f1Pat >= patThresh or
                if verbose >= 2:
                    print(
                        f"Early stopping on epoch {e} triggered: Validation F1 score or loss did not improve for {patThresh} consecutive epochs.")
                break
            if corPat and cc > cNum:  # and bestF1v > minMetric:
                if verbose >= 2:
                    print(
                        f"Early stopping on epoch {e} triggered: Correlation was below {cThresh} for {cNum} consecutive epochs.")
                break

        elif patMetric == 'F1':
            if f1Pat >= patThresh and bestF1v > minMetric:
                if verbose >= 2:
                    print(
                        f"Early stopping on epoch {e} triggered: Validation F1 score or loss did not improve for {patThresh} consecutive epochs.")
                break
            if corPat and cc > cNum:  # and bestF1v > minMetric:
                if verbose >= 2:
                    print(
                        f"Early stopping on epoch {e} triggered: Correlation was below {cThresh} for {cNum} consecutive epochs.")
                break

        elif patMetric == 'both':
            if lossPat >= patThresh or f1Pat >= patThresh and bestF1v > minMetric:
                if verbose >= 2:
                    print(
                        f"Early stopping on epoch {e} triggered: Validation F1 score or loss did not improve for {patThresh} consecutive epochs.")
                break
            if corPat and cc > cNum and bestF1v > minMetric:
                if verbose >= 2:
                    print(
                        f"Early stopping on epoch {e} triggered: Correlation was below {cThresh} for {cNum} consecutive epochs.")
                break
        if perf >= perfThresh:  # and bestF1v > minMetric:
            if verbose >= 2:
                print(
                    f"Early stopping on epoch {e} triggered: Number of times training accuracy hit {perfThreshVal * 100} % exceeded: {perfThresh} ")
            break
        if corPat and cc > cNum:  # and bestF1v > minMetric:
            if verbose >= 2:
                print(
                    f"Early stopping on epoch {e} triggered: Correlation was below {cThresh} for {cNum} consecutive epochs.")
            break

    if verbose >= 3 and cf:
        plt.close(fig)

    return f1vm, lossvm, accvm, acctm, losstm, bestF1v, bestLossv, wmodel, wmodel2, model, wmodelo, wmodel2o, optimizer, scheduler, wmodel2s, wmodels


"""""""""""""""""""""""""""""
START CV
"""""""""""""""""""""""""""""
# only variables we keep across all reps
f1f = np.zeros(reps)
accf = np.zeros(reps)
aucf = np.zeros(reps)

# we will use random seed integers to preallocate folds...
if isinstance(ldCV, str):
    with open(ldCV, 'rb') as f:
        sds = pickle.load(f)['sds']
        reps = len(sds)
else:
    sds = [random.randint(0, 4294967295) for _ in range(reps)]  # that number is the max possible allowed by random

for r in range(reps):
    # first initialize data trackers that get saved for each repeat
    hypeTrackerF1o2 = []
    hypeTrackerLosso2 = []
    valF1Trackero2 = []
    testF1Trackero2 = []
    trainF1Trackero2 = []
    modelTracker = []
    modelStateTracker = []
    modelStateTracker2 = []
    optimTracker = []
    optimStateTracker = []
    optimStateTracker2 = []
    schedStateTracker = []
    schedStateTracker2 = []

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
    finpredTrackerYp = []
    finpredTrackervYh = []
    finpredTrackervY = []

    if verbose >= 1:
        print(
            '-------------------------------------------------------------------------------------------------------Repeat is:',
            r + 1)

    xr = np.transpose(xr)  # need to transpose for sklearn
    outer_cv = StratifiedKFold(n_splits=kfo, shuffle=True, random_state=sds[r])
    inner_cv = StratifiedKFold(n_splits=kfi, shuffle=True, random_state=sds[r])

    # create inner/outer folds
    nested_indices = []
    for outer_train_idx, outer_test_idx in outer_cv.split(xr, yF):
        inner_indices = []
        for inner_train_idx, inner_val_idx in inner_cv.split(xr[outer_train_idx, :], yF[outer_train_idx]):
            inner_train_idx_global = [outer_train_idx[idx] for idx in inner_train_idx]
            inner_val_idx_global = [outer_train_idx[idx] for idx in inner_val_idx]
            inner_indices.append((inner_train_idx_global, inner_val_idx_global))
        nested_indices.append((outer_train_idx, outer_test_idx, inner_indices))

    # Create new train/validation sets from the outer training for retraining the CNN after tuning
    outer_train_val = []
    for outer_train_idx, outer_test_idx, inner_indices in nested_indices:
        new_train_idx, val_idx = train_test_split(outer_train_idx, test_size=valH, stratify=yF[outer_train_idx],
                                                  shuffle=True, random_state=sds[r])
        outer_train_val.append((new_train_idx, val_idx))
    xr = np.transpose(xr)  # at this point xr should be features x samples

    for o in range(kfo):
        best_score_F1 = 0  # track best score you get during tuning FOR F1
        best_score_loss = 1000000000000000000000000000000  # track best score you get during tuning
        best_hyperparams_F1 = None  # track best hyperparams you get when tuning
        best_hyperparams_loss = None  # track best hyperparams you get when tuning
        hyperparameters_chosen = []  # track hyperparams so we do not choose repeats in random search
        if verbose >= 1:
            print(
                f"----------------------------------------------------------------------Outer fold is {o + 1} of {kfo}")

        if len(lr_space) * len(drp_space) * len(l2_space) * len(dpth_space) * len(
                network_space) > 1:  # check if you have just one parameter set you're passing in
            for h in range(hiter):
                if verbose >= 1:
                    print(f"---------------------------------------Random subset is: {h + 1} of {hiter}")

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
                        print(f"----------------Inner fold is: {i + 1}  of {kfi}")
                    if sT == 'standard':
                        # weighted loss function
                        tmp = np.bincount(yT[nested_indices[o][2][i][0]])  # get inverse class weights for loss function
                        w = len(nested_indices[o][2][i][0]) / tmp.astype(float)
                        lossFn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w[1]))
                    elif sT == 'balanced':
                        # no weights because we will bootstrap the undersampled class
                        lossFn = nn.BCEWithLogitsLoss()

                    tri = torchDataExtractor(nested_indices[o][2][i][0], xr, yT, x.shape, mbs, nm, sampType=sT, shuf=None)
                    tei = torchDataExtractor(nested_indices[o][2][i][1], xr, yT, x.shape, mbs, nm,
                                             sampType='standard', shuf=False)  # this has to be standard because you should not be bootstrapping validation data. makes no sense.

                    # train the model
                    f1vm, lossvm, accvm, acctm, losstm, bestF1v, bestLossv_val, _, _, _, _, _, _, _, _, _ = dnnTrainer(
                        tri, tei, inc, lossFn, num_epochs, lr, drp, l2, dpth, ws, cThresh, cNum, minMetric, 1,
                        patThresh, perfThresh, perfThreshVal, verbose, patMetric, eta_min, T_0, T_mult, adaptLR=adaptLR,
                        para=para, cf=True,
                        title=f"Repeat {r + 1} / {reps}. Test fold: {o + 1} / {kfo}, Inner fold: {i + 1} / {kfi} and random search: {h + 1} / {hiter}")

                    avg_inner_score_loss += bestLossv_val / (
                            kfi - 1)  # average score for hyperparameter tuning within inner fold CHANGE TO bestF1v to select based on f1 and bestLossv_val to select based on loss
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
                        print(
                            f"******************* Found better F1 score {avg_inner_score_F1} but it's above ceiling:{tuneMetricCeil}")

                if avg_inner_score_loss < best_score_loss:
                    if (
                            priorityMetricTune == 'loss' and avg_inner_score_loss > tuneMetricCeil) or priorityMetricTune != 'loss':
                        best_score_loss = avg_inner_score_loss
                        best_hyperparams_loss = (lr, drp, l2, dpth, netType)
                        if verbose >= 1:
                            print('******************* Best loss is now updated to :', avg_inner_score_loss)
                            print(f"******************* HYPERPARAMETERS: {hyperparams}")
                    else:
                        print(
                            f"******************* Found better loss {avg_inner_score_F1} but it's below floor:{tuneMetricCeil}")
        else:
            lr = random.choice(lr_space)
            drp = random.choice(drp_space)
            l2 = random.choice(l2_space)
            dpth = random.choice(dpth_space)
            netType = random.choice(network_space)

        # get best hyperparams for the outer fold
        if priorityMetricTune == 'F1':
            if best_hyperparams_F1 is not None:  # we need this in case you are doing something dumb (e.g., while debugging) and the best score is as bad as what we initialized (i.e., worst possible score)
                lr, drp, l2, dpth, netType = best_hyperparams_F1
        elif priorityMetricTune == 'loss':
            if best_hyperparams_loss is not None:  # we need this in case you are doing something dumb (e.g., while debugging) and the best score is as bad as what we initialized (i.e., worst possible score)
                lr, drp, l2, dpth, netType = best_hyperparams_loss

        tri = torchDataExtractor(outer_train_val[o][0], xr, yT, x.shape, mbs, nm, sampType=sT, shuf=None)
        vali = torchDataExtractor(outer_train_val[o][1], xr, yT, x.shape, mbs, nm,
                                  sampType='standard', shuf=False)  # again, would make no sense to bootstrap undersampled class here
        tei = torchDataExtractor(nested_indices[o][1], xr, yT, x.shape, mbs, nm,
                                 sampType='standard', shuf=False)  # same...don't touch the test data

        if sT == 'standard':
            tmp = np.bincount(yT[outer_train_val[o][
                0]])  # get inverse class weights for loss function...again (outer folds can be slightly different)
            w = len(outer_train_val[o][0]) / tmp.astype(float)
            lossFn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w[1]))
        elif sT == 'balanced':
            lossFn = nn.BCEWithLogitsLoss()

        # retrain best network and get its performance
        _, _, _, _, _, bestF1_val, bestLossv_val, wmodel, wmodel2, model, wmodelo, wmodel2o, optimizer, scheduler, wmodel2s, wmodels = dnnTrainer(
            tri, vali, inc, lossFn, num_epochs_test, lr, drp, l2, dpth, ws, cThresh, cNum, minMetric, maxMetric,
            patThreshTest, perfThreshTest, perfThreshVal, verboseTest, patMetric, eta_min, T_0, T_mult, adaptLR=adaptLR,
            para=para, cf=False, title=f"Test fold: {o + 1} / {kfo} for repeat {r + 1} / {reps}")

        if priorityMetricTest == 'F1':
            lossval, f1val, recval, precval, accval, yhval, ysval, yhpval = getPerf(vali, model, optimizer, scheduler,
                                                                                    lossFn, model_state=wmodel,
                                                                                    optimizer_state=wmodelo,
                                                                                    scheduler_state=wmodels,
                                                                                    multiClass=False)  # change this to wmodel to select best model based on F1, change this to wmodel2 to select best model based on loss
            loss, f1, rec, prec, acc, yh, ys, yp = getPerf(tei, model, optimizer, scheduler, lossFn, model_state=wmodel,
                                                           optimizer_state=wmodelo, scheduler_state=wmodels,
                                                           multiClass=False)  # change this to wmodel to select best model based on F1, change this to wmodel2 to select best model based on loss
            if verbose >= 1:
                print(
                    f"Inner validation F1: {best_score_F1}, Outer validation F1: {f1val}, Test F1: {f1}, Outer validation loss: {lossval}, Test loss: {loss}")
        elif priorityMetricTest == 'loss':
            lossval, f1val, recval, precval, accval, yhval, ysval, yhpval = getPerf(vali, model, optimizer, scheduler,
                                                                                    lossFn, model_state=wmodel2,
                                                                                    optimizer_state=wmodel2o,
                                                                                    scheduler_state=wmodel2s,
                                                                                    multiClass=False)  # change this to wmodel to select best model based on F1, change this to wmodel2 to select best model based on loss
            loss, f1, rec, prec, acc, yh, ys, yp = getPerf(tei, model, optimizer, scheduler, lossFn,
                                                           model_state=wmodel2, optimizer_state=wmodel2o,
                                                           scheduler_state=wmodel2s,
                                                           multiClass=False)  # change this to wmodel to select best model based on F1, change this to wmodel2 to select best model based on loss
            if verbose >= 1:
                print(
                    f"Inner validation loss: {best_score_loss}, Outer validation loss: {lossval}, Test loss: {loss}, Outer validation F1: {f1val}, Test f1: {f1}")

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

        modelStateTracker.append(wmodel)
        modelStateTracker2.append(wmodel2)
        modelTracker.append(model)
        optimTracker.append(optimizer)
        optimStateTracker.append(wmodelo)
        optimStateTracker2.append(wmodel2o)
        schedStateTracker.append(wmodels)
        schedStateTracker2.append(wmodel2s)

        finpredTrackerYh.append(yh)
        finpredTrackerY.append(ys)
        finpredTrackerYp.append(yp)

        finpredTrackervYh.append(yhval)
        finpredTrackervY.append(ysval)

    ysC = [item for lst in finpredTrackerY[:] for item in lst]
    yhC = [item for lst in finpredTrackerYh[:] for item in lst]
    f1f[r] = f1_score(ysC, yhC, pos_label=1)
    accf[r] = accuracy_score(ysC, yhC)
    aucf[r] = roc_auc_score(ysC, yhC)

    print(f"----> FINAL FOR ENTIRE REP, F1: {f1f[r]}, ACCURACY: {accf[r]}, AUC: {aucf[r]}")

    # save outputs on each repeat to be safe...
    dd = {
        'trainLossTrackero2': trainLossTrackero2,
        'vallossTrackero2': vallossTrackero2,
        'testlossTrackero2': testlossTrackero2,
        'valtrlossALLTrackero2': valtrlossALLTrackero2,
        'valtelossALLTrackero2': valtelossALLTrackero2,
        'valteF1ALLTrackero2': valteF1ALLTrackero2,
        'valtrACCALLTrackero2': valtrACCALLTrackero2,
        'valteACCALLTrackero2': valteACCALLTrackero2,
        'hypeTrackerF1o2': hypeTrackerF1o2,
        'hypeTrackerLosso2': hypeTrackerLosso2,
        'valF1Trackero2': valF1Trackero2,
        'testF1Trackero2': testF1Trackero2,
        'trainF1Trackero2': trainF1Trackero2,
        'valaccTrackero2': valaccTrackero2,
        'testaccTrackero2': testaccTrackero2,
        'modelStateTracker': modelStateTracker,
        'modelStateTracker2': modelStateTracker2,
        'modelTracker': modelTracker,
        'optimTracker': optimTracker,
        'optimStateTracker': optimStateTracker,
        'optimStateTracker2': optimStateTracker2,
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
        'perfThreshVal': perfThreshVal,
        'f1f': f1f,
        'accf': accf,
        'aucf': aucf
    }
    oFf = oF[0:-4] + '_Rep_' + str(r + 1) + '.pkl'
    with open(oFf, 'wb') as f:
        print('Saving data...')
        pickle.dump(dd, f)
