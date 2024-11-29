from napari_easy_augment_batch_dl.base_model import BaseModel, LoadMode
import numpy as np
from tnia.deeplearning.dl_helper import collect_training_data
from cellpose import models, io
from dataclasses import dataclass, field
from tnia.deeplearning.dl_helper import quantile_normalization
import os

@dataclass
class AltCellPoseInstanceModel(BaseModel):
    
    # below are the parameters that are harvested for automatic GUI generation

    # first set of parameters have advanced False and training False and will be shown in the main dialog
    diameter363: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': 0.0, 'max': 500.0, 'default': 30.0, 'step': 1.0})
    prob_thresh: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': -10.0, 'max': 10.0, 'default': 0.0, 'step': 0.1})
    flow_thresh: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': -10.0, 'max': 10.0, 'default': 0.0, 'step': 0.1})
    chan_segment: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 0, 'max': 100, 'default': 0, 'step': 1})
    chan2: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 0, 'max': 100, 'default': 0, 'step': 1})
    niter: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 0, 'max': 100000, 'default': 200, 'step': 1})

    # second set of parameters have advanced True and training False and will be shown in the advanced popup dialog

    # third set of parameters have advanced False and training True and will be shown in the training popup dialog
    num_epochs: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 0, 'max': 100000, 'default': 100, 'step': 1})
    model_name: str = field(metadata={'type': 'str', 'harvest': True, 'advanced': False, 'training': True, 'default': 'cyto3', 'step': 1})



    def __init__(self, patch_path: str, model_path: str,  num_classes: int, start_model: str = None):
        super().__init__(patch_path, model_path, num_classes)

        # start logger (to see training across epochs)
        logger = io.logger_setup()

        # if no start model passed in, set model to none and wait until user selects a model 
        if start_model is None:
            self.model = None
        # if model passed in and is type 'Cellpose' set the model
        elif type(start_model) == models.Cellpose:
            self.model = start_model
        # otherwise if path passed in load model
        else:
            self.model = models.CellposeModel(gpu=True, model_type=None, pretrained_model=start_model)

        # set defaults for parameters
        self.diameter363 = 30
        self.prob_thresh = 0.0
        self.flow_thresh = 0.4
        self.chan_segment = 0
        self.chan2 = 0
        self.niter = 200
        
        self.descriptor = "CellPose Alt Model"
        self.load_mode = LoadMode.File
        
        self.num_epochs = 100
        self.model_name = self.generate_model_name('cellpose')

        self.sgd = False
    
        # fourth set of parameters are options that will be shown in combo boxes
        
        # initial model names
        self.model_names = ['notset', 'cyto3', 'tissuenet_cp3']
        
        # pretrained model names
        self.builtin_names = ['cyto3', 'tissuenet_cp3']
        
        # options for optimizers
        self.optimizers = ['adam', 'sgd']
        
        # we also have the normalizaton parameters
        self.quantile_low = 0.01
        self.quantile_high = 0.998
    
    def train(self, num_epochs, updater=None):
        add_trivial_channel = False
        X, Y = collect_training_data(self.patch_path, sub_sample=1, downsample=False, normalize_input=False, add_trivial_channel = add_trivial_channel, relabel=True)

        train_percentage = 0.9

        X_ = X.copy()
        Y_ = Y.copy()

        X_train = X_[:int(len(X_)*train_percentage)]
        Y_train = Y_[:int(len(Y_)*train_percentage)]
        X_test = X_[int(len(X_)*train_percentage):]
        Y_test = Y_[int(len(Y_)*train_percentage):]

        print(X_train[0].shape)
        print(Y_train[0].shape)

        #print(help(self.model.train_seg))

        if self.model is None:
            self.model = models.CellposeModel(gpu=True, model_type=None)


        # if self.model path ends with models
        if os.path.basename(self.model_path)=='models':
            save_path = os.path.dirname(self.model_path)
        else:
            save_path = self.model_path

        from cellpose import train

        new_model_path = train.train_seg(self.model.net, X_train, Y_train, 
            test_data=X_test,
            test_labels=Y_test,
            channels=[self.chan_segment, self.chan2], 
            save_path=save_path, 
            n_epochs = self.num_epochs,
            # TODO: make below GUI options? 
            #learning_rate=learning_rate, 
            #weight_decay=weight_decay, 
            #nimg_per_epoch=200,
            model_name=self.model_name)

    def predict(self, img: np.ndarray):
        # this is a bit tricky... have to make sure normalization done during evaluation matches training
        # TODO: Continue iterating and double checking this
        img_normalized = quantile_normalization(img, quantile_low = self.quantile_low, quantile_high= self.quantile_high, channels=True).astype(np.float32)
        return self.model.eval(img_normalized, diameter=self.diameter363, normalize=False, channels=[self.chan_segment, self.chan2], flow_threshold=self.flow_thresh, cellprob_threshold=self.prob_thresh, niter=self.niter)[0]

    def get_model_names(self):
        return self.model_names 
    
    def get_optimizers(self):
        return self.optimizers 
   
    def set_builtin_model(self, model_name):
        self.model = models.CellposeModel(gpu=True, model_type=model_name)
    
    def load_model_from_disk(self, model_path):
        self.model = models.CellposeModel(gpu=True, model_type=None, pretrained_model=model_path)
        
        # model path needs to be the base of model_path that was loaded
        # (otherwise when training there will be an extra 'models' directory createed)
        self.model_path = os.path.dirname(model_path)

        base_name = os.path.basename(model_path)
        self.model_name = base_name
        self.pretrained_models[base_name] = self.model

    def set_optimizer(self, optimizer):
        self.sgd = optimizer == 'sgd'
