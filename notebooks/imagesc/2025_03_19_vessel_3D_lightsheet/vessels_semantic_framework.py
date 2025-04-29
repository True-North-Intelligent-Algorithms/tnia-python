import numpy as np
from napari_easy_augment_batch_dl.frameworks.base_framework import BaseFramework, LoadMode
from tifffile import imread
import json
from napari_easy_augment_batch_dl.frameworks.pytorch_semantic_dataset import PyTorchSemanticDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from monai.networks.nets import BasicUNet, UNet
import torch
from tnia.deeplearning.dl_helper import quantile_normalization
import os
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from glob import glob
from monai.inferers import sliding_window_inference
from semantic_helper import train

@dataclass
class VesselsSemanticFramework(BaseFramework):
    """
    Pytorch Semantic Framework
    
    This framework is used to train a Pytorch Semantic Segmentation model.

    """
    show_background_class: bool = field(default=True, metadata={'type': 'bool', 'harvest': True, 'advanced': False, 'training': False, 'default': True})
    tile_size: int = field(default=1024, metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 128, 'max': 100000, 'default': 1024, 'step': 1})
    
    sparse: bool = field(default=True, metadata={'type': 'bool', 'harvest': True, 'advanced': False, 'training': True, 'default': True})
    num_classes: int = field(default=2, metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 1, 'max': 10, 'default': 2, 'step': 1, 'show_auto_checkbox':True})
    depth: int = field(default=5,metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 4, 'max': 6, 'default': 5, 'step': 1})
    features_level_1: int = field(default=32, metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 8, 'max': 64, 'default': 32, 'step': 1})
    
    weight_c1: int = field(default=1, metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 1, 'max': 100, 'default': 1, 'step': 1})
    weight_c2: int = field(default=1,metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 1, 'max': 100, 'default': 1, 'step': 1})
    weight_c3: int = field(default=1,metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 1, 'max': 100, 'default': 1, 'step': 1})
    
    num_epochs: int = field(default=100,metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 0, 'max': 100000, 'default': 100, 'step': 1})
    learning_rate: float = field(default=0.0001, metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': True, 'min': 0, 'max': 1., 'default': 0.0001, 'step': .001})
    dropout: float = field(default=0.0,metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': True, 'min': 0, 'max': 1., 'default': 0.0, 'step': .01})
    model_name: str = field(default='semantic', metadata={'type': 'str', 'harvest': True, 'advanced': False, 'training': True, 'default': 'semantic', 'step': 1})
        
    descriptor = "Vessels Semantic Model"
    
    def __init__(self, parent_path: str,  num_classes: int, start_model: str = None):
        super().__init__(parent_path, num_classes)
        
        self.model = None
        
        self.model_name = self.generate_model_name(
            base_name="model"
        )
        
        self.load_mode = LoadMode.File
        
        '''
        self.sparse = True 
        self.tile_size = 1024
        self.show_background_class = True
        self.num_epochs = 100
        self.learning_rate = 1.e-3 
        self.dropout = 0.0
        '''
        self.num_classes_auto = True

        self.model_name = self.generate_model_name('semantic_vessels')

    def generate_model_name(self, base_name="model"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_name}_{current_time}.pth"
        return model_name
        
    def create_callback(self, updater):
        self.updater = updater
    
    def train(self, updater=None):

        patch_path = Path(self.patch_path)
        
        if updater is None:
            updater = self.updater
        
        if updater is not None:
            updater('Training Pytorch Semantic model', 0)

        cuda_present = torch.cuda.is_available()
        ndevices = torch.cuda.device_count()
        use_cuda = cuda_present and ndevices > 0
        device = torch.device("cuda" if use_cuda else "cpu")  # "cuda:0" ... default device, "cuda:1" would be GPU index 1, "cuda:2" etc

        with open(patch_path / 'info.json', 'r') as json_file:
            data = json.load(json_file)
            sub_sample = data.get('sub_sample',1)
            print('sub_sample',sub_sample)
            axes = data['axes']
            print('axes',axes)
            num_inputs = data['num_inputs']
            print('num_inputs',num_inputs)
            num_truths = data['num_truths']
            print('num_truths',num_truths)

        image_patch_path = patch_path / 'input0'

        tif_files = glob(str(image_patch_path / '*.tif'))
        first_im = imread(tif_files[0])
        target_shape=first_im.shape

        if axes == 'YX':
            num_in_channels=1
        else:
            num_in_channels=3

        assert patch_path.exists(), f"root directory with images and masks {patch_path} does not exist"

        X = sorted(patch_path.rglob('**/input0/*.tif'))

        Y = []
        for i in range(num_truths):
            Y.append(sorted(patch_path.rglob(f'**/ground truth{i}/*.tif')))

        train_data = PyTorchSemanticDataset(
            image_files=X,
            label_files_list=Y,
            target_shape=target_shape
        )

        # NOTE: the length of the dataset might not be the same as n_samples
        #       because files not having the target shape will be discarded
        print(len(train_data))

        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

        if self.num_classes_auto == True:
            if self.sparse:
                # if sparse background will be label 1 so number of classes is the max label indexes
                # ie if the max label index is 3 then there are 3 classes, 1, 2, 3 and 0 is unlabeled
                # (we subtract 1 at later step so 1 (background) becomes 0 and 0 (not labeled) becomes -1)
                self.num_classes = train_data.max_label_index
            else:
                # if not sparse background will be label 0 so number of classes is the max label indexes + 1
                # ie if there are 3 classes the indexes are 0, 1, 2, so need to add 1 to the max index to get number of classes
                self.num_classes = train_data.max_label_index+1

        # there is an inconstency in how different classes can be defined
        # 1. every class has it's own label image (one-hot encoded)
        # 2. every class has a unique value in the label image
        # When I wrote a lot of this code I was thinking of the first case, but now see the second may be easier for the user
        # so number of output channels is the max of the truth image
        # use monai to create a model, note we don't use an activation function because 
        # we use CrossEntropyLoss that includes a softmax, and our prediction will include the softmax
        if self.model == None:
            
            channels = tuple(self.features_level_1 * (2 ** (i-1) if i > 1 else 1) for i in range(self.depth+1))
            strides = tuple(2 for i in range(self.depth))
            #channels = (self.features_level_1, self.features)
            
            # Example 2: A deeper UNet
            self.model = UNet(
                spatial_dims=2,
                in_channels=num_in_channels,
                out_channels=self.num_classes,
                channels=channels,  #
                strides=strides,
                num_res_units=2,  # BasicUNet has no residual blocks
                act=("LeakyReLU", {"negative_slope": 0.01, "inplace": True}),
                norm="batch",
                dropout=self.dropout,
            )
        
        self.model = self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train(True)

        weights = torch.ones(self.num_classes, dtype=torch.float32)
        weights[0] = self.weight_c1
        weights[1] = self.weight_c2

        if self.num_classes > 2:
            weights[2] = self.weight_c3

        weights = weights*0.01

        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weights).to(device)
        dtype = torch.LongTensor

        train(train_loader, self.model, loss_function, optimizer, dtype, self.num_epochs, device, sparse=self.sparse)
        
        torch.save(self.model, Path(self.model_path) / self.model_name)

    def predict(self, image):
        device = torch.device("cuda")
        self.model.to(device)
        
        image_ = quantile_normalization(image.astype(np.float32))

        # move channel position to first axis if data has channel
        if len(image_.shape) == 3:
            features = image_.transpose(2,0,1)
        else:
            # add trivial channel axis
            features = np.expand_dims(image_, axis=0)
        
        # make into tensor and add trivial batch dimension 
        x = torch.from_numpy(features).unsqueeze(0).to(device)       
        
        # move into evaluation mode
        self.model.eval()

        with torch.no_grad():
            # perform sliding window inference to avoid running out of memory on smaller GPUS
            y = sliding_window_inference(
                x,                      # Input tensor
                (self.tile_size,self.tile_size),              # Patch size
                1,                       # Batch size during inference
                self.model,              # Model for inference
                mode='gaussian',        # Inference mode
                overlap=0.125          # Overlap factor
            )

        # Apply softmax along the class dimension (dim=1)
        probabilities = F.softmax(y, dim=1)
        # now predicted classes are max of probabilities along the class dimension
        predicted_classes = torch.argmax(probabilities, dim=1)

        if self.show_background_class == False:
            predicted_classes = predicted_classes - 1
             
        return (predicted_classes.cpu().detach().numpy().squeeze()+1)

    def load_model_from_disk(self, model_name):
        self.model = torch.load(model_name, weights_only=False)
        base_name = os.path.basename(model_name)
        self.model_dictionary[base_name] = self.model

# this line is needed to register the framework on import
BaseFramework.register_framework('VesselsSemanticFramework', VesselsSemanticFramework)


