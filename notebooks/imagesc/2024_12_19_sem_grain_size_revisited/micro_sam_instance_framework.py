from napari_easy_augment_batch_dl.frameworks.base_framework import BaseFramework, LoadMode
from dataclasses import dataclass, field
import os
import numpy as np
import torch
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

@dataclass
class MicroSamInstanceFramework(BaseFramework):
    """
    Micro-sam Instance Framework

    This framework is used to train a Microsam Instance Segmentation model.
    """
    
    # below are the parameters that are harvested for automatic GUI generation

    # first set of parameters have advanced False and training False and will be shown in the main dialog

    # third set of parameters have advanced False and training True and will be shown in the training popup dialog
    num_epochs: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 0, 'max': 100000, 'default': 100, 'step': 1})
    model_name: str = field(metadata={'type': 'str', 'harvest': True, 'advanced': False, 'training': True, 'default': 'cyto3', 'step': 1})
        
    # second set of parameters have advanced True and training False and will be shown in the advanced popup dialog
    # None yet...
    
    descriptor = "Micro-sam Instance Framework"

    def __init__(self, parent_path: str,  num_classes: int, start_model: str = None):
        super().__init__(parent_path, num_classes)
        
        self.model = None 

        # microsam models are stored in a directory        
        self.load_mode = LoadMode.Directory
        
        self.num_epochs = 100
        self.model_name = self.generate_model_name('microsam')
    
        # initial model names
        self.model_names = ['vit_b']
        
        # pretrained model names
        self.builtin_names = ['vit_b']
        
        self.model_type = "vit_b"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training
        
    def train(self, updater=None):
        """
        Train the Micro-sam model

        The training patches should already exist in the patch_path directory.
        """
        image_dir = self.patch_path  / 'input0_255'
        segmentation_dir = self.patch_path / 'ground truth0'

        # Load images from multiple files in folder via pattern (here: all tif files)
        raw_key, label_key = "*.tif", "*.tif"

        # The 'roi' argument can be used to subselect parts of the data.
        # Here, we use it to select the first 70 images (frames) for the train split and the other frames for the val split.
        train_roi = np.s_[:, :, :]
        val_roi = np.s_[:5, :, :]

        batch_size = 4  # the training batch size
        patch_shape = (1, 512, 512)  # the size of patches for training

        train_instance_segmentation = True

        sampler = MinInstanceSampler(min_size=25)  # NOTE: The choice of 'min_size' value is paired with the same value in 'min_size' filter in 'label_transform'.

        train_loader = sam_training.default_sam_loader(
            raw_paths=image_dir,
            raw_key=raw_key,
            label_paths=segmentation_dir,
            label_key=label_key,
            with_segmentation_decoder=train_instance_segmentation,
            patch_shape=patch_shape,
            batch_size=batch_size,
            is_seg_dataset=True,
            rois=train_roi,
            shuffle=True,
            raw_transform=sam_training.identity,
            sampler=sampler,
        )

        val_loader = sam_training.default_sam_loader(
            raw_paths=image_dir,
            raw_key=raw_key,
            label_paths=segmentation_dir,
            label_key=label_key,
            with_segmentation_decoder=train_instance_segmentation,
            patch_shape=patch_shape,
            batch_size=batch_size,
            is_seg_dataset=True,
            rois=val_roi,
            shuffle=True,
            raw_transform=sam_training.identity,
            sampler=sampler,
        )
        
        # All hyperparameters for training.
        n_objects_per_batch = 5  # the number of objects per batch that will be sampled

        # The model_type determines which base model is used to initialize the weights that are finetuned.
        # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
        model_type = "vit_b"

        # Run training (best metric 0.027211)
        sam_training.train_sam(
            name=self.model_name,
            save_root=self.model_path, #os.path.join(root_dir, "models"),
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=self.epochs,
            n_objects_per_batch=n_objects_per_batch,
            with_segmentation_decoder=train_instance_segmentation,
            device=self.device,
        )

    def predict(self, img: np.ndarray):
        """
        Predict the segmentation of the image using the current microsam model
        """
        print("Predicting using Micro-sam model")
        best_checkpoint = os.path.join(self.model_path, "best.pt")

        # TODO: tile shape and halo parameters....
        prediction = self.run_automatic_instance_segmentation(
            image=img, checkpoint_path=best_checkpoint, model_type=self.model_type, device=self.device, tile_shape=(384, 384), halo = (64, 64)
        )

        return prediction 
    
    def run_automatic_instance_segmentation(self, image, checkpoint_path, model_type="vit_b_lm", device=None, tile_shape = None, halo = None):
        """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

        NOTE: AIS is supported only for `µsam` models.

        Args:
            image: The input image.
            checkpoint_path: The path to stored checkpoints.
            model_type: The choice of the `µsam` model.
            device: The device to run the model inference.

        Returns:
            The instance segmentation.
        """
        # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=model_type, # choice of the Segment Anything model
            checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
            device=device,  # the device to run the model inference.
            is_tiled = (tile_shape is not None),  # whether the model is tiled or not.
        )

        # Step 2: Get the instance segmentation for the given image.
        prediction = automatic_instance_segmentation(
            predictor=predictor,  # the predictor for the Segment Anything model.
            segmenter=segmenter,  # the segmenter class responsible for generating predictions.
            input_path=image,
            #ndim=2,
            tile_shape=tile_shape,
            halo=halo,
        )

        return prediction
    
    
    def get_model_names(self):
        return self.model_names 
    
    def set_builtin_model(self, model_name):
         pass
    
    def load_model_from_disk(self, model_path):

        self.model_path = model_path

# this line is needed to register the framework on import
BaseFramework.register_framework('MicroSamInstanceFramework', MicroSamInstanceFramework)
