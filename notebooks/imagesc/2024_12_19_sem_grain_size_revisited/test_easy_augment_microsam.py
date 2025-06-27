import napari

from napari_easy_augment_batch_dl import easy_augment_batch_dl
from micro_sam_instance_framework import MicroSamInstanceFramework
from pathlib import Path

viewer = napari.Viewer()

batch_dl = easy_augment_batch_dl.NapariEasyAugmentBatchDL(viewer, label_only = False)

viewer.window.add_dock_widget(
    batch_dl
)


parent_path = Path(r'D:\images\tnia-python-images\\imagesc\\2024_12_19_sem_grain_size_revisit')
model_path = parent_path / 'models' / 'microsam' / 'checkpoints'
model_name = 'microsam_grains'

model_type = MicroSamInstanceFramework.descriptor

batch_dl.load_image_directory(parent_path)
# optionally set a pretrained model and settings so we can do prediction

if model_name is not None:
    batch_dl.network_architecture_drop_down.setCurrentText(model_type)

    widget = batch_dl.deep_learning_widgets[model_type]

    widget.load_model_from_path(model_path / model_name)

napari.run()