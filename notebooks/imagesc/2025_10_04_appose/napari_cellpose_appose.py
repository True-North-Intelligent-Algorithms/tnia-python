import appose
from skimage.io import imread
import os

# Local paths to image and environments
IMAGE_PATH = "/home/bnorthan/images/tnia-python-images/imagesc/2025_05_10_SOTA_Test_Set/TestHidden_002.tif"

# Environment paths
CELLPOSE3_ENV_PATH = "/home/bnorthan/mambaforge/envs/microsam_cellpose_instanseg"
CELLPOSESAM_ENV_PATH = "/home/bnorthan/mambaforge/envs/microsam_cellpose_sam"

def build_env(path: str):
    """Small helper to build an appose environment from a filesystem path."""
    return appose.base(path).build()

def show_task_output(task, environment_name: str = ""):
    """Print concise task diagnostics.

    Shows error (if present), task.message (if present), and outputs['message'] value.
    environment_name: optional prefix derived from environment path.
    """
    prefix = f"[{environment_name}] " if environment_name else ""
    if getattr(task, 'error', None):
        print(f"{prefix}ERROR: {task.error}")
    
    # Some task APIs may not set .message; guard accordingly
    if getattr(task, 'message', None):
        print(f"{prefix}task.message: {task.message}")
    
    outputs = getattr(task, 'outputs', {}) or {}
    if 'message' in outputs:
        print(f"{prefix}outputs['message']: {outputs['message']}")

run_cellpose = """
import cellpose
import appose
from cellpose import models
    
major_number = cellpose.version.split('.')[0]
msg = f"Cellpose version: {cellpose.version} (major number: {major_number})"
task.outputs["message"] = msg

if major_number == '3':
    model = models.Cellpose(gpu=True, model_type='cyto2')
elif major_number == '4':
    model = models.CellposeModel(gpu=True)

array = ndarr.ndarray()

result = model.eval(array, diameter=60, niter=2000)[0]

ndarr_result = appose.NDArray(dtype=str(array.dtype), shape=result.shape)
ndarr_result.ndarray()[:] = result

task.outputs["cellpose_result"] = ndarr_result

"""

img = imread(IMAGE_PATH)
ndarr_img = appose.NDArray(dtype=str(img.dtype), shape=img.shape)
ndarr_img.ndarray()[:] = img

cellpose3_env = build_env(CELLPOSE3_ENV_PATH)
cellposesam_env = build_env(CELLPOSESAM_ENV_PATH)

cellpose3_name = os.path.basename(CELLPOSE3_ENV_PATH.rstrip('/'))
cellposesam_name = os.path.basename(CELLPOSESAM_ENV_PATH.rstrip('/'))

with cellpose3_env.python() as python:
    run_cellpose_task = python.task(run_cellpose, inputs = {"ndarr": ndarr_img}, queue="main")
    run_cellpose_task.wait_for()
    show_task_output(run_cellpose_task, environment_name=cellpose3_name)

    cellpose3_result = run_cellpose_task.outputs["cellpose_result"]
    
with cellposesam_env.python() as python:
    run_cellpose_task = python.task(run_cellpose, inputs = {"ndarr": ndarr_img}, queue="main")
    run_cellpose_task.wait_for()
    show_task_output(run_cellpose_task, environment_name=cellposesam_name)

    try:
        print(run_cellpose_task.outputs['message'])
        cellposesam_result = run_cellpose_task.outputs["cellpose_result"]
    except KeyError:
        print(f"{cellposesam_name} did not produce a valid output.")
        cellposesam_result = None

import napari
viewer = napari.Viewer()
viewer.add_image(ndarr_img.ndarray(), name="Original Image")
viewer.add_labels(cellpose3_result.ndarray(), name="Cellpose v3 Result")

if cellposesam_result is not None:
    viewer.add_labels(cellposesam_result.ndarray(), name="Cellpose v4 Result")
napari.run()
