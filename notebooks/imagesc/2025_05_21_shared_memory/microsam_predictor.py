import torch
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def microsam_predict(image):
    tile_shape = None #(384, 384)
    halo = (64, 64)
    model_type = "vit_b_lm"
    model_type = "vit_l_histopathology"

    print(f"Using model type: {model_type}")


    device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training
    # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, # choice of the Segment Anything model
        #checkpoint=best_checkpoint,  # overwrite to pass your own finetuned model.
        device=device,  # the device to run the model inference.
        is_tiled = False #(tile_shape is not None),  # whether the model is tiled or not.
    )

    # Step 2: Get the instance segmentation for the given image.
    prediction = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,
        ndim=2,
        #tile_shape=tile_shape,
        #halo=halo,
    )

    return prediction

