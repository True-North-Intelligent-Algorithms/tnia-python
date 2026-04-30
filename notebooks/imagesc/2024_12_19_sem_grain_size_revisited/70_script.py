import imageio.v3 as imageio

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def run_automatic_instance_segmentation(image, checkpoint_path, model_type, device, tile_shape, halo):

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=checkpoint_path,
        device=device,
        is_tiled=(tile_shape is not None),
    )

    prediction = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=tile_shape,
        halo=halo,
    )

    return prediction


def main():
    image = imageio.imread("/home/bnorthan/images/tnia-python-images/imagesc/2024_12_19_sem_grain_size_revisit2/211122_AM_Al2O3_SE_027_sp.tif")

    prediction = run_automatic_instance_segmentation(
        image=image,
        checkpoint_path="/home/bnorthan/code/i2k/tnia/tnia-python/notebooks/imagesc/2024_12_19_sem_grain_size_revisited/models/checkpoints/sam_grains4//best.pt",
        model_type="vit_b",
        device=None,
        tile_shape=(384, 384),
        halo=(64, 64),
    )

    import napari
    v = napari.Viewer()
    v.add_image(image, name="Image")
    v.add_labels(prediction, name="Prediction")
    napari.run()


main()
