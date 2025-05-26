from segment_everything.detect_and_segment import segment_from_stacked_labels
from segment_everything.prompt_generator import YoloDetector
from segment_everything.weights_helper import get_weights_path
from segment_everything.stacked_labels import StackedLabels
from segment_everything.detect_and_segment import segment_from_stacked_labels

def mobilesam_predict(image):
    conf = 0.3
    iou = 0.8
    imagesz = 1024

    yolo_detecter = YoloDetector(str(get_weights_path("ObjectAwareModel")), "ObjectAwareModelFromMobileSamV2", device='cuda')

    results = yolo_detecter.get_results(image, conf=conf, iou= iou, imgsz=imagesz, max_det=10000)
    bbs=results[0].boxes.xyxy.cpu().numpy()
    stacked_labels = StackedLabels.from_yolo_results(bbs, None, image)
    segmented_stacked_labels = segment_from_stacked_labels(stacked_labels, "MobileSamV2")
    segmented_stacked_labels.sort_largest_to_smallest()
    labels = segmented_stacked_labels.make_2d_labels(type="min")

    return labels