import numpy as np
import SimpleITK as sitk


def ct_seg_perf(
    pred_mask=None,
    true_mask=None,
    pred_mask_path=None,
    true_mask_path=None,
):
    """
    Evaluate the performance of CT image segmentation.

    Parameters:
    pred_mask (np.ndarray): Predicted segmentation mask.
    true_mask (np.ndarray): Ground truth segmentation mask.
    pred_mask_path (str): File path to the predicted mask.
    true_mask_path (str): File path to the ground truth mask.

    Returns:
    dict: A dictionary containing Dice coefficient, IOU score, accuracy, precision, recall, F1 score, and false positive rate for each class.
    """
    # If file paths are provided, read the masks from the files
    if pred_mask_path is not None:
        pred_mask = read_mask(pred_mask_path)
    if true_mask_path is not None:
        true_mask = read_mask(true_mask_path)

    if np.all(true_mask == 0) and np.all(pred_mask == 0):
        return {
            "1": {
                "Dice": 1.0,
                "IOU": 1.0,
                "Accuracy": 1.0,
                "Precision": 1.0,
                "Recall": 1.0,
                "F1": 1.0,
                "FPrate": 0.0,
            }
        }

    # Remove parts where true_mask == 0
    true_mask_nonzero = true_mask[true_mask != 0]
    true_classes = np.unique(true_mask_nonzero)

    # Remove parts where pred_mask == 0
    pred_mask_nonzero = pred_mask[pred_mask != 0]
    pred_classes = np.unique(pred_mask_nonzero)

    # Get the union of the unique classes from both masks
    classes = np.union1d(true_classes, pred_classes)

    # Initialize the results dictionary
    results = {}

    # Calculate performance metrics for each class
    for class_id in classes:
        pred_binary_mask = (pred_mask == class_id).astype(np.int8)
        true_binary_mask = (true_mask == class_id).astype(np.int8)

        # Calculate intersection and union
        intersection = np.bitwise_and(pred_binary_mask, true_binary_mask)
        union = np.bitwise_or(pred_binary_mask, true_binary_mask)
        false_positive = np.bitwise_and(
            pred_binary_mask, np.bitwise_not(true_binary_mask)
        )

        # Calculate performance metrics
        dice = (
            2 * intersection.sum() / (pred_binary_mask.sum() + true_binary_mask.sum())
            if pred_binary_mask.sum() + true_binary_mask.sum() > 0
            else 0
        )
        iou = intersection.sum() / union.sum() if union.sum() > 0 else 0
        accuracy = (
            np.equal(pred_binary_mask, true_binary_mask).sum() / true_binary_mask.size
        )
        precision = (
            intersection.sum() / pred_binary_mask.sum()
            if pred_binary_mask.sum() > 0
            else 0
        )
        recall = (
            intersection.sum() / true_binary_mask.sum()
            if true_binary_mask.sum() > 0
            else 0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        fp_rate = (
            false_positive.sum() / (false_positive.sum() + true_binary_mask.sum())
            if (false_positive.sum() + true_binary_mask.sum()) > 0
            else 0
        )

        # Add the results to the dictionary
        results[str(class_id)] = {
            "Dice": dice,
            "IOU": iou,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FPrate": fp_rate,
        }

    return results


def read_mask(mask_path):
    """
    Read the mask from the given file path.

    Parameters:
    mask_path (str): File path to the mask.

    Returns:
    np.ndarray: The mask array.
    """
    if mask_path.endswith(".npy"):
        return np.load(mask_path)
    else:
        return sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
