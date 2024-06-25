import numpy as np
from sklearn.metrics import confusion_matrix
import logging
import warnings
import random
from torch.backends import cudnn
import torch
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
# cudnn.deterministic = True
# cudnn.benchmark = False
warnings.filterwarnings("ignore")

'''class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)'''


def eval(gt,pred):
    confusion_mat = confusion_matrix(gt.flatten(), pred.flatten(), labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    iou_per_class = np.diag(confusion_mat) / (confusion_mat.sum(axis=1) + confusion_mat.sum(axis=0) - np.diag(confusion_mat))
    mean_iou = np.nanmean(iou_per_class)
    return mean_iou

def evaluate_eval( hist, epoch=0, dataset_name=None, dataset=None):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    """
    if hist is not None:
        # axis 0: gt, axis 1: prediction
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

        print_evaluate_results(hist, iu, dataset_name=dataset_name, dataset=dataset)
        freq = hist.sum(axis=1) / hist.sum()
        mean_iu = np.nanmean(iu)
        logging.info('mean {}'.format(mean_iu))
        print('mean {}'.format(mean_iu))
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    else:
        mean_iu = 0

    if hist is not None:
        logging.info('-' * 107)
        fmt_str = '[epoch %d], [dataset name %s], [acc %.5f], [acc_cls %.5f], ' +\
                  '[mean_iu %.5f], [fwavacc %.5f]'
        logging.info(fmt_str % (epoch, dataset_name, acc, acc_cls, mean_iu, fwavacc))
        
def print_evaluate_results(hist, iu, dataset_name=None, dataset=None):
    # fixme: Need to refactor this dict
    try:
        id2cat = dataset.id2cat
    except:
        id2cat = {i: i for i in range(19)}
    iu_false_positive = hist.sum(axis=1) - np.diag(hist)
    iu_false_negative = hist.sum(axis=0) - np.diag(hist)
    iu_true_positive = np.diag(hist)

    logging.info('Dataset name: {}'.format(dataset_name))
    print('Dataset name: {}'.format(dataset_name))
    logging.info('IoU:')
    print('IoU')
    logging.info('label_id      label    iU    Precision Recall TP     FP    FN')
    print('label_id      label    iU    Precision Recall TP     FP    FN')
    for idx, i in enumerate(iu):
        # Format all of the strings:
        idx_string = "{:2d}".format(idx)
        class_name = "{:>13}".format(id2cat[idx]) if idx in id2cat else ''
        iu_string = '{:5.1f}'.format(i * 100)
        total_pixels = hist.sum()
        tp = '{:5.1f}'.format(100 * iu_true_positive[idx] / total_pixels)
        fp = '{:5.1f}'.format(
            iu_false_positive[idx] / iu_true_positive[idx])
        fn = '{:5.1f}'.format(iu_false_negative[idx] / iu_true_positive[idx])
        precision = '{:5.1f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx]))
        recall = '{:5.1f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx]))
        logging.info('{}    {}   {}  {}     {}  {}   {}   {}'.format(
            idx_string, class_name, iu_string, precision, recall, tp, fp, fn))
        print('{}    {}   {}  {}     {}  {}   {}   {}'.format(
            idx_string, class_name, iu_string, precision, recall, tp, fp, fn))
        
def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(num_classes * label_true[mask].astype(int) + label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    
    return hist

def fast_hist_pasta(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist
