# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu), Ziyi Dou (zdou@cs.ucla.edu)
# --------------------------------------------------------
import copy
import itertools
import logging
from collections import OrderedDict
import torch
from pycocotools.cocoeval import COCOeval

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


class RetrievalEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).
    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name=None,
        output_dir=None,
        ensemble=False,
        distributed=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
            allow_cached_coco (bool): Whether to use cached coco json from previous validation
                runs. You should set this to False if you need to use different validation data.
                Defaults to True.
        """
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._ensemble = ensemble
        self._distributed = distributed

    def reset(self):
        self._i2t_score = []
        self._t2i_score = []

    def process(self, inputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for _input in inputs:
            self._i2t_score.append(_input['i2t'])
            self._t2i_score.append(_input['t2i'])

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """        
        i2t_scores = torch.stack(self._i2t_score)
        t2i_scores = torch.stack(self._t2i_score)

        i2t_topk10 = i2t_scores.topk(10, dim=1)
        i2t_topk5 = i2t_scores.topk(5, dim=1)
        i2t_topk1 = i2t_scores.topk(1, dim=1)
        
        i2t10 = (i2t_topk10.indices == 0).max(dim=1).values.sum() / i2t_topk10.indices.size(0)
        i2t5 = (i2t_topk5.indices == 0).max(dim=1).values.sum() / i2t_topk5.indices.size(0)
        i2t1 = (i2t_topk1.indices == 0).max(dim=1).values.sum() / i2t_topk1.indices.size(0)
        
        t2i_topk10 = t2i_scores.topk(10, dim=1)
        t2i_topk5 = t2i_scores.topk(5, dim=1)
        t2i_topk1 = t2i_scores.topk(1, dim=1)
        
        t2i10 = (t2i_topk10.indices == 0).max(dim=1).values.sum() / t2i_topk10.indices.size(0)
        t2i5 = (t2i_topk5.indices == 0).max(dim=1).values.sum() / t2i_topk5.indices.size(0)
        t2i1 = (t2i_topk1.indices == 0).max(dim=1).values.sum() / t2i_topk1.indices.size(0)
        
        self._results = OrderedDict()
        # Copy so the caller can do whatever with results
        self._results['recall'] = {}
        self._results['recall']['i2t10'] = float("{:.3f}".format(i2t10.item() * 100))
        self._results['recall']['i2t5'] = float("{:.3f}".format(i2t5.item() * 100))
        self._results['recall']['i2t1'] = float("{:.3f}".format(i2t1.item() * 100))
        self._results['recall']['t2i10'] = float("{:.3f}".format(t2i10.item() * 100))
        self._results['recall']['t2i5'] = float("{:.3f}".format(t2i5.item() * 100))
        self._results['recall']['t2i1'] = float("{:.3f}".format(t2i1.item() * 100))
        self._logger.info(self._results)
        return copy.deepcopy(self._results)