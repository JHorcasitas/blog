import logging
from typing import Dict
from collections import defaultdict

from torch import Tensor

from ImageNet.config import map_data_path


class MetricTracker:
    """Keep track of TP, FP, FN and compute some aggregation functions on
    them."""

    def __init__(self, class_names: Dict[int, str]) -> None:
        index_list, class_list = [], []
        with open(map_data_path, 'rt') as f:
            for line in f:
                line = line.strip('\n').split()
                index_list.append(line[1])
                class_list.append(line[2])
        self.index_class_map = dict(zip(index_list, class_list))
        self.num_classes = len(self.index_class_ma)
        self.reset()

    @property
    def precision(self):
        precision = {}
        for k, v in self.metrics:
            try:
                precision[k] = v['TP'] / (v['TP'] + v['FP'])
            except ZeroDivisionError:
                precision[k] = None
        return precision

    @property
    def recall(self):
        precision = {}
        for k, v in self.metrics:
            try:
                precision[k] = v['TP'] / (v['TP'] + v['FN'])
            except ZeroDivisionError:
                precision[k] = None
        return precision

    @property
    def top_1_score(self):
        resume = defaultdict(int)
        for v in self.metrics.values():
            resume['TP'] += v['TP']
            resume['FP'] += v['FP']
            resume['FN'] += v['FN']
        return resume['TP'] / (resume['TP'] + resume['FP'] + resume['FN'])

    def reset(self):
        """Deletes all previous saved data. Set self.metrics to its default
        value. Metrics doesn't contain TN because it doesn't make sense for
        a multi class classification problem."""
        self.metrics = {}
        for i in range(0, self.num_classes):
            self.metrics[i] = {'TP': 0, 'FP': 0, 'FN': 0}

    def update_metrics(self, pred: Tensor, target: Tensor) -> None:
        """
        :param pred: [description]
        :param target: [description]
        """
        for i in range(len(pred)):
            if target[i] == pred[i]:
                self.metrics[target[i]]['TP'] += 1
            else:
                self.metrics[pred[i]]['FP'] += 1
                self.metrics[target[i]]['FN'] += 1

    def log_metrics(self):
        t1, p, r = self.top_1_score, self.precision, self.recall
        for i in range(0, self.num_classes):
            message = 'Class: {}, Precision: {}, Recall: {}, F1: {}'
            class_name = self.index_class_map[i]
            logging.info(message.format(class_name, p[i], r[i], t1[i]))
