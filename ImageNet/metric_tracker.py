import logging
from collections import defaultdict

from torch import Tensor

from ImageNet.config import map_data_path


class MetricTracker:
    """Keep track of TP, FP, FN and compute some aggregation functions on
    them."""

    def __init__(self) -> None:
        index_list, class_list = [], []
        with open(map_data_path, 'rt') as f:
            for line in f:
                line = line.strip('\n').split()
                index_list.append(int(line[1]))
                class_list.append(line[2])
        self.index_class_map = dict(zip(index_list, class_list))
        self.num_classes = len(self.index_class_map)
        self.reset()

    @property
    def precision(self):
        precision = {}
        for k, v in self.metrics.items():
            try:
                precision[k] = v['TP'] / (v['TP'] + v['FP'])
            except ZeroDivisionError:
                precision[k] = None
        return precision

    @property
    def recall(self):
        precision = {}
        for k, v in self.metrics.items():
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
                self.metrics[target[i].item()]['TP'] += 1
            else:
                self.metrics[pred[i].item()]['FP'] += 1
                self.metrics[target[i].item()]['FN'] += 1

    def log_metrics(self):
        p, r = self.precision, self.recall
        for i in range(0, self.num_classes):
            message = 'Class: {}, Precision: {}, Recall: {}'
            logging.info(message.format(self.index_class_map[i], p[i], r[i]))
        logging.info(f'T1 score: {self.top_1_score}')
