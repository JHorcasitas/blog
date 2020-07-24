from copy import deepcopy
from statistics import mean
from functools import partial
from typing import Dict, Optional, List, Callable, Any

from torch import Tensor


class MetricTracker:
    """Keep track of TP, TN, FP, FN and compute some aggregation functions on
    them.

    :param class_names: Dictionary mapping the index of a class and its name.
    """

    def __init__(self, class_names: Dict[int, str]) -> None:
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.reset()

    @property
    def precision(self):
        precision = {}
        for k, v in self.index2name(self.metrics):
            try:
                precision[k] = v['TP'] / (v['TP'] + v['TN'])
            except ZeroDivisionError:
                precision[k] = None
        return precision

    @property
    def recall(self):
        precision = {}
        for k, v in self.index2name(self.metrics):
            try:
                precision[k] = v['TP'] / (v['TP'] + v['FN'])
            except ZeroDivisionError:
                precision[k] = None
        return precision

    @property
    def f1_score(self):
        f1, p, r = {}, self.precision, self.recall
        for name, in self.class_names:
            try:
                f1[name] = 2 * ((p[name] * r[name]) / (p[name] + r[name]))
            except (ZeroDivisionError, TypeError):
                f1[name] = None
        return f1

    @property
    def global_f1_score(self):
        f1 = self.f1_score
        if all(v is not None for v in f1.values()):
            return mean(f1.values())
        else:
            return None

    def reset(self):
        """Deletes all previous saved data. Sets self.metrics to its default
        value"""
        self.metrics = {}
        for i in range(1, self.num_classes + 1):
            self.metrics[i] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    def increase_dict_count(self, dict_obj, key):
        dict_obj[key] += 1

    def update_metrics(self, pred: Tensor, target: Tensor) -> None:
        """
        :param pred: [description]
        :param target: [description]
        """
        for i in range(len(pred)):
            if target[i] == pred[i]:
                self.nested_dict_update(
                    dict_obj=self.metrics,
                    keys=[target[i]],
                    func=partial(self.increase_dict_count, key='TP'))
                self.nested_dict_update(
                    dict_obj=self.metrics,
                    keys=[j for j in range(self.num_classes) if j != target[i]],
                    func=partial(self.increase_dict_count, key='TN'))
            else:
                self.nested_dict_update(
                    dict_obj=self.metrics,
                    keys=[target[i]],
                    func=partial(self.increase_dict_count, key='FP'))
                self.nested_dict_update(
                    dict_obj=self.metrics,
                    keys=[pred[i]],
                    func=partial(self.increase_dict_count, key='FN'))
                self.nested_dict_update(
                    dict_obj=self.metrics,
                    keys=[j for j in range(len(pred)) if j not in {target[i], pred[i]}],
                    func=partial(self.increase_dict_count, key='TN'))

    def nested_dict_update(self,
                           dict_obj: Dict[str, int],
                           keys: List[int],
                           func: Callable[[Dict[str, int]], None]) -> None:
        for k, v in deepcopy(dict_obj.items()):
            if k in keys:
                func(dict_obj[k])

    def index2name(self, dict_obj: Dict[int, Any]) -> Dict[str, Any]:
        """Maps the keys of a dictionary from index values to class names."""
        ret_dict = {}
        for index, value in dict_obj.items():
            ret_dict[self.class_names[index]] = value
        return ret_dict

    def log_metrics(self, logger, opt_msg: Optional[str] = None):
        p, r, f = self.precision, self.recall, self.f1_score
        for name in self.class_names:
            base_msg = f'Class: {name}, Precision: {p[name]}, Recall: {r[name]}, F1 Score: {f[name]}'
            full_msg = base_msg + ', ' + opt_msg if opt_msg else base_msg
            logger.info(full_msg)
