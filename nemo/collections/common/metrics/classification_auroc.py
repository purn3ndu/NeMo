# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from pytorch_lightning.metrics import TensorMetric
from sklearn.metrics import roc_auc_score
from typing import List, Union
__all__ = ['tensor2list', 'auroc']


def tensor2list(tensor: torch.Tensor) -> List[Union[int, float]]:
    """ Converts tensor to a list """
    return tensor.detach().cpu().tolist()

def auroc(preds: List[float], labels: List[float]):
    auc_roc = roc_auc_score(labels, preds, multi_class='ovr')
    return  auc_roc
