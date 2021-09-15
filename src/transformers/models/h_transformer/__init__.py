# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

# rely on isort to merge the imports
from ...file_utils import _LazyModule, is_tokenizers_available
from ...file_utils import is_torch_available


_import_structure = {
    "configuration_h_transformer": ["H_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "HTransformerConfig"],
    "tokenization_h_transformer": ["HTransformerTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_h_transformer_fast"] = ["HTransformerTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_h_transformer"] = [
        "H_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "HTransformerForConditionalGeneration",
        "HTransformerForQuestionAnswering",
        "HTransformerForSequenceClassification",
        "HTransformerForCausalLM",
        "HTransformerForMaskedLM",
        "HTransformerModel",
        "HTransformerPreTrainedModel",
    ]




if TYPE_CHECKING:
    from .configuration_h_transformer import H_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, HTransformerConfig
    from .tokenization_h_transformer import HTransformerTokenizer

    if is_tokenizers_available():
        from .tokenization_h_transformer_fast import HTransformerTokenizerFast

    if is_torch_available():
        from .modeling_h_transformer import (
            H_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            HTransformerForQuestionAnswering,
            HTransformerForMaskedLM,
            HTransformerPreTrainedModel,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
