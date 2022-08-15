# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2022
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os
from typing import List
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class StyleAttributeData:
    source_attribute: str
    target_attribute: str
    examples: List[str]
    cls_model_path: str
    seq2seq_model_path: str
    sbert_model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_base_url: str = "https://huggingface.co/"

    def __post_init__(self):
        self._make_attribute_selection_string()
        self._make_attribute_AND_string()
        self._make_attribute_THAN_string()

    def _make_attribute_selection_string(self):
        self.attribute_selecting_string = (
            f"{self.source_attribute}-{self.target_attribute}"
        )

    def _make_attribute_AND_string(self):
        self.attribute_AND_string = (
            f"**{self.source_attribute}** and **{self.target_attribute}**"
        )

    def _make_attribute_THAN_string(self):
        self.attribute_THAN_string = (
            f"**{self.source_attribute}** than **{self.target_attribute}**"
        )

    def build_model_url(self, model_type: str):
        """
        Build a complete HuggingFace url for the given `model_type`.

        Args:
            model_type (str): "cls", "seq2seq", "sbert"
        """
        attr_name = f"{model_type}_model_path"
        return os.path.join(self.hf_base_url, getattr(self, attr_name))


# instantiate data classes & collect all data class instances
DATA_PACKET = {
    "subjective-to-neutral": StyleAttributeData(
        source_attribute="subjective",
        target_attribute="neutral",
        examples=[
            "another strikingly elegant four-door design for the bentley s3 continental came from james.",
            "the band plays an engaging and contagious rhythm known as brega pop and calypso.",
            "chemical abstracts service (cas), a prominent division of the american chemical society, is the world's leading source of chemical information.",
            "the final fight scene is with the martial arts great, master ninja sho kosugi.",
        ],
        cls_model_path="cffl/bert-base-styleclassification-subjective-neutral",
        seq2seq_model_path="cffl/bart-base-styletransfer-subjective-to-neutral",
    ),
    "informal-to-formal": StyleAttributeData(
        source_attribute="informal",
        target_attribute="formal",
        examples=[
            "that was funny LOL",
            "btw - ur avatar looks familiar",
            "i loooooooooooooooooooooooove going to the movies.",
            "haha, thatd be dope",
        ],
        cls_model_path="cointegrated/roberta-base-formality",
        seq2seq_model_path="prithivida/informal_to_formal_styletransfer",
    ),
}


def format_classification_results(id2label: dict, cls_result):
    """
    Formats classification output to be plotted using Altair.

    Args:
        id2label (dict): Transformer model's label dictionary
        cls_result (List): Classification pipeline output
    """

    labels = [v for k, v in id2label.items()]

    format_cls_result = []

    for i in range(len(labels)):
        temp = defaultdict()
        temp["type"] = labels[i].capitalize()
        temp["value"] = round(cls_result[0]["distribution"][i], 4)

        if i == 0:
            temp["percentage_start"] = 0
            temp["percentage_end"] = temp["value"]
        else:
            temp["percentage_start"] = 1 - temp["value"]
            temp["percentage_end"] = 1

        format_cls_result.append(temp)

    return format_cls_result


def string_to_list_string(text: str):
    return np.expand_dims(np.array(text), axis=0).tolist()
