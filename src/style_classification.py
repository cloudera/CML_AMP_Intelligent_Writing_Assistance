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

from typing import List, Union

import torch
import numpy as np
from pyemd import emd
from transformers import pipeline


class StyleIntensityClassifier:
    """
    Utility for classifying style and calculating Style Transfer Intensity between
    two pieces of text (i.e. input and output of TST model).

    This custom evaluation metric aims to quantify the magnitude of transferred
    style between two texts. To accomplish this, we pass input and output texts
    through a trained style classifier to produce two distributions. We then
    utilize Earth Movers Distance (EMD) to calculate the minimum "cost"/"work"
    required to turn the input distribution into the output distribution. This
    metric allows us to capture a more nuanced, per-example measure of style
    transfer when compared to simply aggregating binary classifications over
    records in a dataset.

    Attributes:
        model_identifier (str)

    """

    def __init__(self, model_identifier: str):
        self.model_identifier = model_identifier
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        self._build_pipeline()

    def _build_pipeline(self):

        self.pipeline = pipeline(
            task="text-classification",
            model=self.model_identifier,
            device=self.device,
            return_all_scores=True,
        )

    def score(self, input_text: Union[str, List[str]]):
        """
        Classify a given input text using the model initialized by the class.

        Args:
            input_text (`str` or `List[str]`) - Input text for classification

        Returns:
            classification (dict) - a dictionary containing the label, score, and
                distribution between classes

        """
        if isinstance(input_text, str):
            tmp = list()
            tmp.append(input_text)
            input_text = tmp

        result = self.pipeline(input_text)
        distributions = np.array(
            [[label["score"] for label in item] for item in result]
        )
        return [
            {
                "label": self.pipeline.model.config.id2label[scores.argmax()],
                "score": round(scores.max(), 4),
                "distribution": scores.tolist(),
            }
            for scores in distributions
        ]

    def calculate_transfer_intensity(
        self, input_text: List[str], output_text: List[str], target_class_idx: int = 1
    ) -> List[float]:
        """
        Calcualates the style transfer intensity (STI) between two pieces of text.

        Args:
            input_text (list) - list of input texts with indicies corresponding
                to counterpart in output_text
            ouptput_text (list) - list of output texts with indicies corresponding
                to counterpart in input_text
            target_class_idx (int) - index of the target style class used for directional
                score correction

        Returns:
            A list of floats with corresponding style transfer intensity scores.

        """

        if len(input_text) != len(output_text):
            raise ValueError(
                "input_text and output_text must be of same length with corresponding items"
            )

        input_dist = [item["distribution"] for item in self.score(input_text)]
        output_dist = [item["distribution"] for item in self.score(output_text)]

        return [
            self.calculate_emd(input_dist[i], output_dist[i], target_class_idx)
            for i in range(len(input_dist))
        ]

    def calculate_transfer_intensity_fraction(
        self, input_text: List[str], output_text: List[str], target_class_idx: int = 1
    ) -> List[float]:
        """
        Calcualates the style transfer intensity (STI) _fraction_ between two pieces of text.
        See `calcualte_sti_fraction()` for details.

        Args:
            input_text (list) - list of input texts with indicies corresponding
                to counterpart in output_text
            ouptput_text (list) - list of output texts with indicies corresponding
                to counterpart in input_text
            target_class_idx (int) - index of the target style class used for directional
                score correction

        Returns:
            A list of floats with corresponding style transfer intensity scores.

        """

        if len(input_text) != len(output_text):
            raise ValueError(
                "input_text and output_text must be of same length with corresponding items"
            )

        input_dist = [item["distribution"] for item in self.score(input_text)]
        output_dist = [item["distribution"] for item in self.score(output_text)]

        return [
            self.calculate_sti_fraction(
                input_dist[i],
                output_dist[i],
                ideal_dist=[0.0, 1.0],
                target_class_idx=target_class_idx,
            )
            for i in range(len(input_dist))
        ]

    def calculate_sti_fraction(
        self, input_dist, output_dist, ideal_dist=[0.0, 1.0], target_class_idx=1
    ):
        """
        Calculate the direction-corrected style transfer intensity fraction between
        two style distributions of equal length.

        If output_dist moves closer towards target style class, the metric represents the percentage of
        the possible _target_ style distribution that was captured during the transfer. If output_dist
        moves further from the target style class, the metric represents the percentage of the possible
        _source_ style distribution that was captured.

        Args:
            input_dist (list) - probabilities assigned to the style classes
                from the input text to style transfer model
            output_dist (list) - probabilities assigned to the style classes
                from the outut text of the style transfer model
            ideal_dist (list, optional): The maximum possibly distribution. Defaults to [0.0, 1.0].
            target_class_idx (int, optional)

        Returns:
            sti_fraction (float)
        """

        sti = self.calculate_emd(input_dist, output_dist, target_class_idx)

        if sti > 0:
            potential = self.calculate_emd(input_dist, ideal_dist, target_class_idx)
        else:
            potential = self.calculate_emd(
                input_dist, ideal_dist[::-1], target_class_idx
            )

        return sti / potential

    @staticmethod
    def calculate_emd(input_dist, output_dist, target_class_idx):
        """
        Calculate the direction-corrected Earth Mover's Distance (aka Wasserstein distance)
        between two distributions of equal length. Here we penalize the EMD score if
        the output text style moved further away from the target style.

        Reference: https://github.com/passeul/style-transfer-model-evaluation/blob/master/code/style_transfer_intensity.py

        Args:
            input_dist (list) - probabilities assigned to the style classes
                from the input text to style transfer model
            output_dist (list) - probabilities assigned to the style classes
                from the outut text of the style transfer model

        Returns:
            emd (float) - Earth Movers Distance between the two distributions

        """

        N = len(input_dist)
        distance_matrix = np.ones((N, N))
        dist = emd(np.array(input_dist), np.array(output_dist), distance_matrix)

        transfer_direction_correction = (
            1 if output_dist[target_class_idx] >= input_dist[target_class_idx] else -1
        )

        return round(dist * transfer_direction_correction, 4)
