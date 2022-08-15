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

import torch
from transformers_interpret import SequenceClassificationExplainer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from apps.visualization_utils import visualize_text


class CustomSequenceClassificationExplainer(SequenceClassificationExplainer):
    """
    Subclassing to replace `visualize()` method with custom styling.

    Namely, removing a few columns, styling fonts, and re-arrangning legend position.
    """

    # NOTE - this function is borrowed and overwritten from the Transformers Interpret library
    # https://github.com/cdpierse/transformers-interpret/blob/3076df0d94963e66bade1e7047808e64d4490247/transformers_interpret/explainers/sequence_classification.py#L133
    def visualize(self, html_filepath: str = None, true_class: str = None):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.
        Otherwise pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.
        If the true class is known for the text that can be passed to `true_class`
        """
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        attr_class = self.id2label[self.selected_index]

        if self._single_node_output:
            if true_class is None:
                true_class = round(float(self.pred_probs))
            predicted_class = round(float(self.pred_probs))
            attr_class = round(float(self.pred_probs))
        else:
            if true_class is None:
                true_class = self.selected_index
            predicted_class = self.predicted_class_name

        score_viz = self.attributions.visualize_attributions(  # type: ignore
            self.pred_probs,
            predicted_class,
            true_class,
            attr_class,
            tokens,
        )

        # NOTE: here is the overwritten function
        html = visualize_text([score_viz])

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)

        return html


class InterpretTransformer:
    """
    Utility for visualizing word attribution scores from Transformer models.

    This class utilizes the [Transformers Interpret](https://github.com/cdpierse/transformers-interpret)
    libary to calculate word attributions using a techinique called Integrated Gradients.

    Attributes:
        cls_model_identifier (str)

    """

    def __init__(self, cls_model_identifier: str):

        self.cls_model_identifier = cls_model_identifier
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

        self._initialize_hf_artifacts()

    def _initialize_hf_artifacts(self):
        """
        Initialize a HuggingFace artifacts (tokenizer and model) according
        to the provided identifiers for both SBert and the classification model.
        Then initialize the word attribution explainer with the HF model+tokenizer.

        """

        # classifer
        self.cls_tokenizer = AutoTokenizer.from_pretrained(self.cls_model_identifier)
        self.cls_model = AutoModelForSequenceClassification.from_pretrained(
            self.cls_model_identifier
        )
        self.cls_model.to(self.device)

        # transformers interpret
        self.explainer = CustomSequenceClassificationExplainer(
            self.cls_model, self.cls_tokenizer
        )

    def visualize_feature_attribution_scores(self, text: str, class_index: int = 0):
        """
        Calculates and visualizes feature attributions using integrated gradients.

        Args:
            text (str) - text to get attributions for
            class_index (int) - Optional output index to provide attributions for

        """
        self.explainer(text, index=class_index)
        return self.explainer.visualize()
