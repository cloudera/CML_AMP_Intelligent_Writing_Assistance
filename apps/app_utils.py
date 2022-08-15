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

from typing import List

import tokenizers
import streamlit as st

from src.style_transfer import StyleTransfer
from src.style_classification import StyleIntensityClassifier
from src.content_preservation import ContentPreservationScorer
from src.transformer_interpretability import InterpretTransformer
from apps.data_utils import StyleAttributeData, string_to_list_string

# CALLBACKS
def increment_page_progress():
    st.session_state.page_progress += 1


def reset_page_progress_state():
    del st.session_state.st_result
    st.session_state.page_progress = 1


# UTILITY CLASSES
class DisableableButton:
    """
    Utility class for creating "disable-able" buttons upon click.

    We initialize an empty container, then update that container with buttons
    upon calling `create_enabled_button` and `disable` methods where clicking
    is enabled and then disabled, respectively.

    """

    def __init__(self, button_number, button_text):
        self.button_number = button_number
        self.button_text = button_text

    def _init_placeholder_container(self):
        self.ph = st.empty()

    def create_enabled_button(self):
        self._init_placeholder_container()
        self.ph.button(
            self.button_text,
            on_click=increment_page_progress,
            key=f"ph{self.button_number}_before",
            disabled=False,
        )

    def disable(self):
        self.ph.button(
            self.button_text, key=f"ph{self.button_number}_after", disabled=True
        )


# CACHED FUNCTIONS
@st.cache(
    hash_funcs={tokenizers.Tokenizer: lambda _: None},
    allow_output_mutation=True,
    show_spinner=False,
)
def get_cached_style_intensity_classifier(
    style_data: StyleAttributeData,
) -> StyleIntensityClassifier:
    """
    Return a cached style classifier.

    This function overwrites the existing model's config values for
    `id2label` and `label2id`.

    Args:
        style_data (StyleAttributeData)

    Returns:
        StyleIntensityClassifier
    """
    sic = StyleIntensityClassifier(style_data.cls_model_path)

    # create or overwrite id-label lookup in model config
    sic.pipeline.model.config.__dict__["id2label"] = {
        i: a
        for i, a in enumerate(
            [
                style_data.source_attribute.capitalize(),
                style_data.target_attribute.capitalize(),
            ]
        )
    }
    sic.pipeline.model.config.__dict__["label2id"] = {
        v: k for k, v in sic.pipeline.model.config.__dict__["id2label"].items()
    }

    return sic


@st.cache(
    hash_funcs={tokenizers.Tokenizer: lambda _: None},
    allow_output_mutation=True,
    show_spinner=False,
)
def get_cached_word_attributions(
    text_sample: str, style_data: StyleAttributeData
) -> str:
    """
    Calculated word attributions and return HTML visual.

     This function overwrites the existing model's config values for
    `id2label` and `label2id`.

    Args:
        text_sample (str)
        style_data (StyleAttributeData)

    Returns:
        str
    """
    it = InterpretTransformer(cls_model_identifier=style_data.cls_model_path)

    # create or overwrite id-label lookup in model config
    it.explainer.id2label = {
        i: a
        for i, a in enumerate(
            [
                style_data.source_attribute.capitalize(),
                style_data.target_attribute.capitalize(),
            ]
        )
    }
    it.explainer.label2id = {v: k for k, v in it.explainer.id2label.items()}
    return it.visualize_feature_attribution_scores(text_sample).data


@st.cache(
    hash_funcs={tokenizers.Tokenizer: lambda _: None},
    allow_output_mutation=True,
    show_spinner=False,
)
def get_sti_metric(
    input_text: str, output_text: str, style_data: StyleAttributeData
) -> List[float]:
    """
    Calculate Style Transfer Intensity (STI)

    Args:
        input_text (str)
        output_text (str)
        style_data (StyleAttributeData)

    Returns:
        List[float]
    """
    sti = StyleIntensityClassifier(
        model_identifier=style_data.cls_model_path,
    )
    return sti.calculate_transfer_intensity_fraction(
        string_to_list_string(input_text), string_to_list_string(output_text)
    )


@st.cache(
    hash_funcs={tokenizers.Tokenizer: lambda _: None},
    allow_output_mutation=True,
    show_spinner=False,
)
def get_cps_metric(
    input_text: str, output_text: str, style_data: StyleAttributeData
) -> List[float]:
    """
    Calculate Content Preservation Score (CPS)

    Args:
        input_text (str)
        output_text (str)
        style_data (StyleAttributeData)

    Returns:
        List[float]
    """
    cps = ContentPreservationScorer(
        cls_model_identifier=style_data.cls_model_path,
        sbert_model_identifier=style_data.sbert_model_path,
    )
    return cps.calculate_content_preservation_score(
        string_to_list_string(input_text),
        string_to_list_string(output_text),
        mask_type="none",
    )


def generate_style_transfer(
    text_sample: str,
    style_data: StyleAttributeData,
    max_gen_length: int,
    num_beams: int,
    temperature: int,
):
    """
    Run inference on seq2seq model and persist result to
    `session_state` varaible.

    Args:
        text_sample (str): _description_
        style_data (StyleAttributeData): _description_
        max_gen_length (int): _description_
        num_beams (int): _description_
        temperature (int): _description_
    """
    with st.spinner("Transferring style, hang tight!"):

        generate_kwargs = {
            "max_gen_length": max_gen_length,
            "num_beams": num_beams,
            "temperature": temperature,
        }

        st_class = StyleTransfer(
            model_identifier=style_data.seq2seq_model_path,
            **generate_kwargs,
        )

        st_result = st_class.transfer(text_sample)

    st.session_state.st_result = st_result
