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

import pytest
import transformers

from src.style_transfer import StyleTransfer
from src.style_classification import StyleIntensityClassifier
from src.content_preservation import ContentPreservationScorer
from src.transformer_interpretability import InterpretTransformer


@pytest.fixture
def subjectivity_example_data():
    examples = [
        """there is an iconic roadhouse, named "spud's roadhouse", which sells fuel and general shop items , has great meals and has accommodation.""",
        "chemical abstracts service (cas), a prominent division of the american chemical society, is the world's leading source of chemical information.",
        "the most serious scandal was the iran-contra affair.",
        "another strikingly elegant four-door saloon for the s3 continental came from james young.",
        "other ambassadors also sent their messages of condolence following her passing.",
    ]

    ground_truth = [
        'there is a roadhouse, named "spud\'s roadhouse", which sells fuel and general shop items and has accommodation.',
        "chemical abstracts service (cas), a division of the american chemical society, is a source of chemical information.",
        "one controversy was the iran-contra affair.",
        "another four-door saloon for the s3 continental came from james young.",
        "other ambassadors also sent their messages of condolence following her death.",
    ]

    return {"examples": examples, "ground_truth": ground_truth}


@pytest.fixture
def subjectivity_styletransfer():
    MODEL_PATH = "cffl/bart-base-styletransfer-subjective-to-neutral"
    return StyleTransfer(model_identifier=MODEL_PATH, max_gen_length=200)


@pytest.fixture
def subjectivity_styleintensityclassifier():
    CLS_MODEL_PATH = "cffl/bert-base-styleclassification-subjective-neutral"
    return StyleIntensityClassifier(model_identifier=CLS_MODEL_PATH)


@pytest.fixture
def subjectivity_contentpreservationscorer():
    CLS_MODEL_PATH = "cffl/bert-base-styleclassification-subjective-neutral"
    SBERT_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
    return ContentPreservationScorer(
        cls_model_identifier=CLS_MODEL_PATH, sbert_model_identifier=SBERT_MODEL_PATH
    )


@pytest.fixture
def subjectivity_interprettransformer():
    CLS_MODEL_PATH = "cffl/bert-base-styleclassification-subjective-neutral"
    return InterpretTransformer(cls_model_identifier=CLS_MODEL_PATH)


# test class initialization
def test_StyleTransfer_init(subjectivity_styletransfer):
    assert isinstance(
        subjectivity_styletransfer.pipeline,
        transformers.pipelines.text2text_generation.Text2TextGenerationPipeline,
    )


def test_StyleIntensityClassifier_init(subjectivity_styleintensityclassifier):
    assert isinstance(
        subjectivity_styleintensityclassifier.pipeline,
        transformers.pipelines.text_classification.TextClassificationPipeline,
    )


def test_ContentPreservationScorer_init(subjectivity_contentpreservationscorer):
    assert isinstance(
        subjectivity_contentpreservationscorer.cls_model,
        transformers.models.bert.modeling_bert.BertForSequenceClassification,
    )
    assert isinstance(
        subjectivity_contentpreservationscorer.sbert_model,
        transformers.models.bert.modeling_bert.BertModel,
    )


def test_InterpretTransformer_init(subjectivity_interprettransformer):
    assert isinstance(
        subjectivity_interprettransformer.cls_model,
        transformers.models.bert.modeling_bert.BertForSequenceClassification,
    )


# test class functionality
def test_StyleTransfer_transfer(subjectivity_styletransfer, subjectivity_example_data):
    assert subjectivity_example_data[
        "ground_truth"
    ] == subjectivity_styletransfer.transfer(subjectivity_example_data["examples"])


def test_StyleIntensityClassifier_calculate_transfer_intensity_fraction(
    subjectivity_styleintensityclassifier, subjectivity_example_data
):
    sti_frac = (
        subjectivity_styleintensityclassifier.calculate_transfer_intensity_fraction(
            input_text=subjectivity_example_data["examples"],
            output_text=subjectivity_example_data["ground_truth"],
        )
    )
    assert sti_frac == [
        0.9891820847234861,
        0.9808499743983614,
        0.8070009460737938,
        0.9913705583756346,
        0.9611679711017459,
    ]


def test_ContentPreservationScorer_calculate_content_preservation_score(
    subjectivity_contentpreservationscorer, subjectivity_example_data
):
    cps = subjectivity_contentpreservationscorer.calculate_content_preservation_score(
        input_text=subjectivity_example_data["examples"],
        output_text=subjectivity_example_data["ground_truth"],
        mask_type="none",
    )
    assert cps == [0.9369, 0.9856, 0.7328, 0.9718, 0.9709]
