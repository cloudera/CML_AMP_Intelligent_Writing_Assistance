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
from transformers import pipeline


class StyleTransfer:
    """
    Model wrapper for a Text2TextGeneration pipeline used to transfer a style attribute on a given piece of text.

    Attributes:
        model_identifier (str) - Path to the model that will be used by the pipeline to make predictions
        max_gen_length (int) - Upper limit on number of tokens the model can generate as output

    """

    def __init__(
        self,
        model_identifier: str,
        max_gen_length: int = 200,
        num_beams=4,
        temperature=1,
    ):
        self.model_identifier = model_identifier
        self.max_gen_length = max_gen_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        self._build_pipeline()

    def _build_pipeline(self):

        self.pipeline = pipeline(
            task="text2text-generation",
            model=self.model_identifier,
            device=self.device,
            max_length=self.max_gen_length,
            num_beams=self.num_beams,
            temperature=self.temperature,
        )

    def transfer(self, input_text: Union[str, List[str]]) -> List[str]:
        """
        Transfer the style attribute on a given piece of text using the
        initialized `model_identifier`.

        Args:
            input_text (`str` or `List[str]`) - Input text for style transfer

        Returns:
            generated_text (`List[str]`) - The generated text outputs

        """
        return [item["generated_text"] for item in self.pipeline(input_text)]
