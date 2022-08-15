# Exploring Intelligent Writing Assistance

A demonstration of how the NLP task of _text style transfer_ can be applied to enhance the human writing experience using [HuggingFace Transformers](https://huggingface.co/) and [Streamlit](https://streamlit.io/).

![](/static/images/app_screenshot.png)

> This repo accompanies Cloudera Fast Forward Labs' [blog series](https://blog.fastforwardlabs.com/2022/03/22/an-introduction-to-text-style-transfer.html) in which we explore the task of automatically neutralizing subjectivity bias in free text.

The goal of this application is to demonstrate how the NLP task of text style transfer can be applied to enhance the human writing experience. In this sense, we intend to peel back the curtains on how an intelligent writing assistant might function — walking through the logical steps needed to automatically re-style a piece of text (from informal-to-formal **or** subjective-to-neutral) while building up confidence in the model output.

Through the application, we emphasize the imperative for a human-in-the-loop user experience when designing natural language generation systems. We believe text style transfer has the potential to empower writers to better express themselves, but not by blindly generating text. Rather, generative models, in conjunction with interpretability methods, should be combined to help writers understand the nuances of linguistic style and suggest stylistic edits that may improve their writing.

## Project Structure

```
.
├── LICENSE
├── README.md
├── apps                                      # Code to support the Streamlit application
│   ├── app.py
│   ├── app_utils.py
│   ├── data_utils.py
│   └── visualization_utils.py
├── requirements.txt
├── scripts                                   # Utility scripts for project and application setup
│   ├── download_models.py
│   ├── install_dependencies.py
│   └── launch_app.py
├── setup.py
├── src                                       # Main library + classes used throughout the app
│   ├── __init__.py
│   ├── content_preservation.py
│   ├── style_classification.py
│   ├── style_transfer.py
│   └── transformer_interpretability.py
├── static
│   └── images
└── tests                                     # Basic testing to validate classes in src/ directory
    ├── __init__.py
    └── test_model_classes.py
```

By launching this applied machine learning prototype (AMP) on CML, the following steps will be taken to recreate the project in your workspace:

1. A Python session is run to install all project dependencies and download and cache all HuggingFace models used throughout the application
2. A Streamlit application is deployed to the project

## Launching the Project on CML

This AMP was developed against Python 3.9. There are two ways to launch the project on CML:

1. **From Prototype Catalog** - Navigate to the AMPs tab on a CML workspace, select the "Exploring Intelligent Writing Assistance" tile, click "Launch as Project", click "Configure Project"
2. **As an AMP** - In a CML workspace, click "New Project", add a Project Name, select "AMPs" as the Initial Setup option, copy in this repo URL, click "Create Project", click "Configure Project"

## Running the Project Outside of CML

The code and application within were developed against Python 3.9, and are likely also to function with more recent versions of Python.

To setup the project, first create and activate a new virtual environment through your preferred means. Then pip install dependencies and download the required HuggingFace models:

```
python -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt && python3 scripts/download_models.py
```

Finally, launch the Streamlit application:

```
streamlit run apps/app.py
```

**Note:** Since the app utilizes several large Transformer models, you'll need at least 2vCPU / 4GB RAM.

## Tests

A handful of tests are included that can be used to validate the basic initialiation and functionality of the custom classes found in the `src/` directory. These tests should be run before merging any changes to the repo by running the `pytest` command from the project root directory.
