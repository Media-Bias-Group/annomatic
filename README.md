# Annomatic

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Annomatic is a powerful framework for automating the process of text annotation
using Language Models (LLMs). This tool simplifies and streamlines the pipeline
for annotating text data with the help of state-of-the-art language models,
making it easier for researchers and developers to leverage the power of LLMs
for their projects.

## Features

- Automatic text annotation via Language Models.
- Streamlined annotation pipeline.
- Easy-to-use interface.
- Highly customizable annotation settings.
- Supports a wide range of language models.

## Installation

To get started with Annomatic, follow these installation steps:

1. Clone the Annomatic repository to your local machine:

   ```bash
   git clone https://github.com/mandlc/annomatic.git

2. Install dependencies

   The installation depends on what Language Models you choose:

    1. OpenAI
    ```bash
    poetry install --with openai
    ```

    2. HuggingFace
    ```bash
    poetry install --with huggingface
    ```
   additional to this be sure that you have installed the right version of
   pytorch.

    2. vLLM
    ```bash
    poetry install --with vllm
    ```

## Usage
   You can find examples of how to use Annomatic in the [Examples](examples)
   directory.
## Contributing

## License
