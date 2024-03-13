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

   To install the required dependencies, run the following command:

   ```bash
   poetry install
   ```
   or
   ```bash
   pip install -r requirements.txt
   ```

   Annomatic relies on Haystack 2.0 for its core functionality. Therefore,
   certain models rely on extra dependencies. The following optional
    dependencies are required for the following models:

    1. HuggingFace
    ```bash
    poetry install --with huggingface
    ```

A list for all available LLMs can be found in the [Haystack docs](https://docs.haystack.deepset.ai/docs/generators)
with additional installation information.

## Usage
   You can find examples of how to use Annomatic in the [Examples](examples)
   directory.
## Contributing

## License
