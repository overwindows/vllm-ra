# vLLM Examples

This directory contains various examples demonstrating how to use vLLM for different inference scenarios and use cases.

## Basic Examples

- `llm_engine_example.py`: A basic example showing how to use the LLMEngine class directly for text generation
- `offline_inference.py`: Simple example for offline inference with vLLM
- `relay_inference.py`: Example demonstrating relay inference capabilities

## API and Client Examples

- `api_client.py`: Example client for interacting with vLLM's API
- `openai_completion_client.py`: Example showing OpenAI-compatible completion API usage
- `openai_chatcompletion_client.py`: Example demonstrating OpenAI-compatible chat completion API

## Web Interface

- `gradio_webserver.py`: A Gradio-based web interface for interacting with vLLM

## Advanced Examples

- `llm_analyzer_vllm.py`: Advanced example for analyzing LLM outputs
- `llm_analyzer_vllm_oaas.py`: LLM analyzer with OaaS (Online-as-a-Service) integration
- `llm_analyzer_vllm_oaas_async.py`: Asynchronous version of the LLM analyzer with OaaS

## Shell Scripts

- `vllm_engine.sh`: Shell script for running the vLLM engine
- `vllm_oaas_offline.sh`: Script for offline OaaS operations

## Templates

- `template_alpaca.jinja`: Template for Alpaca-style prompts
- `template_chatml.jinja`: Template for ChatML format
- `template_inkbot.jinja`: Template for Inkbot-style prompts

## Test Data

- `test_data`: Directory containing test data for examples
- `input.json`: Sample input data for testing
- `prompt.txt`: Example prompts for testing

## Usage

Each example can be run independently. Most examples include command-line arguments for configuration. For example:

```bash
python llm_engine_example.py --model <model_name> --tensor-parallel-size <size>
```

For more detailed information about specific examples, please refer to the comments and documentation within each file.

## Requirements

Make sure you have vLLM installed and configured properly before running these examples. Some examples may require additional dependencies which are listed in their respective files.

## Environment Setup

The environment has been set up using the `vllm_env.yml` file which has already been exported. This file contains all the necessary dependencies and configurations for running the vLLM examples.

To activate the environment (if not already active):

```bash
conda activate vllm_env
```

If you need to recreate the environment from the yml file:

```bash
conda env create -f vllm_env.yml
conda activate vllm_env
```

The environment includes all required packages for running the examples in this directory.
