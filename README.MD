# MediChat-LLM

## Overview

MediChat-LLM is an advanced chatbot designed specifically for medical professionals and enthusiasts. Utilizing state-of-the-art language models, MediChat-LLM can provide insightful, accurate, and contextually relevant responses to medical inquiries. This project is built to leverage the power of large language models, particularly the Llama 2 model, to create an intelligent assistant that can access and utilize a custom knowledge base derived from medical literature.

## Project Repository

Project repository link: [GitHub - dhairya8luthra/medichat-LLM](https://github.com/dhairya8luthra/medichat-LLM)

## Setup Instructions

### Prerequisites

- Python 3.8
- Conda (optional, but recommended for environment management)

### Installation Steps

1. **Clone the Repository**

    ```sh
    git clone https://github.com/dhairya8luthra/medichat-LLM.git
    cd medichat-LLM
    ```

2. **Create and Activate a Conda Environment**

    ```sh
    conda create -n mchatbot python=3.8 -y
    conda activate mchatbot
    ```

3. **Install Required Packages**

    ```sh
    pip install -r requirements.txt
    ```

4. **Download and Configure the Llama 2 Model**

    Download the Llama 2 model from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin) and place it in the appropriate directory as specified in the project.

5. **Set Up Environment Variables**

    Create a `.env` file in the root directory and add your API keys:

    ```env
    PINECONE_API_KEY=your_pinecone_api_key
    CHATGPT_API_KEY=your_chatgpt_api_key
    ```

### Customizing the Knowledge Base

1. **Add Medical Literature**

    Place your medical PDFs in the `data` folder.

2. **Index the Knowledge Base**

    Run the indexing script to process and store the medical literature:

    ```sh
    python store_index.py
    ```

### Final Setup

Run the main setup script to finalize the configuration:

```sh
python setup.py
```
## Start the Application
To start the application, run:

```sh
python app.py
```

## Usage Notes
If you do not wish to set up your own Pinecone and ChatGPT services, you can use the provided API keys for evaluation purposes.
Please refer to the evaluation link for further instructions.
https://docs.google.com/document/d/10zW903jVO2JCGJFZo6OjTHbzxLw0PiZCjFLKXcJNtK4/edit

## Contributing
We welcome contributions to MediChat-LLM. If you encounter any issues or have suggestions for improvements, please create an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
MediChat-LLM utilizes the following resources:

- Llama 2 model by Hugging Face
- Pinecone for vector database management
- ChatGPT API for language processing

