# Chroma URL Loader for LangChain

This repository features a Python script (`url_loader.py`) that demonstrates the integration of LangChain for processing data from URLs, extracting text, and establishing a Chroma vector store. The script utilizes the LangChain library for natural language processing tasks and incorporates multithreading to enhance concurrent processing.

## Requirements

- [LangChain](https://github.com/langchain-ai): LangChain is a library for natural language processing tasks, including document loading, text extraction, and vector stores.
- [Chroma](https://github.com/chroma-core/chroma): Chroma is a library for efficient similarity search and clustering of dense vectors.

## Installation

1. Install LangChain, Chroma, and other prerequisites using the following commands:

   ```bash
   pip install langchain
   pip install chroma
   pip install -r requirements.txt
   ```

2. Clone this repository:

   ```bash
   git clone https://github.com/umangpurwar03/Chroma-URL-Loader-LLM
   ```

3. Navigate to the repository directory:

   ```bash
   cd Chroma-URL-Loader-LLM
   ```

## Usage

1. Adjust the `url_list` variable in `url_loader.py` to include the URLs from which you want to extract text.

2. Execute the script:

   ```bash
   python chroma_url_loader.py
   ```

This script concurrently processes each URL using multithreading. It loads the data, extracts text, generates embeddings using Hugging Face models, and stores the vectors in a Chroma vector store.

## Customization

- Modify the `model_name` parameter in the `HuggingFaceEmbeddings` initialization to customize the model used for embeddings.

- Adjust other parameters, such as chunk size and overlap, in the `RecursiveCharacterTextSplitter` initialization based on your specific requirements.

- Customize additional parameters and configurations to align with your unique use case.

## Multithreading

The script employs multithreading to concurrently process multiple URLs. The `process_urls_in_parallel` function initiates a separate thread for each URL, enhancing overall processing efficiency. Adjust the number of threads to match your system's capabilities and needs.

## License

This code is distributed under the [MIT License](LICENSE). Feel free to utilize and modify it as per your requirements.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai)
- [Chroma](https://github.com/chroma-core/chroma)

If you find this code beneficial or have suggestions for enhancements, please contribute or open an issue.
