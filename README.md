**1. Imports and data loading:**

- Knowledge base (KB) is loaded from `"NHSRC_KB - English.csv"` and FAQs are loaded from `"All_FAQS_with_MetaData - v1.csv"`.

**2. Checking document consistency:**

- A loop iterates through FAQs documents and checks if they exist in the KB using document names and page numbers.
- Any missing documents are printed for reference.

**3. Corpus generation:**

- A dictionary, `corpus`, is created to store KB documents as key-value pairs, where the key is the document name and page number combination, and the value is the document text.
- The loop iterates through each row in the KB and populates the dictionary.

**4. Creating queries and relevant documents from FAQs:**

- `generate_queries`, takes the FAQs dataframe as input and creates two dictionaries: `queries` and `relevant_docs`.
- `queries` maps unique IDs to FAQ questions. Unique IDs are generated using uuid.
- `relevant_docs` maps query IDs to lists of document IDs (names and page numbers) mentioned in the corresponding FAQ.

**5. Splitting data into train and validation sets:**

- `train_test_split` from `sklearn.model_selection` is used to split both FAQs and corpus dictionaries into training and validation sets with a 75:25 ratio.
- Resetting indices ensures consistent indexing in the split sets.

**6. Generating query-document dictionaries for training and validation:**

- The `generate_queries` function is applied to both training and validation FAQs dataframes to create corresponding query and relevant document dictionaries.

**7. Merging data into datasets:**

- Two dictionaries, `train_dataset` and `val_dataset`, are created to store training and validation data, respectively.
- Each dictionary contains three keys: `queries`, `corpus`, and `relevant_docs`.
- The `queries` and `relevant_docs` are the dictionaries generated after applying the `generate_queries` function to the training and validation datasets.
- The `corpus` for both train_dataset and val_dataset are the same which was generated in Step 3. 

**8. Finetuning using LlamaIndex:**

- `train_dataset` and `val_dataset` are saved as JSON files.
- `EmbeddingQAFinetuneDataset` from `llama_index.finetuning` is used to load the JSON files and prepare datasets for finetuning. 
- There was a error when loading the dictionaries directly. So we must save them and load from the json file.
  I) A `SentenceTransformersFinetuneEngine` object is created with the chosen pre-trained model ("BAAI/bge-small-en"), output path ("test_model"), train_dataset(EmbeddingQAFinetuneDataset) and val_datset(EmbeddingQAFinetuneDataset).
- The engine's `finetune()` method performs the actual finetuning on the training data and validates on the validation data.
- Finally, the `get_finetuned_model()` method retrieves the finetuned model for further evaluation.

II) **Finetuning Linear adapter on top of fine-tuned embedding model**
This section aims to finetune a linear adapter on top of a previously fine-tuned embedding model. Here are the main steps:

**Import Libraries:** Import libraries, including EmbeddingAdapterFinetuneEngine from LlamaIndex, resolve_embed_model to get the pre-trained embedding model, and torch for PyTorch.

**Resolve Embedding Model:** Get the pre-trained embedding model that has been fine-tuned previously.

**Define Finetuning Engine:** Create an instance of EmbeddingAdapterFinetuneEngine, providing the training dataset, base embedding model, output path for the finetuned model, and other parameters like the optimizer class and learning rate.

**Finetune the Model:** Call the finetune method on the finetune engine to start the finetuning process.

**Evaluate the Finetuned Model:** Obtain the finetuned model and evaluate its performance on a validation dataset using the evaluate function.

III) **Finetuning Linear adapter on top of original embedding model**
This section is similar to the previous one but involves finetuning on top of the original embedding model instead of a previously fine-tuned one.

**Resolve Original Embedding Model:** Get the pre-trained embedding model without any previous finetuning.

**Define Finetuning Engine:** Create an instance of EmbeddingAdapterFinetuneEngine similar to the previous section.

**Finetune the Model:** Call the finetune method on the finetune engine.

**Evaluate the Finetuned Model:** Obtain the finetuned model and evaluate its performance on a validation dataset.

IV) **Fine-tuning a Two-Layer Adapter on top of the original embedding model**
This section introduces a more complex finetuning scenario by adding a two-layer adapter on top of the original embedding model.

**Define a Two-Layer Adapter:** Create a two-layer neural network as an adapter with specific input, hidden, and output dimensions.

**Define Finetuning Engine:** Create an instance of EmbeddingAdapterFinetuneEngine with the original embedding model, the two-layer adapter, and other parameters.

**Finetune the Model:** Call the finetune method on the finetune engine.

**Load Model from Checkpoint:** Load the model from a checkpoint in the middle of the training process.

**Evaluate the Finetuned Model:** Evaluate the performance of the finetuned model on a validation dataset.

V) **Fine-tuning a Two-Layer Adapter on top of the finetuned embedding model**
This section follows a similar approach to the previous one but involves adding a two-layer adapter on top of a previously fine-tuned embedding model.

**Define Finetuning Engine:** Create an instance of EmbeddingAdapterFinetuneEngine with the previously fine-tuned embedding model, the two-layer adapter, and other parameters.

**Finetune the Model:** Call the finetune method on the finetune engine.

**Load Model from Checkpoint:** Load the model from a checkpoint in the middle of the training process.

**Evaluate the Finetuned Model:** Evaluate the performance of the finetuned model on a validation dataset.

**9. Evaluating the fine-tuned model:**

- Two evaluation functions are defined:
    - `evaluate` uses the `ServiceContext` and `VectorStoreIndex` from `llama_index` to perform top-k retrieval for each query and calculate hit rate (whether the relevant document is retrieved).
    - `evaluate_st` uses `InformationRetrievalEvaluator` from `sentence_transformers` to calculate various metrics like MAP (Mean Average Precision) and NDCG (Normalized Discounted Cumulative Gain).
- Both functions are used to evaluate the original pre-trained model ("BAAI/bge-small-en") and the fine-tuned model on the validation set.

**10. Summary and comparison:**

- Hit rates and other metrics for both models are compared and presented in tables and charts.
- The fine-tuned model generally outperforms the original model, demonstrating the effectiveness of the finetuning process.

