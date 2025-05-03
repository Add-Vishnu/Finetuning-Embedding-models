from sentence_transformers import SentenceTransformer
from typing import Union, List, Any, Optional
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.finetuning import SentenceTransformersFinetuneEngine
import os

class Embedding:
    """
    Embedding class for loading and using pre-trained and finetuned embedding models for text embeddings.

    Parameters:
    - model_name (str): The name of the pre-trained embedding model or the path of a finetuned model.

    Methods:
    - get_embedding(texts: Union[str, List[str]])
        Returns the embedding(s) for the input text(s).

    - finetune(model_name: str, train_dataset: Union[str, dict], val_dataset: Union[str, dict],
               batch_size: int = 10, epochs: int = 2, loss: Any = None,
               evaluation_steps: int = 50, output_model_path: str = None)
        Finetunes the specified model using the provided training and validation datasets.

        Parameters:
        - model_name (str): The name or path of the pre-trained sentence transformer model to finetune.
        - train_dataset (Union[str, dict]): The training dataset in the format:
            {'queries': train_queries, 'corpus': corpus, 'relevant_docs': train_relevant_docs}
            If providing file paths, use the from_json method of EmbeddingQAFinetuneDataset.
        - val_dataset (Union[str, dict]): The validation dataset in the same format as the training dataset.
            If providing file paths, use the from_json method of EmbeddingQAFinetuneDataset.
        - batch_size (int): The batch size for training (default: 10).
        - epochs (int): The number of training epochs (default: 2).
        - loss (Any): The loss function to use during training (default: None).
        - evaluation_steps (int): Frequency of evaluation during training (default: 50).
        - output_model_path (str): Path to save the finetuned model (default: None).

        Returns:
        - str: A message indicating the success or failure of the finetuning process.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def get_embedding(self, texts: Union[str, List[str]]) -> Union[List[List[float]], List[float]]:
        """
        Returns the embedding(s) for the input text(s).

        Parameters:
        - texts (Union[str, List[str]]): The input text or list of texts to get embeddings for.

        Returns:
        - Union[List[List[float]], List[float]]: List of embeddings for each input text.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts)

        if len(texts) == 1:
            return embeddings[0]
        else:
            return embeddings

    def finetune(self, model_name : str, train_dataset : str, val_dataset: str,
                 batch_size : int = 10, epochs : int =2, loss : Any|None = None,
                 evaluation_steps: int = 50, output_model_path : str = None):
      """
      Finetunes the specified model using the provided training and validation datasets.

      Parameters:
      - model_name (str): The name or path of the pre-trained sentence transformer model to finetune.
      - train_dataset (Union[str, dict]): The training dataset in the format:
          {'queries': train_queries, 'corpus': corpus, 'relevant_docs': train_relevant_docs}
          If providing file paths, use the from_json method of EmbeddingQAFinetuneDataset.
      - val_dataset (Union[str, dict]): The validation dataset in the same format as the training dataset.
          If providing file paths, use the from_json method of EmbeddingQAFinetuneDataset.
      - batch_size (int): The batch size for training (default: 10).
      - epochs (int): The number of training epochs (default: 2).
      - loss (Any): The loss function to use during training (default: None).
      - evaluation_steps (int): Frequency of evaluation during training (default: 50).
      - output_model_path (str): Path to save the finetuned model (default: None).

      """
      try:
        print(train_dataset)
        if isinstance(train_dataset,str) and isinstance(val_dataset,str):
          train_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset)
          val_dataset = EmbeddingQAFinetuneDataset.from_json(val_dataset)
      except:
        print("""Check the datasets. The datasets should be in this format
          train_dataset = { 'queries' : train_queries , 'corpus' : corpus, 'relevant_docs' : train_relevant_docs, }

          val_dataset = { 'queries' : val_queries , 'corpus' : corpus, 'relevant_docs' : val_relevant_docs, }       
          """)

      
      model_finetuned_bool = False
      try:
        if output_model_path is None:
            cwd = os.getcwd()
            output_model_path = f"src/{model_name}_fine_tuned_model"
        finetune_engine = SentenceTransformersFinetuneEngine(
            train_dataset,
            model_id=model_name,
            model_output_path=output_model_path,
            val_dataset=val_dataset,
            batch_size=batch_size,
            epochs=epochs,
            evaluation_steps=evaluation_steps,
            loss=loss
        )
        finetune_engine.finetune()
        model_finetuned_bool = True
      except:
        print(f"{model_name} cannot be recognized. Enter the correct path/name of the embedding file.")
      if model_finetuned_bool:
        print(f"The {model_name} is finetuned and stored in the '{output_model_path}_fine_tuned_model'.")
          
