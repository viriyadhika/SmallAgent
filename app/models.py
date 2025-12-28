from typing import List

from unstructured.partition.pdf import partition_pdf

import torch
import torch.nn.functional as F

class Embedder:
    def __init__(self, tokenizer, model, batch_size: int = 16, max_length: int = 512):
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length

    def embed(self, texts: List[str]) -> torch.Tensor:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                emb = F.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)
