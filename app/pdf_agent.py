import os
from pathlib import Path
from typing import List, Dict

import torch.nn.functional as F

from PyPDF2 import PdfReader, PdfWriter
from unstructured.partition.pdf import partition_pdf
from app.models import Embedder
from typing import Callable
from typing import List

from unstructured.partition.pdf import partition_pdf

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
from app.db import ChromaStore
from app.common_agent import ToolResult, Tool
from app.chat import ChatGenerator

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

class PDFSplitter:
    def __init__(self, pages_per_part: int = 20):
        self.pages_per_part = pages_per_part

    def split(self, pdf_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        outputs = []

        for start in range(0, total_pages, self.pages_per_part):
            end = min(start + self.pages_per_part, total_pages)

            writer = PdfWriter()
            for page_idx in range(start, end):
                writer.add_page(reader.pages[page_idx])

            out_path = Path(output_dir) / f"{Path(pdf_path).stem}_p{start+1:04d}-{end:04d}.pdf"
            with open(out_path, "wb") as f:
                writer.write(f)

            # store 1-based original page offset
            outputs.append({
                "path": str(out_path),
                "start_page": start + 1
            })

        return outputs

class PDFPartitioner:
    def partition(self, pdf_path: str) -> List[Dict]:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=1800,
            new_after_n_chars=1200,
            combine_text_under_n_chars=300,
        )

        chunks = []
        for el in elements:
            meta = el.metadata.to_dict()
            text = meta.get("text_as_html", el.text)

            if not text or not text.strip():
                continue

            chunks.append({
                "text": text.strip(),
                "page": meta.get("page_number"),
                "type": getattr(el, "category", None),
            })

        return chunks

class PDFIngestor:
    def __init__(
        self,
        splitter: PDFSplitter,
        partitioner: PDFPartitioner,
        embedder: Embedder,
        store: ChromaStore,
    ):
        self.splitter = splitter
        self.partitioner = partitioner
        self.embedder = embedder
        self.store = store
        self.global_chunk_id = 0

    def ingest(
        self,
        pdf_path: str,
        source_name: str,
        split_dir: str,
        preprocess_text: Callable[[str], str]
    ):
        part_pdfs = self.splitter.split(pdf_path, split_dir)
        print(f"Split into {len(part_pdfs)} PDF parts")

        for idx, part in enumerate(part_pdfs, start=1):
            part_pdf = part["path"]
            part_start_page = part["start_page"]

            chunks = self.partitioner.partition(part_pdf)
            if not chunks:
                continue

            texts = []
            metadatas = []

            for c in chunks:
                chunk_page = c["page"]
                original_page = (
                    part_start_page + chunk_page - 1
                    if chunk_page is not None
                    else None
                )

                texts.append(
                    f"[{Path(part_pdf).name} | Page {original_page}] {preprocess_text(c['text'])}"
                )

                metadatas.append({
                    "source": source_name,
                    "part_pdf": Path(part_pdf).name,
                    "page": original_page,
                    "type": c["type"],
                })

            ids = [
                f"{source_name}_{self.global_chunk_id + i}"
                for i in range(len(chunks))
            ]

            embeddings = self.embedder.embed(texts)

            self.store.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )

            self.global_chunk_id += len(chunks)

            print(f"[Part {idx}/{len(part_pdfs)}] Ingested {len(chunks)} chunks")


class CrossEncoderReranker:
    def __init__(self, model_name: str, top_k: int = 5):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        documents: List[str],
        metadatas: List[Dict],
    ) -> List[Tuple[float, str, Dict]]:
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)

        # 🔑 Handle NLI models
        if scores.ndim == 2:
            # entailment is index 2
            scores = scores[:, 2]

        ranked = sorted(
            zip(scores, documents, metadatas),
            key=lambda x: x[0],
            reverse=True,
        )

        return ranked[: self.top_k]


class RAGRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: ChromaStore,
        reranker: CrossEncoderReranker,
        recall_k: int = 20,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.recall_k = recall_k

    def search(self, query: str):
        # 1️⃣ Embed query
        q_emb = self.embedder.embed([query])

        # 2️⃣ High-recall vector search
        docs, metas = self.vector_store.search(
            query_embedding=q_emb,
            top_k=self.recall_k,
        )
        print(docs)

        # 3️⃣ Rerank for answer relevance
        ranked = self.reranker.rerank(
            query=query,
            documents=docs,
            metadatas=metas,
        )

        return ranked
    
class PDFReaderTool(Tool):
    def __init__(self):
        super().__init__(
            name="pdf_read",
            func="""
            Read a PDF document sequentially without ingesting it into the vector database.
            Use this tool when the full document must be read in order and the task doesn't need semantic search

            Arguments:
            `pdf_path`: Relative path to the PDF file to read (string)

            Output: Plain text from the document
            """
        )

    def execute(self, pdf_path: str) -> ToolResult:
        reader = PdfReader(pdf_path)

        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(f"[Page {i}]\n{text.strip()}")

        full_text = "\n\n".join(pages)

        if len(full_text) > 10_000:
            result = {
                "status": "failed",
                "content": "PDF reading is inefficient when the PDF is too long",
                "num_pages": len(pages)
            }
        else:
            result = {
                "status": "success",
                "content": full_text,
                "num_pages": len(pages),
            }

        return ToolResult(
            result=result,
            artifact_path=None
        )


class PDFSplitterTool(Tool):
    def __init__(self):
        super().__init__(
            name="pdf_split",
            func="""
            Split a PDF document into smaller PDF files by page count.
            Use this tool when 
            1. A PDF is too large to process as a single file and you need absolutely everything from the resources.
            2. You need absolutely everything in the resource

            Arguments:
            `pdf_path`: Relative path to the input PDF file (string)
            `output_dir`: Directory where split PDF files will be written (string)

            Output:
            - A list of file paths to the split PDF documents
            - Each split preserves original page order
            """
        )
        self.splitter = PDFSplitter(pages_per_part=1)

    def execute(
        self,
        pdf_path: str,
        output_dir: str,
    ) -> ToolResult:

        splits = self.splitter.split(
            pdf_path=pdf_path,
            output_dir=output_dir,
        )

        pdf_split_paths = [part["path"] for part in splits]

        return ToolResult(
            result={
                "status": "success",
                "pdf_split_paths": pdf_split_paths,
                "num_splits": len(pdf_split_paths),
                "original_pdf": pdf_path,
            },
            artifact_path=None
        )


class PDFIngestorTool(Tool):
    def __init__(self, ingestor: PDFIngestor):
        super().__init__(
            name="pdf_ingest",
            func="""
            Ingest a PDF document into the vector database for retrieval-augmented generation.
            
            Use this tool when:
            - A new PDF document must be made searchable
            - The document has NOT been ingested yet
            - You need some section of the book so a search from a RAG will be nice
            - This tool does not need PDF split to happen first and can be used immediately
            Arguments:
            `pdf_path`: relative path to the PDF file to ingest (string)
            `source_name`: Unique identifier for the document (string). 
                           Examples: "aapl_2019", "harry_potter_1"
            `split_dir`: Directory where intermediate split PDFs will be written (string)
            """
        )
        self.ingestor = ingestor

    def execute(
        self,
        pdf_path: str,
        source_name: str,
        split_dir: str,
    ) -> ToolResult:

        self.ingestor.ingest(
            pdf_path=pdf_path,
            source_name=source_name,
            split_dir=split_dir,
            preprocess_text=lambda x: x,
        )

        return ToolResult(
            result={
                "status": "success",
                "pdf_path": pdf_path,
                "source": source_name
            },
            artifact_path=None
        )

class QueryAnswererTool(Tool):
    def __init__(
        self,
        rag: RAGRetriever,
        generator: ChatGenerator,
    ):
        super().__init__(
            name="answer_question",
            func="""
            This tool perform a query on the question against a RAG

            Arguments: `question` - The question to be asked to the RAG
            """
        )
        self.rag = rag
        self.generator = generator

    def execute(self, question: str) -> ToolResult:
        # Run baseline RAG
        docs = self.rag.search(question)

        raw_docs = []
        for score, doc, meta in docs:
            raw_docs.append(doc)
        context = "\n".join(raw_docs)

        prompt = f"""
            An answer is valid ONLY IF:
            - The answer appears in the context, or
            - The context contains a clear, explicit statement in which you can strongly infer to answer the question.

            Example:
            (answer from the context)
            Context: The Aurora Project was led by Dr. Elena Markov. It officially launched in 2019 and focused on polar climate research.
            Question: Who led the Aurora Project?
            Correct answer: "Dr. Elena Markov"

            (no common knowledge)
            Context: The city of Norhaven was founded in 1820. It later became an important trading port. 
            Question: In which country is Norhaven located?
            Correct answer: "none"

            Context:
            {context}

            Question:
            {question}
        """
        messages = [
            {"role": "user", "content": prompt }
        ]

        thinking, answer = self.generator.generate(messages)

        return ToolResult(
            result={
                "status": "success",
                "answer": answer,
                "question": question
            },
            artifact_path=None
        )
