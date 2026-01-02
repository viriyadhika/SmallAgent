# Standard library
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Third-party libraries
import fitz  # PyMuPDF
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import CrossEncoder
from ultralytics import YOLO
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection,
)
from unstructured.partition.pdf import partition_pdf

# Internal application imports
from app.models import Embedder
from app.db import ChromaStore
from app.common_agent import Tool, ToolResult
from app.chat import ChatGenerator
import cv2
import clip
import requests
from transformers import AutoTokenizer, RobertaForSequenceClassification


# Local modules
import app.utils.postprocess as postprocess

class PDFTableExtractor:
    def __init__(
        self,
        dpi: int = 300,
        conf_threshold: float = 0.5,
        padding_pdf: int = 10,
        device: str = "cpu",
        model_dir: Path = Path("data/models"),
        roberta_path: Path = Path("data/roberta_figure_diff.pt")
    ):
        self.dpi = dpi
        self.device = device
        self.conf_threshold = conf_threshold
        self.padding_pdf = padding_pdf
        self.clip, self.clip_preprocess = clip.load("ViT-B/32")

        # -----------------------
        # Load YOLO document-layout model
        # -----------------------
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)

        url = "https://github.com/moured/YOLOv11-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov11x_best.pt"
        yolo_path = "data/models/yolov11x_best.pt"

        os.makedirs("data", exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(yolo_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded to {yolo_path}")

        # yolo_path = hf_hub_download(
        #     repo_id="Armaggheddon/yolo11-document-layout",
        #     filename="yolo11n_doc_layout.pt",
        #     repo_type="model",
        #     local_dir=model_dir,
        # )
        self.yolo = YOLO(yolo_path)

        # -----------------------
        # Load Table Transformer
        # -----------------------
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        self.tatr = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        ).eval()

        # -----------------------
        # TATR class config
        # -----------------------
        self.structure_class_names = [
            "table",
            "table column",
            "table row",
            "table column header",
            "table projected row header",
            "table spanning cell",
            "no object",
        ]
        self.structure_class_map = {
            k: v for v, k in enumerate(self.structure_class_names)
        }
        self.structure_class_thresholds = {
            "table": 0.5,
            "table column": 0.5,
            "table row": 0.5,
            "table column header": 0.5,
            "table projected row header": 0.5,
            "table spanning cell": 0.5,
            "no object": 10,
        }

        self.reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            # model_name="cross-encoder/nli-deberta-v3-base",
            top_k=3
        )

        self.class_model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=2)
        self.class_model.classifier.load_state_dict(torch.load(roberta_path, map_location=self.device))
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.class_model.to(self.device)
        self.class_model.eval()

    def _is_good_caption(self, text: str, threshold=0.5) -> bool:
        """
        Returns True if classifier predicts this text is a useful caption.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            logits = self.class_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            print(probs)

        # label 1 = positive
        return probs[0, 1].item() >= threshold
    
    def extract(self, page):
        def bbox_iou(b1, b2):
            x0 = max(b1[0], b2[0])
            y0 = max(b1[1], b2[1])
            x1 = min(b1[2], b2[2])
            y1 = min(b1[3], b2[3])
            inter_w = max(0, x1 - x0)
            inter_h = max(0, y1 - y0)
            inter_area = inter_w * inter_h
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = area1 + area2 - inter_area
            return inter_area / union if union > 0 else 0.0
        
        def get_pdf_text_blocks(page, table_bboxes_pdf, iou_threshold=0.01):
            """
            Returns text blocks in PDF coordinates,
            excluding blocks that overlap detected tables.

            table_bboxes_pdf: List[(x0, y0, x1, y1)] in PDF coords
            """
            blocks = []

            for b in page.get_text("blocks"):
                x0, y0, x1, y1, text, *_ = b
                if not text.strip():
                    continue
                
                block_bbox = (x0, y0, x1, y1)

                # ❌ skip blocks inside tables
                inside_table = False
                for table_bbox in table_bboxes_pdf:
                    if bbox_iou(block_bbox, table_bbox) > iou_threshold:
                        inside_table = True
                        break
                    
                if inside_table:
                    continue
                
                blocks.append({
                    "bbox": block_bbox,
                    "text": text.strip()
                })

            return blocks
        
        """
        image: PIL.Image (cropped figure/table)
        text_blocks: list of { "bbox": (x0,y0,x1,y1), "text": str }
        """
        # --------------------------------------------------
        # Helpers
        # --------------------------------------------------
        def bbox_center(b):
            x0, y0, x1, y1 = b
            return ((x0 + x1) / 2, (y0 + y1) / 2)
        
        def center_distance(b1, b2):
            c1 = bbox_center(b1)
            c2 = bbox_center(b2)
            return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        
        def find_caption_for_picture(
            target_bbox,
            image,
            text_blocks,
            clip_model,
            clip_preprocess,
            k_spatial=4,
        ):
            clip_model = clip_model.to(self.device)

            # ---- image embedding ----
            with torch.no_grad():
                image_tensor = clip_preprocess(image).unsqueeze(0).to(self.device)
                image_emb = clip_model.encode_image(image_tensor)
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                image_emb = image_emb.squeeze(0)  # (D,)

            # ---- spatial filtering ----
            candidates = []
            for tb in text_blocks:
                if bbox_iou(target_bbox, tb["bbox"]) > 0:
                    continue
                d = center_distance(target_bbox, tb["bbox"])
                candidates.append((d, tb))

            candidates.sort(key=lambda x: x[0])
            candidates = [tb for _, tb in candidates[:k_spatial]]

            if not candidates:
                return None

            # ---- CLIP text embeddings ----
            texts = [tb["text"][:300] for tb in candidates]
            tokens = clip.tokenize(texts, truncate=True).to(self.device)

            with torch.no_grad():
                text_embs = clip_model.encode_text(tokens)
                text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)  # (N, D)

            # ---- similarity (pure CLIP) ----
            sims = (text_embs @ image_emb).cpu().tolist()

            # ---- sort by CLIP similarity ----
            ranked = sorted(
                zip(candidates, sims),
                key=lambda x: x[1],
                reverse=True,
            )

            # ---- debug output (recommended) ----
            print("\n=============== FIGURE CAPTION SCORING (CLIP ONLY) ===============")
            for i, (tb, score) in enumerate(zip(candidates, sims), start=1):
                print(f"Candidate {i}")
                print("Text:")
                print(tb["text"])
                print(f"CLIP similarity : {float(score):.4f}")
                print("-" * 60)
            print("=================================================================\n")

            for (tb, score) in ranked[:k_spatial]:
                if self._is_good_caption(tb["text"]):
                    return tb
            
            return None
        
        def find_caption_for_table(
            target_bbox,
            table_rect,
            page,
            text_blocks,
            reranker,
            dpi,
            k_spatial=4,
        ):
            # ---- extract table markdown FIRST ----
            table_md = self._extract_table_markdown_for_caption(page, table_rect)

            if not table_md:
                return None

            query = (
                "Which sentence is the title or caption describing the following table?\n\n"
                + table_md
            )

            # ---- spatial candidate generation ----
            candidates = []
            for tb in text_blocks:
                if bbox_iou(target_bbox, tb["bbox"]) > 0:
                    continue
                d = center_distance(target_bbox, tb["bbox"])
                candidates.append((d, tb))

            candidates.sort(key=lambda x: x[0])
            candidates = [tb for _, tb in candidates[:k_spatial]]

            if not candidates:
                return None

            documents = [tb["text"] for tb in candidates]

            ranked = reranker.rerank(
                query=query,
                documents=documents,
                metadatas=candidates,
            )

            # ---- debug ----
            print("\n=========== TABLE CAPTION (MARKDOWN QUERY) ===========")
            print("QUERY:")
            print(query[:800], "\n")

            for i, (score, text, _) in enumerate(ranked, 1):
                print(f"Rank {i} | Score {score:.4f}")
                print(text)
                print("-" * 60)

            print("=====================================================\n")

            # ---- classifier gate ----
            for score, text, tb in ranked[:k_spatial]:
                if self._is_good_caption(text):
                    return tb

            return None

        
        def pdf_bbox_to_img_bbox(bbox_pdf, scale_x, scale_y):
            x0, y0, x1, y1 = bbox_pdf
            return (
                int(x0 / scale_x),
                int(y0 / scale_y),
                int(x1 / scale_x),
                int(y1 / scale_y),
            )

        results = []
        viz_items = []

        # ---- Render page to pixmap ----
        page_pix = page.get_pixmap(dpi=self.dpi)
        page_img = Image.frombytes(
            "RGB",
            (page_pix.width, page_pix.height),
            page_pix.samples
        )
        img = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)

        detections = self.yolo(img, conf=self.conf_threshold)[0]

        scale_x = page.rect.width / page_pix.width
        scale_y = page.rect.height / page_pix.height

        # ---- Extract all PDF text blocks once ----


        table_picture_bboxes = []
        table_picture_boxes = []
        for box in detections.boxes:
            cls_name = self.yolo.names[int(box.cls)]
            if cls_name not in {"Table", "Picture"}:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            bbox_pdf = (
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            )

            table_picture_bboxes.append(bbox_pdf)
            table_picture_boxes.append(box)

        pdf_text_blocks = get_pdf_text_blocks(page, table_picture_bboxes)

        for box in table_picture_boxes:
            cls_id = int(box.cls)
            cls_name = self.yolo.names[cls_id]
            conf = float(box.conf)

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Convert to PDF coordinates
            bbox_pdf = (
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            )

            pdf_rect = fitz.Rect(*bbox_pdf)
            cropped_pixmap = page.get_pixmap(clip=pdf_rect, dpi=self.dpi)
            cropped_img = Image.frombytes(
                "RGB",
                (cropped_pixmap.width, cropped_pixmap.height),
                cropped_pixmap.samples
            )

            if cls_name == "Table":
                caption_block = find_caption_for_table(
                    target_bbox=bbox_pdf,
                    table_rect=pdf_rect,
                    page=page,
                    text_blocks=pdf_text_blocks,
                    reranker=self.reranker,
                    dpi=self.dpi,
                )

                extracted_text = caption_block["text"] if caption_block is not None else ""

                # ---- Draw caption bounding box ----
                cap_x0, cap_y0, cap_x1, cap_y1 = pdf_bbox_to_img_bbox(
                    caption_block["bbox"],
                    scale_x,
                    scale_y
                ) if caption_block is not None else (0,0,0,0)

                caption_bbox = (cap_x0, cap_y0, cap_x1, cap_y1)
                results.append(self._process_table(page, box.xyxy[0].tolist(), scale_x, scale_y, extracted_text, caption_bbox))
            if cls_name == "Picture":
                caption_block = find_caption_for_picture(
                    target_bbox=bbox_pdf,
                    image=cropped_img,
                    text_blocks=pdf_text_blocks,
                    clip_model=self.clip,
                    clip_preprocess=self.clip_preprocess,
                )

                extracted_text = caption_block["text"] if caption_block is not None else ""

                # ---- Draw caption bounding box ----
                cap_x0, cap_y0, cap_x1, cap_y1 = pdf_bbox_to_img_bbox(
                    caption_block["bbox"],
                    scale_x,
                    scale_y
                ) if caption_block is not None else (0,0,0,0)

                caption_bbox = (cap_x0, cap_y0, cap_x1, cap_y1)
                results.append({
                    "image": cropped_img,
                    "confidence": conf,
                    "caption": extracted_text,
                    "caption_bbox": caption_bbox
                })

            viz_items.append({
                "figure_bbox_pdf": bbox_pdf,
                "figure_type": cls_name,
                "caption_bbox_pdf": caption_block["bbox"] if caption_block is not None else None,
                "caption_text": extracted_text,
                "scale_x": scale_x,
                "scale_y": scale_y,
            })

            # ---- Visualization ----

        return results, viz_items
    
    def _extract_table_markdown_for_caption(self, page, table_rect):
        table_pix = page.get_pixmap(clip=table_rect, dpi=self.dpi)
        table_img = Image.frombytes(
            "RGB", (table_pix.width, table_pix.height), table_pix.samples
        )

        page_tokens = self._build_page_tokens(page, table_rect, table_pix)

        # Run TATR
        _, cells, _ = self._run_tatr(table_img, page_tokens)

        if not cells:
            return None

        markdown = self.cells_to_markdown(cells)

        return markdown

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _process_table(self, page, bbox, scale_x, scale_y, caption, caption_bbox):
        x1, y1, x2, y2 = bbox

        pdf_rect = fitz.Rect(
            max(0, x1 * scale_x - self.padding_pdf),
            max(0, y1 * scale_y - self.padding_pdf),
            min(page.rect.width, x2 * scale_x + self.padding_pdf),
            min(page.rect.height, y2 * scale_y + self.padding_pdf),
        )

        # Crop table image (PDF → image)
        table_pix = page.get_pixmap(clip=pdf_rect, dpi=self.dpi)
        table_img = Image.frombytes(
            "RGB", (table_pix.width, table_pix.height), table_pix.samples
        )

        # Build page_tokens
        page_tokens = self._build_page_tokens(page, pdf_rect, table_pix)

        # Run TATR
        _, cells, confidence = self._run_tatr(table_img, page_tokens)

        # Convert to HTML
        markdown = self.cells_to_markdown(cells)

        return {
            "image": table_img,
            "cells": cells,
            "confidence": confidence,
            "markdown": markdown,
            "caption": caption,
            "caption_bbox": caption_bbox
        }

    def _build_page_tokens(self, page, table_rect, table_pix):
        words = page.get_text("words", clip=table_rect)

        scale_x = table_pix.width / table_rect.width
        scale_y = table_pix.height / table_rect.height

        tokens = []
        for w in words:
            x0, y0, x1, y1, text, block, line, word = w

            x0 = (x0 - table_rect.x0) * scale_x
            x1 = (x1 - table_rect.x0) * scale_x
            y0 = (y0 - table_rect.y0) * scale_y
            y1 = (y1 - table_rect.y0) * scale_y

            tokens.append(
                {
                    "bbox": [x0, y0, x1, y1],
                    "text": text,
                    "block_num": int(block),
                    "line_num": int(line),
                    "span_num": int(word),
                }
            )
        return tokens

    def _run_tatr(self, image: Image.Image, page_tokens):
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.tatr(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        det = self.processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes
        )[0]

        bboxes, scores, labels = postprocess.apply_class_thresholds(
            det["boxes"].tolist(),
            det["labels"].tolist(),
            det["scores"].tolist(),
            self.structure_class_names,
            self.structure_class_thresholds,
        )

        table_objects = [
            {"bbox": b, "score": s, "label": l}
            for b, s, l in zip(bboxes, scores, labels)
        ]
        table = {"objects": table_objects, "page_num": 0}

        table_structures, cells, confidence = postprocess.objects_to_cells(
            table,
            table_objects,
            page_tokens,
            self.structure_class_names,
            self.structure_class_thresholds,
        )

        return table_structures, cells, confidence

    # --------------------------------------------------
    # HTML
    # --------------------------------------------------
    @staticmethod
    def cells_to_html(cells):
        max_row = max(max(c["row_nums"]) for c in cells)
        max_col = max(max(c["column_nums"]) for c in cells)

        occupied = [[False] * (max_col + 1) for _ in range(max_row + 1)]
        by_pos = {(min(c["row_nums"]), min(c["column_nums"])): c for c in cells}

        html = ["<table>"]
        for r in range(max_row + 1):
            html.append("  <tr>")
            for c in range(max_col + 1):
                if occupied[r][c]:
                    continue

                cell = by_pos.get((r, c))
                if not cell:
                    html.append("    <td></td>")
                    continue

                for rr in cell["row_nums"]:
                    for cc in cell["column_nums"]:
                        occupied[rr][cc] = True

                tag = "th" if cell.get("header") else "td"
                attrs = []
                if len(cell["row_nums"]) > 1:
                    attrs.append(f'rowspan="{len(cell["row_nums"])}"')
                if len(cell["column_nums"]) > 1:
                    attrs.append(f'colspan="{len(cell["column_nums"])}"')

                attr_str = " " + " ".join(attrs) if attrs else ""
                text = (cell.get("cell_text") or "").strip()
                html.append(f"    <{tag}{attr_str}>{text}</{tag}>")
            html.append("  </tr>")
        html.append("</table>")
        return "\n".join(html)
    
    @staticmethod
    def cells_to_markdown(cells):
        max_row = max(max(c["row_nums"]) for c in cells)
        max_col = max(max(c["column_nums"]) for c in cells)
    
        # Initialize empty table
        table = [[""] * (max_col + 1) for _ in range(max_row + 1)]
    
        # Fill table (expand row/col spans)
        for cell in cells:
            text = (cell.get("cell_text") or "").strip()
            for r in cell["row_nums"]:
                for c in cell["column_nums"]:
                    table[r][c] = text
    
        lines = []
    
        # Header row (assume first row is header if any cell has header=True)
        header_row = table[0]
        lines.append("| " + " | ".join(header_row) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")
    
        # Body rows
        for row in table[1:]:
            lines.append("| " + " | ".join(row) + " |")
    
        return "\n".join(lines)

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

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        outputs = []

        for start in range(0, total_pages, self.pages_per_part):
            end = min(start + self.pages_per_part, total_pages)

            # create a new empty PDF
            out_doc = fitz.open()

            # insert page range [start, end)
            out_doc.insert_pdf(
                doc,
                from_page=start,
                to_page=end - 1
            )

            out_path = (
                Path(output_dir)
                / f"{Path(pdf_path).stem}_p{start+1:04d}-{end:04d}.pdf"
            )

            out_doc.save(out_path)
            out_doc.close()

            outputs.append({
                "path": str(out_path),
                # store 1-based original page offset
                "start_page": start + 1,
            })

        doc.close()
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
            text = el.text

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
        self.pdf_table_extractor = PDFTableExtractor()

    def execute(self, pdf_path: str) -> ToolResult:
        doc = fitz.open(pdf_path)

        pages_output = []

        for page_idx, page in enumerate(doc, start=1):
            # -------- Page text --------
            page_text = page.get_text("text") or ""
            page_parts = [f"[Page {page_idx}]\n{page_text.strip()}"]

            # -------- Tables --------
            tables, _ = self.pdf_table_extractor.extract(page)

            for table in tables:
                html = table.get("markdown", "").strip()
                caption = table.get("caption", "").strip()
                if html:
                    page_parts.append(
                        f"\n[Table]\n{html} \n[Description]\n {caption} \n"
                    )

            pages_output.append("\n".join(page_parts))

        full_text = "\n\n".join(pages_output)

        if len(full_text) > 10_000:
            result = {
                "status": "failed",
                "content": "PDF reading is inefficient when the PDF is too long",
                "num_pages": len(pages_output),
            }
        else:
            result = {
                "status": "success",
                "content": full_text,
                "num_pages": len(pages_output),
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
