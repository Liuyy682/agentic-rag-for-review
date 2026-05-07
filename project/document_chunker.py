import os
import glob
import difflib
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from markdown_cleaner import clean_markdown_text, write_cleaning_log

class DocumentChuncker:
    def __init__(self):
        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON, 
            strip_headers=False
        )
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE, 
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE
        self.__max_parent_size = config.MAX_PARENT_SIZE

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []

        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
        
        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path):
        doc_path = Path(md_path)

        with open(doc_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        parent_chunks = self.__create_page_aware_parent_chunks(markdown_text, doc_path)
        
        merged_parents = self.__merge_small_parents(parent_chunks)
        split_parents = self.__split_large_parents(merged_parents)
        cleaned_parents = self.__clean_small_chunks(split_parents)
        
        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, cleaned_parents, doc_path)
        return all_parent_chunks, all_child_chunks

    def __create_page_aware_parent_chunks(self, markdown_text, doc_path):
        if not config.MARKDOWN_CLEANING_ENABLED:
            return self.__parent_splitter.split_text(markdown_text)

        cleaned = clean_markdown_text(
            markdown_text,
            source_file=f"{doc_path.stem}.pdf",
            scan_lines=config.HEADER_FOOTER_SCAN_LINES,
            min_repeat_pages=config.MIN_REPEAT_PAGES,
            min_repeat_ratio=config.MIN_REPEAT_RATIO,
        )
        self.__write_cleaning_outputs(markdown_text, cleaned, doc_path)

        parent_chunks = []
        for page in cleaned.pages:
            if not page.cleaned_text.strip():
                continue

            page_chunks = self.__parent_splitter.split_text(page.cleaned_text)
            if not page_chunks:
                continue

            for chunk in page_chunks:
                chunk.metadata.update({
                    "source_file": f"{doc_path.stem}.pdf",
                    "page_number": page.page_number,
                    "page_numbers": [page.page_number] if page.page_number is not None else [],
                    "slide_title": page.slide_title,
                })
            parent_chunks.extend(page_chunks)

        return parent_chunks

    def __write_cleaning_outputs(self, raw_markdown_text, cleaned, doc_path):
        cleaned_dir = Path(config.MARKDOWN_CLEANED_DIR)
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = cleaned_dir / f"{doc_path.stem}.md"
        cleaned_path.write_text(cleaned.cleaned_text, encoding="utf-8")

        log_path = Path(config.MARKDOWN_CLEANING_LOG_DIR) / f"{doc_path.stem}.jsonl"
        write_cleaning_log(cleaned, log_path)

        diff_dir = Path(config.MARKDOWN_CLEANING_DIFF_DIR)
        diff_dir.mkdir(parents=True, exist_ok=True)
        diff_lines = difflib.unified_diff(
            raw_markdown_text.splitlines(keepends=True),
            cleaned.cleaned_text.splitlines(keepends=True),
            fromfile=f"markdown_docs/{doc_path.name}",
            tofile=f"markdown_docs_cleaned/{doc_path.name}",
        )
        diff_path = diff_dir / f"{doc_path.stem}.diff"
        diff_path.write_text("".join(diff_lines), encoding="utf-8")

    def __merge_small_parents(self, chunks):
        if not chunks:
            return []
        
        merged, current = [], None
        
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                self.__merge_metadata(current.metadata, chunk.metadata)

            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None
        
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    self.__merge_metadata(merged[-1].metadata, {k: v})
            else:
                merged.append(current)
        
        return merged

    def __split_large_parents(self, chunks):
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:
                split_chunks.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP
                )
                sub_chunks = splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)
        
        return split_chunks

    def __clean_small_chunks(self, chunks):
        cleaned = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    self.__merge_metadata(cleaned[-1].metadata, chunk.metadata)
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    self.__merge_metadata(chunks[i + 1].metadata, chunk.metadata, prepend=True)
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        
        return cleaned

    def __merge_metadata(self, target, source, prepend=False):
        for key, value in source.items():
            if key == "page_numbers":
                existing = target.get(key, [])
                if not isinstance(existing, list):
                    existing = [existing]
                incoming = value if isinstance(value, list) else [value]
                merged = incoming + existing if prepend else existing + incoming
                target[key] = [item for item in dict.fromkeys(merged) if item is not None]
            elif key == "page_number":
                existing_pages = target.get("page_numbers", [])
                if not isinstance(existing_pages, list):
                    existing_pages = [existing_pages]
                incoming_pages = [value] if value is not None else []
                merged = incoming_pages + existing_pages if prepend else existing_pages + incoming_pages
                target["page_numbers"] = [item for item in dict.fromkeys(merged) if item is not None]
                if target["page_numbers"]:
                    target[key] = target["page_numbers"][0]
            elif key == "slide_title":
                if not value:
                    continue
                existing = target.get(key)
                if not existing:
                    target[key] = value
                elif value not in str(existing).split(" -> "):
                    target[key] = f"{value} -> {existing}" if prepend else f"{existing} -> {value}"
            elif key in target and target[key] != value:
                target[key] = f"{value} -> {target[key]}" if prepend else f"{target[key]} -> {value}"
            else:
                target[key] = value

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path):
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({
                "source": str(doc_path.stem)+".pdf",
                "source_file": str(doc_path.stem)+".pdf",
                "parent_id": parent_id,
                "chunk_index": i,
            })
            
            all_parent_pairs.append((parent_id, p_chunk))
            child_chunks = self.__child_splitter.split_documents([p_chunk])
            for child_index, child_chunk in enumerate(child_chunks):
                child_chunk.metadata["chunk_id"] = f"{parent_id}_child_{child_index}"
                child_chunk.metadata["chunk_index"] = len(all_child_chunks) + child_index
            all_child_chunks.extend(child_chunks)
