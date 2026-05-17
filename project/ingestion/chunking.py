import os
import glob
import difflib
import config
from collections import defaultdict
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from ingestion.cleaning import clean_markdown_text, parse_pages, write_cleaning_log
from ingestion.index_manifest import INDEX_SCHEMA_VERSION, current_index_config

class DocumentChunker:
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

    def create_chunks_single(self, md_path, page_numbers=None):
        """Create chunks for the whole document or for a page-level rebuild window."""
        doc_path = Path(md_path)

        with open(doc_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        parent_chunk_groups = self.__create_page_aware_parent_chunk_groups(markdown_text, doc_path, page_numbers)
        
        all_parent_chunks, all_child_chunks = [], []
        parent_counts_by_page = defaultdict(int)
        for parent_chunks in parent_chunk_groups:
            merged_parents = self.__merge_small_parents(parent_chunks)
            split_parents = self.__split_large_parents(merged_parents)
            cleaned_parents = self.__clean_small_chunks(split_parents)
            self.__create_child_chunks(
                all_parent_chunks,
                all_child_chunks,
                cleaned_parents,
                doc_path,
                parent_counts_by_page,
            )
        return all_parent_chunks, all_child_chunks

    def __create_page_aware_parent_chunk_groups(self, markdown_text, doc_path, page_numbers=None):
        """Split only selected pages, keeping contiguous pages together for merging."""
        selected_pages = {int(page) for page in page_numbers} if page_numbers else None
        source_file = f"{doc_path.stem}.md"
        if not config.MARKDOWN_CLEANING_ENABLED:
            pages = parse_pages(markdown_text, source_file=source_file)
            return self.__split_pages_into_parent_groups(pages, doc_path, selected_pages, use_cleaned_text=False)

        cleaned = clean_markdown_text(
            markdown_text,
            source_file=source_file,
            scan_lines=config.HEADER_FOOTER_SCAN_LINES,
            min_repeat_pages=config.MIN_REPEAT_PAGES,
            min_repeat_ratio=config.MIN_REPEAT_RATIO,
        )
        self.__write_cleaning_outputs(markdown_text, cleaned, doc_path)
        return self.__split_pages_into_parent_groups(cleaned.pages, doc_path, selected_pages, use_cleaned_text=True)

    def __split_pages_into_parent_groups(self, pages, doc_path, selected_pages, use_cleaned_text):
        page_groups = self.__contiguous_page_groups(pages, selected_pages)
        parent_chunk_groups = []

        for page_group in page_groups:
            parent_chunks = []
            for page in page_group:
                page_text = page.cleaned_text if use_cleaned_text else page.raw_text
                if not page_text.strip():
                    continue

                page_chunks = self.__parent_splitter.split_text(page_text)
                if not page_chunks:
                    continue

                for chunk in page_chunks:
                    metadata = {
                        "source_file": f"{doc_path.stem}.md",
                        "slide_title": page.slide_title,
                    }
                    if page.page_number is not None:
                        metadata["page_number"] = page.page_number
                        metadata["page_numbers"] = [page.page_number]
                    chunk.metadata.update(metadata)
                parent_chunks.extend(page_chunks)

            if parent_chunks:
                parent_chunk_groups.append(parent_chunks)

        return parent_chunk_groups

    def __contiguous_page_groups(self, pages, selected_pages):
        """Avoid merging across gaps when a local rebuild skips unaffected pages."""
        selected = []
        for page in pages:
            page_number = page.page_number if page.page_number is not None else 1
            if selected_pages is None or page_number in selected_pages:
                selected.append(page)

        if not selected:
            return []

        groups = [[selected[0]]]
        previous = selected[0].page_number if selected[0].page_number is not None else 1
        for page in selected[1:]:
            current = page.page_number if page.page_number is not None else 1
            if current == previous + 1:
                groups[-1].append(page)
            else:
                groups.append([page])
            previous = current

        return groups

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

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path, parent_counts_by_page):
        index_config = current_index_config()
        for p_chunk in parent_chunks:
            # Parent IDs are anchored to source page, not whole-document order,
            # so rebuilding page 3 does not shift IDs for pages 4+.
            page_numbers = p_chunk.metadata.get("page_numbers") or [p_chunk.metadata.get("page_number") or 1]
            page_numbers = [page for page in page_numbers if page is not None]
            start_page = min(page_numbers) if page_numbers else 1
            local_parent_index = parent_counts_by_page[start_page]
            parent_counts_by_page[start_page] += 1
            parent_id = f"{doc_path.stem}_page_{start_page}_parent_{local_parent_index}"
            p_chunk.metadata.update({
                "source": str(doc_path.stem)+".md",
                "source_file": str(doc_path.stem)+".md",
                "parent_id": parent_id,
                "chunk_index": len(all_parent_pairs),
                "doc_id": doc_path.stem,
                "index_schema_version": INDEX_SCHEMA_VERSION,
                "chunker_config_hash": index_config["chunker_config_hash"],
                "cleaner_config_hash": index_config["cleaner_config_hash"],
                "embedding_config_hash": index_config["embedding_config_hash"],
            })
            
            all_parent_pairs.append((parent_id, p_chunk))
            child_chunks = self.__child_splitter.split_documents([p_chunk])
            for child_index, child_chunk in enumerate(child_chunks):
                child_chunk.metadata["chunk_id"] = f"{parent_id}_child_{child_index}"
                child_chunk.metadata["chunk_index"] = len(all_child_chunks) + child_index
                child_chunk.metadata["doc_id"] = doc_path.stem
            all_child_chunks.extend(child_chunks)
