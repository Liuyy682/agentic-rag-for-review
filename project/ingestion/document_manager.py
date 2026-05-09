from pathlib import Path
import shutil
import config
from ingestion.conversion import pdfs_to_markdowns, clear_directory_contents
from ingestion.cleaning import CleanedMarkdown, clean_markdown_text, parse_pages
from ingestion.index_manifest import (
    IndexManifest,
    build_document_record,
    build_page_hashes,
    changed_pages,
    close_rebuild_scope,
    expand_with_neighbors,
    refresh_document_record,
)

class DocumentManager:

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        Path(config.MARKDOWN_CLEANED_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.MARKDOWN_CLEANING_LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.MARKDOWN_CLEANING_DIFF_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.DOCUMENT_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
        self.manifest = IndexManifest()
        
    def add_documents(self, document_paths, progress_callback=None):
        if not document_paths:
            return 0, 0
            
        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md"]]
        
        if not document_paths:
            return 0, 0
            
        added = 0
        skipped = 0
            
        for i, doc_path in enumerate(document_paths):
            if progress_callback:
                progress_callback((i + 1) / len(document_paths), f"Processing {Path(doc_path).name}")
                
            doc_path = Path(doc_path)
            doc_name = doc_path.stem
            md_path = self.markdown_dir / f"{doc_name}.md"
                
            try:            
                if doc_path.suffix.lower() == ".md":
                    if doc_path.resolve() != md_path.resolve():
                        shutil.copy(doc_path, md_path)
                else:
                    pdfs_to_markdowns(str(doc_path), overwrite=True)

                indexed = self._index_markdown(md_path)
                if indexed:
                    added += 1
                else:
                    skipped += 1
                
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                skipped += 1
            
        return added, skipped
    
    def get_markdown_files(self):
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name.replace(".md", ".pdf") for p in self.markdown_dir.glob("*.md")])

    def _index_markdown(self, md_path: Path) -> bool:
        """Index a Markdown file, using page-level diff when a manifest entry exists."""
        source_file = f"{md_path.stem}.pdf"
        markdown_text = md_path.read_text(encoding="utf-8")
        cleaned = self._clean_for_hash(markdown_text, source_file)
        page_hashes = build_page_hashes(cleaned.pages)

        if not page_hashes:
            return False

        manifest = IndexManifest()
        document = manifest.get_document(source_file)
        if manifest.data.get("documents") and not manifest.is_config_compatible():
            print(
                "Index configuration changed; run a full rebuild before using "
                f"page-level incremental updates. Skipped {source_file}."
            )
            return False

        if document is None:
            return self._full_index_document(manifest, md_path, markdown_text, page_hashes, source_file)

        changed = changed_pages(document, page_hashes)
        if not changed:
            return False

        # Start from changed pages plus neighbors, then include any pages tied
        # together by old cross-page parent chunks so stale content is removed.
        old_pages = {int(page_number) for page_number in document.get("pages", {})}
        new_pages = {int(page_number) for page_number in page_hashes}
        seed_pages = expand_with_neighbors(changed, old_pages | new_pages)
        rebuild_pages, stale_parent_ids = close_rebuild_scope(document, seed_pages, new_pages)

        # Delete before writing new chunks to avoid duplicate vectors for the
        # same source pages if the new chunk boundaries differ.
        self.rag_system.vector_db.delete_by_parent_ids(
            self.rag_system.collection_name,
            sorted(stale_parent_ids),
        )
        self.rag_system.parent_store.delete_many(sorted(stale_parent_ids))

        parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(
            md_path,
            page_numbers=sorted(rebuild_pages),
        )
        if child_chunks:
            collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
            collection.add_documents(child_chunks)
        self.rag_system.parent_store.save_many(parent_chunks)

        updated_document = refresh_document_record(
            document,
            markdown_text,
            page_hashes,
            stale_parent_ids,
            parent_chunks,
            child_chunks,
        )
        manifest.set_document(source_file, updated_document)
        manifest.save()
        return True

    def _full_index_document(self, manifest, md_path: Path, markdown_text: str, page_hashes: dict, source_file: str) -> bool:
        """Build a manifest-backed index for a document that has no prior record."""
        self.rag_system.vector_db.delete_by_source_file(self.rag_system.collection_name, source_file)
        self.rag_system.parent_store.delete_by_source_file(source_file)

        parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
        if not child_chunks:
            return False

        collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
        collection.add_documents(child_chunks)
        self.rag_system.parent_store.save_many(parent_chunks)

        document = build_document_record(
            doc_id=md_path.stem,
            source_file=source_file,
            markdown_path=md_path,
            markdown_text=markdown_text,
            page_hashes=page_hashes,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
        )
        manifest.set_document(source_file, document)
        manifest.save()
        return True

    def _clean_for_hash(self, markdown_text: str, source_file: str):
        """Mirror the chunker cleaning path so page hashes match indexed content."""
        if config.MARKDOWN_CLEANING_ENABLED:
            return clean_markdown_text(
                markdown_text,
                source_file=source_file,
                scan_lines=config.HEADER_FOOTER_SCAN_LINES,
                min_repeat_pages=config.MIN_REPEAT_PAGES,
                min_repeat_ratio=config.MIN_REPEAT_RATIO,
            )

        pages = parse_pages(markdown_text, source_file=source_file)
        for page in pages:
            page.cleaned_text = page.raw_text
            page.cleaned_lines = page.raw_lines
        return CleanedMarkdown(
            source_file=source_file,
            cleaned_text="\n\n".join(page.cleaned_text for page in pages if page.cleaned_text.strip()),
            pages=pages,
            events=[],
            candidates=[],
        )
    
    def clear_all(self):
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.markdown_dir)
        clear_directory_contents(Path(config.MARKDOWN_CLEANED_DIR))
        clear_directory_contents(Path(config.MARKDOWN_CLEANING_LOG_DIR))
        clear_directory_contents(Path(config.MARKDOWN_CLEANING_DIFF_DIR))
        clear_directory_contents(Path(config.DOCUMENT_IMAGE_DIR))
        
        self.rag_system.parent_store.clear_store()
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)
        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)
        self.manifest = IndexManifest()
        self.manifest.save()
