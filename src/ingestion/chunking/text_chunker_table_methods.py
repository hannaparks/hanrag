# Extension methods for TextChunker to handle table-aware chunking
# This should be added to the TextChunker class in text_chunker.py

def chunk_with_tables(
    self,
    text: str,
    document_id: str,
    metadata: Dict[str, Any]
) -> Tuple[List[TextChunk], List[TextChunk]]:
    """
    Chunk content that may contain tables, using specialized table chunking
    Returns: (child_chunks, parent_chunks)
    """
    
    # Check if document contains table data
    has_table_data = metadata.get('has_table_data', False)
    table_format = metadata.get('table_format')
    
    if has_table_data and table_format:
        # Handle pure table formats (CSV, Excel)
        if table_format in [TableFormat.CSV.value, TableFormat.EXCEL.value]:
            return self._chunk_pure_table_document(text, document_id, metadata)
        
        # Handle mixed content with tables (Word, HTML, Markdown)
        elif table_format in [TableFormat.WORD_TABLE.value, TableFormat.HTML.value]:
            return self._chunk_mixed_content_with_tables(text, document_id, metadata)
    
    # Fallback to regular hierarchical chunking
    return self.hierarchical_chunk(text, document_id)

def _chunk_pure_table_document(
    self,
    text: str,
    document_id: str,
    metadata: Dict[str, Any]
) -> Tuple[List[TextChunk], List[TextChunk]]:
    """Handle documents that are purely tabular (CSV, Excel)"""
    
    child_chunks = []
    parent_chunks = []
    
    # For CSV files
    if metadata.get('table_format') == TableFormat.CSV.value:
        table_dict = metadata.get('table_data', {})
        if table_dict:
            # Reconstruct TableData from dict
            table_data = TableData(
                headers=table_dict.get('headers', []),
                rows=table_dict.get('rows', []),
                format=TableFormat.CSV,
                metadata=table_dict.get('metadata', {})
            )
            
            # Use table chunker
            table_child, table_parent = self.table_chunker.chunk_table(
                table_data, document_id, metadata.get('file_path', '')
            )
            child_chunks.extend(table_child)
            parent_chunks.extend(table_parent)
    
    # For Excel files with multiple sheets
    elif metadata.get('table_format') == TableFormat.EXCEL.value:
        all_tables = metadata.get('all_tables', [])
        
        for table_dict in all_tables:
            # Reconstruct TableData from dict
            table_data = TableData(
                headers=table_dict.get('headers', []),
                rows=table_dict.get('rows', []),
                format=TableFormat.EXCEL,
                metadata=table_dict.get('metadata', {})
            )
            
            sheet_name = table_data.metadata.get('sheet_name', 'Sheet')
            sheet_doc_id = f"{document_id}_{sheet_name}"
            
            # Chunk each sheet separately
            table_child, table_parent = self.table_chunker.chunk_table(
                table_data, sheet_doc_id, metadata.get('file_path', '')
            )
            child_chunks.extend(table_child)
            parent_chunks.extend(table_parent)
    
    # If no table chunks created, fall back to regular chunking
    if not child_chunks:
        return self.hierarchical_chunk(text, document_id)
    
    return child_chunks, parent_chunks

def _chunk_mixed_content_with_tables(
    self,
    text: str,
    document_id: str,
    metadata: Dict[str, Any]
) -> Tuple[List[TextChunk], List[TextChunk]]:
    """Handle documents with both text and tables"""
    
    # First, create regular text chunks for non-table content
    child_chunks, parent_chunks = self.hierarchical_chunk(text, document_id)
    
    # Then, add specialized table chunks
    tables = metadata.get('tables', [])
    
    for idx, table_dict in enumerate(tables):
        if not table_dict:
            continue
        
        # Reconstruct TableData
        table_data = TableData(
            headers=table_dict.get('headers', []),
            rows=table_dict.get('rows', []),
            format=TableFormat(table_dict.get('format', TableFormat.TEXT_TABLE.value)),
            metadata=table_dict.get('metadata', {})
        )
        
        table_doc_id = f"{document_id}_table_{idx}"
        
        # Create table chunks
        table_child, table_parent = self.table_chunker.chunk_table(
            table_data, table_doc_id, metadata.get('file_path', '')
        )
        
        # Link table chunks to main document parent
        if parent_chunks and table_child:
            main_parent_id = parent_chunks[0].chunk_id
            for chunk in table_child:
                chunk.metadata['main_document_parent'] = main_parent_id
        
        child_chunks.extend(table_child)
        parent_chunks.extend(table_parent)
    
    return child_chunks, parent_chunks