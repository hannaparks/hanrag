from typing import List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from ..parsers.table_parser import TableData
from .base import TextChunk


@dataclass
class TableChunkingConfig:
    """Configuration for table chunking strategies"""
    # Row-based chunking
    max_rows_per_chunk: int = 20  # Maximum rows in a single chunk
    include_headers_in_each_chunk: bool = True  # Include column headers in every chunk
    preserve_row_integrity: bool = True  # Never split individual rows
    
    # Column-based chunking
    enable_column_chunks: bool = True  # Create separate chunks for columns
    max_values_per_column_chunk: int = 50  # Max values in column summary
    
    # Semantic chunking
    group_by_column: Optional[str] = None  # Group rows by a specific column value
    create_summary_chunk: bool = True  # Create overview chunk for entire table
    
    # Size limits
    max_chunk_tokens: int = 512  # Maximum tokens per chunk
    overlap_rows: int = 2  # Number of rows to overlap between chunks


class TableChunker:
    """Specialized chunker for tabular data"""
    
    def __init__(self, config: Optional[TableChunkingConfig] = None):
        self.config = config or TableChunkingConfig()
    
    def chunk_table(
        self,
        table_data: TableData,
        document_id: str,
        source_path: str
    ) -> Tuple[List[TextChunk], List[TextChunk]]:
        """
        Chunk table data into row-based and summary chunks
        Returns: (child_chunks, parent_chunks)
        """
        child_chunks = []
        parent_chunks = []
        
        # Create table summary chunk (parent)
        if self.config.create_summary_chunk:
            summary_chunk = self._create_table_summary_chunk(
                table_data, document_id, source_path
            )
            parent_chunks.append(summary_chunk)
        
        # Create row-based chunks
        row_chunks = self._create_row_chunks(
            table_data, document_id, source_path,
            parent_id=parent_chunks[0].chunk_id if parent_chunks else None
        )
        child_chunks.extend(row_chunks)
        
        # Create column-based chunks if enabled
        if self.config.enable_column_chunks:
            column_chunks = self._create_column_chunks(
                table_data, document_id, source_path,
                parent_id=parent_chunks[0].chunk_id if parent_chunks else None
            )
            child_chunks.extend(column_chunks)
        
        # Create semantic group chunks if configured
        if self.config.group_by_column and self.config.group_by_column in table_data.headers:
            group_chunks = self._create_semantic_group_chunks(
                table_data, document_id, source_path,
                parent_id=parent_chunks[0].chunk_id if parent_chunks else None
            )
            child_chunks.extend(group_chunks)
        
        logger.info(f"Created {len(child_chunks)} child chunks and {len(parent_chunks)} parent chunks for table")
        return child_chunks, parent_chunks
    
    def _create_table_summary_chunk(
        self,
        table_data: TableData,
        document_id: str,
        source_path: str
    ) -> TextChunk:
        """Create a summary chunk for the entire table"""
        
        # Build comprehensive summary
        summary_parts = [
            f"Table Summary: {table_data.row_count} rows × {table_data.column_count} columns",
            f"Format: {table_data.format.value}",
            f"Source: {source_path}",
            "",
            f"Columns: {', '.join(table_data.headers)}",
            ""
        ]
        
        # Add column statistics
        df = table_data.to_dataframe()
        summary_parts.append("Column Information:")
        
        for col in table_data.headers:
            if col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].count()
                unique = df[col].nunique()
                summary_parts.append(f"- {col}: {dtype} ({non_null} non-null, {unique} unique)")
        
        # Add data sample
        summary_parts.append("\nFirst 5 rows:")
        for i, row in enumerate(table_data.rows[:5]):
            row_summary = f"Row {i+1}: " + ", ".join([f"{k}={v}" for k, v in row.items()])
            summary_parts.append(row_summary)
        
        content = "\n".join(summary_parts)
        
        return TextChunk(
            content=content,
            start_index=0,
            end_index=len(content),
            chunk_id=f"{document_id}_table_summary",
            parent_id=None,
            metadata={
                "chunk_type": "table_summary",
                "document_id": document_id,
                "table_format": table_data.format.value,
                "row_count": table_data.row_count,
                "column_count": table_data.column_count,
                "headers": table_data.headers,
                "source": source_path
            }
        )
    
    def _create_row_chunks(
        self,
        table_data: TableData,
        document_id: str,
        source_path: str,
        parent_id: Optional[str] = None
    ) -> List[TextChunk]:
        """Create chunks based on table rows"""
        chunks = []
        
        # Process rows in batches
        for batch_idx in range(0, table_data.row_count, self.config.max_rows_per_chunk):
            # Calculate batch boundaries with overlap
            start_idx = max(0, batch_idx - self.config.overlap_rows)
            end_idx = min(
                table_data.row_count,
                batch_idx + self.config.max_rows_per_chunk
            )
            
            # Get rows for this batch
            batch_rows = table_data.rows[start_idx:end_idx]
            
            # Build chunk content
            content_parts = []
            
            # Add context
            content_parts.append(f"Table rows {start_idx + 1} to {end_idx} of {table_data.row_count}")
            
            # Add headers if configured
            if self.config.include_headers_in_each_chunk:
                content_parts.append(f"Columns: {', '.join(table_data.headers)}")
                content_parts.append("")
            
            # Add row data in readable format
            for i, row in enumerate(batch_rows):
                row_num = start_idx + i + 1
                content_parts.append(f"Row {row_num}:")
                
                for header in table_data.headers:
                    value = row.get(header, "N/A")
                    content_parts.append(f"  {header}: {value}")
                
                content_parts.append("")  # Empty line between rows
            
            content = "\n".join(content_parts).strip()
            
            chunk = TextChunk(
                content=content,
                start_index=start_idx,
                end_index=end_idx,
                chunk_id=f"{document_id}_rows_{batch_idx}_{batch_idx + self.config.max_rows_per_chunk}",
                parent_id=parent_id,
                metadata={
                    "chunk_type": "table_rows",
                    "document_id": document_id,
                    "parent_id": parent_id,
                    "table_format": table_data.format.value,
                    "row_range": [start_idx + 1, end_idx],
                    "row_count": len(batch_rows),
                    "total_rows": table_data.row_count,
                    "headers": table_data.headers,
                    "source": source_path,
                    "has_overlap": start_idx < batch_idx
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_column_chunks(
        self,
        table_data: TableData,
        document_id: str,
        source_path: str,
        parent_id: Optional[str] = None
    ) -> List[TextChunk]:
        """Create chunks for individual columns"""
        chunks = []
        df = table_data.to_dataframe()
        
        for col_idx, column in enumerate(table_data.headers):
            if column not in df.columns:
                continue
            
            # Get column data
            col_data = df[column]
            
            # Build column analysis
            content_parts = [
                f"Column Analysis: {column}",
                f"Table: {source_path}",
                f"Position: Column {col_idx + 1} of {table_data.column_count}",
                "",
                "Statistics:",
                f"- Data type: {col_data.dtype}",
                f"- Non-null values: {col_data.count()}/{len(col_data)}",
                f"- Unique values: {col_data.nunique()}",
                ""
            ]
            
            # Add value distribution for categorical data
            if col_data.nunique() <= 20:  # Categorical-like column
                content_parts.append("Value distribution:")
                value_counts = col_data.value_counts()
                for value, count in value_counts.items():
                    content_parts.append(f"  {value}: {count} ({count/len(col_data)*100:.1f}%)")
                content_parts.append("")
            
            # Add sample values
            content_parts.append(f"Sample values (first {min(10, len(col_data))}):")
            for i, value in enumerate(col_data.head(10)):
                content_parts.append(f"  Row {i+1}: {value}")
            
            if len(col_data) > 10:
                content_parts.append(f"  ... and {len(col_data) - 10} more values")
            
            content = "\n".join(content_parts)
            
            chunk = TextChunk(
                content=content,
                start_index=0,
                end_index=len(content),
                chunk_id=f"{document_id}_column_{column}",
                parent_id=parent_id,
                metadata={
                    "chunk_type": "table_column",
                    "document_id": document_id,
                    "parent_id": parent_id,
                    "column_name": column,
                    "column_index": col_idx,
                    "data_type": str(col_data.dtype),
                    "unique_values": int(col_data.nunique()),
                    "null_count": int(col_data.isna().sum()),
                    "source": source_path
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_semantic_group_chunks(
        self,
        table_data: TableData,
        document_id: str,
        source_path: str,
        parent_id: Optional[str] = None
    ) -> List[TextChunk]:
        """Create chunks grouped by a specific column value"""
        chunks = []
        
        if not self.config.group_by_column:
            return chunks
        
        df = table_data.to_dataframe()
        
        if self.config.group_by_column not in df.columns:
            logger.warning(f"Group by column '{self.config.group_by_column}' not found in table")
            return chunks
        
        # Group by the specified column
        grouped = df.groupby(self.config.group_by_column)
        
        for group_value, group_df in grouped:
            # Convert group back to row format
            group_rows = group_df.to_dict('records')
            
            # Build chunk content
            content_parts = [
                f"Table Group: {self.config.group_by_column} = {group_value}",
                f"Rows in group: {len(group_rows)} of {table_data.row_count} total",
                "",
                f"Columns: {', '.join(table_data.headers)}",
                ""
            ]
            
            # Add row data
            for i, row in enumerate(group_rows):
                content_parts.append(f"Row {i+1}:")
                for header in table_data.headers:
                    value = row.get(header, "N/A")
                    content_parts.append(f"  {header}: {value}")
                content_parts.append("")
            
            content = "\n".join(content_parts).strip()
            
            chunk = TextChunk(
                content=content,
                start_index=0,
                end_index=len(content),
                chunk_id=f"{document_id}_group_{self.config.group_by_column}_{str(group_value)[:20]}",
                parent_id=parent_id,
                metadata={
                    "chunk_type": "table_semantic_group",
                    "document_id": document_id,
                    "parent_id": parent_id,
                    "group_column": self.config.group_by_column,
                    "group_value": str(group_value),
                    "group_size": len(group_rows),
                    "total_rows": table_data.row_count,
                    "headers": table_data.headers,
                    "source": source_path
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def create_hybrid_chunks(
        self,
        table_data: TableData,
        document_id: str,
        source_path: str
    ) -> List[TextChunk]:
        """Create chunks that combine multiple strategies for better retrieval"""
        chunks = []
        
        # Create chunks that include both row context and column summaries
        df = table_data.to_dataframe()
        
        for batch_idx in range(0, table_data.row_count, self.config.max_rows_per_chunk):
            end_idx = min(table_data.row_count, batch_idx + self.config.max_rows_per_chunk)
            batch_rows = table_data.rows[batch_idx:end_idx]
            
            # Build enhanced chunk with multiple representations
            content_parts = [
                f"Table Section: Rows {batch_idx + 1}-{end_idx} of {table_data.row_count}",
                f"Format: {table_data.format.value}",
                "",
                "Column Overview:",
            ]
            
            # Add column summaries for this batch
            batch_df = df.iloc[batch_idx:end_idx]
            for col in table_data.headers:
                if col in batch_df.columns:
                    col_summary = f"- {col}: {batch_df[col].nunique()} unique values"
                    content_parts.append(col_summary)
            
            content_parts.append("\nDetailed Rows:")
            
            # Add row details
            for i, row in enumerate(batch_rows):
                row_num = batch_idx + i + 1
                # Inline format for better semantic understanding
                row_desc = f"Row {row_num}: " + ", ".join([
                    f"{k}={v}" for k, v in row.items()
                ])
                content_parts.append(row_desc)
            
            content = "\n".join(content_parts)
            
            chunk = TextChunk(
                content=content,
                start_index=batch_idx,
                end_index=end_idx,
                chunk_id=f"{document_id}_hybrid_{batch_idx}",
                parent_id=None,
                metadata={
                    "chunk_type": "table_hybrid",
                    "document_id": document_id,
                    "table_format": table_data.format.value,
                    "row_range": [batch_idx + 1, end_idx],
                    "headers": table_data.headers,
                    "source": source_path
                }
            )
            chunks.append(chunk)
        
        return chunks