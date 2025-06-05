import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from loguru import logger


class TableFormat(Enum):
    """Supported table formats"""
    CSV = "csv"
    EXCEL = "excel"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT_TABLE = "text_table"
    WORD_TABLE = "word_table"
    JSON_ARRAY = "json_array"


@dataclass
class TableData:
    """Represents parsed table data"""
    headers: List[str]
    rows: List[Dict[str, Any]]
    format: TableFormat
    metadata: Dict[str, Any]
    
    @property
    def row_count(self) -> int:
        return len(self.rows)
    
    @property
    def column_count(self) -> int:
        return len(self.headers)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame(self.rows)
    
    def to_row_documents(self) -> List[Dict[str, Any]]:
        """Convert each row to a document with full context"""
        documents = []
        
        for idx, row in enumerate(self.rows):
            # Create a document for each row with column context
            row_content = f"Row {idx + 1} of {self.row_count}:\n"
            
            # Add each field with its header
            for header in self.headers:
                value = row.get(header, "")
                if pd.isna(value) or value == "":
                    value = "N/A"
                row_content += f"{header}: {value}\n"
            
            # Create document with metadata
            doc = {
                "content": row_content.strip(),
                "row_index": idx,
                "row_data": row,
                "table_metadata": {
                    "total_rows": self.row_count,
                    "total_columns": self.column_count,
                    "headers": self.headers,
                    "format": self.format.value
                }
            }
            documents.append(doc)
        
        return documents
    
    def to_column_documents(self) -> List[Dict[str, Any]]:
        """Convert each column to a document for column-based retrieval"""
        documents = []
        df = self.to_dataframe()
        
        for column in self.headers:
            if column in df.columns:
                col_data = df[column].tolist()
                
                # Create column summary
                col_content = f"Column: {column}\n"
                col_content += f"Data type: {df[column].dtype}\n"
                col_content += f"Non-null values: {df[column].count()}/{len(df)}\n"
                col_content += f"Unique values: {df[column].nunique()}\n\n"
                
                # Add sample values
                col_content += "Sample values:\n"
                for i, val in enumerate(col_data[:10]):
                    col_content += f"  Row {i+1}: {val}\n"
                
                if len(col_data) > 10:
                    col_content += f"  ... and {len(col_data) - 10} more rows\n"
                
                doc = {
                    "content": col_content,
                    "column_name": column,
                    "column_data": col_data,
                    "table_metadata": {
                        "total_rows": self.row_count,
                        "total_columns": self.column_count,
                        "headers": self.headers,
                        "format": self.format.value
                    }
                }
                documents.append(doc)
        
        return documents


class UnifiedTableParser:
    """Unified parser for all table formats"""
    
    def __init__(self):
        self.markdown_table_pattern = re.compile(
            r'^\|.*\|$',  # Lines starting and ending with |
            re.MULTILINE
        )
        self.text_table_patterns = [
            # ASCII table with +---+---+ borders
            re.compile(r'^\+[-\+]+\+$', re.MULTILINE),
            # Simple aligned columns with consistent spacing
            re.compile(r'^(\S+\s{2,})+\S+$', re.MULTILINE),
            # Tab-separated values
            re.compile(r'^[^\t]+(\t[^\t]+)+$', re.MULTILINE)
        ]
    
    def parse_csv_content(self, content: str, has_header: bool = True) -> TableData:
        """Parse CSV content into TableData"""
        try:
            # Try to parse CSV from string
            from io import StringIO
            df = pd.read_csv(StringIO(content), header=0 if has_header else None)
            
            if not has_header:
                # Generate column names if no header
                df.columns = [f"Column_{i+1}" for i in range(len(df.columns))]
            
            headers = df.columns.tolist()
            rows = df.to_dict('records')
            
            return TableData(
                headers=headers,
                rows=rows,
                format=TableFormat.CSV,
                metadata={
                    "has_header": has_header,
                    "delimiter": ",",
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse CSV content: {e}")
            raise
    
    def parse_markdown_table(self, content: str) -> Optional[TableData]:
        """Parse Markdown table format"""
        lines = content.strip().split('\n')
        
        # Find table lines
        table_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line:
                # Check if it's a separator line
                if re.match(r'^\|[\s\-:]+\|', line):
                    in_table = True
                    continue
                elif in_table or re.match(r'^\|.*\|$', line):
                    table_lines.append(line)
                    in_table = True
            elif in_table and not line.strip():
                # Empty line ends table
                break
            elif in_table:
                # Non-table line ends table
                break
        
        if len(table_lines) < 2:
            return None
        
        # Parse headers (first line)
        headers = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
        
        # Parse data rows
        rows = []
        for line in table_lines[1:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) == len(headers):
                row = {headers[i]: cells[i] for i in range(len(headers))}
                rows.append(row)
        
        if not rows:
            return None
        
        return TableData(
            headers=headers,
            rows=rows,
            format=TableFormat.MARKDOWN,
            metadata={
                "table_style": "pipe_separated",
                "row_count": len(rows),
                "column_count": len(headers)
            }
        )
    
    def parse_text_table(self, content: str) -> Optional[TableData]:
        """Parse various text table formats"""
        
        # Try to detect ASCII box table
        if '+' in content and '-' in content:
            return self._parse_ascii_box_table(content)
        
        # Try to detect aligned columns
        lines = content.strip().split('\n')
        if len(lines) >= 2:
            # Check for consistent column positions
            return self._parse_aligned_columns(lines)
        
        return None
    
    def _parse_ascii_box_table(self, content: str) -> Optional[TableData]:
        """Parse ASCII box-style tables like:
        +------+------+------+
        | Col1 | Col2 | Col3 |
        +------+------+------+
        | Val1 | Val2 | Val3 |
        +------+------+------+
        """
        lines = content.strip().split('\n')
        
        # Find content lines (not borders)
        content_lines = []
        for line in lines:
            if not re.match(r'^\+[-\+]+\+$', line):
                if '|' in line:
                    content_lines.append(line)
        
        if len(content_lines) < 2:
            return None
        
        # Parse headers
        headers = [cell.strip() for cell in content_lines[0].split('|')[1:-1]]
        
        # Parse rows
        rows = []
        for line in content_lines[1:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) == len(headers):
                row = {headers[i]: cells[i] for i in range(len(headers))}
                rows.append(row)
        
        return TableData(
            headers=headers,
            rows=rows,
            format=TableFormat.TEXT_TABLE,
            metadata={
                "table_style": "ascii_box",
                "row_count": len(rows),
                "column_count": len(headers)
            }
        )
    
    def _parse_aligned_columns(self, lines: List[str]) -> Optional[TableData]:
        """Parse space-aligned column tables"""
        if len(lines) < 2:
            return None
        
        # Detect column positions based on spacing patterns
        # This is a simplified version - could be enhanced with better heuristics
        
        # Try to split by multiple spaces
        headers = re.split(r'\s{2,}', lines[0].strip())
        
        if len(headers) < 2:
            # Try tab separation
            headers = lines[0].strip().split('\t')
        
        if len(headers) < 2:
            return None
        
        # Parse rows with same pattern
        rows = []
        for line in lines[1:]:
            if not line.strip():
                continue
            
            # Try same split pattern
            if '\t' in lines[0]:
                cells = line.strip().split('\t')
            else:
                cells = re.split(r'\s{2,}', line.strip())
            
            if len(cells) == len(headers):
                row = {headers[i]: cells[i] for i in range(len(headers))}
                rows.append(row)
        
        if not rows:
            return None
        
        return TableData(
            headers=headers,
            rows=rows,
            format=TableFormat.TEXT_TABLE,
            metadata={
                "table_style": "aligned_columns",
                "row_count": len(rows),
                "column_count": len(headers)
            }
        )
    
    def detect_table_in_text(self, content: str) -> List[Tuple[int, int, str]]:
        """Detect table regions in text content
        Returns: List of (start_pos, end_pos, table_type) tuples
        """
        tables_found = []
        
        # Check for Markdown tables
        for match in re.finditer(r'((?:^\|.*\|$\n?)+)', content, re.MULTILINE):
            if self.parse_markdown_table(match.group()):
                tables_found.append((match.start(), match.end(), "markdown"))
        
        # Check for ASCII box tables
        ascii_pattern = re.compile(r'((?:^\+[-\+]+\+$\n(?:^\|.*\|$\n)?)+)', re.MULTILINE)
        for match in ascii_pattern.finditer(content):
            tables_found.append((match.start(), match.end(), "ascii_box"))
        
        # Sort by position and remove overlaps
        tables_found.sort(key=lambda x: x[0])
        
        # Remove overlapping regions
        filtered_tables = []
        last_end = -1
        
        for start, end, table_type in tables_found:
            if start >= last_end:
                filtered_tables.append((start, end, table_type))
                last_end = end
        
        return filtered_tables
    
    def extract_html_tables(self, html_content: str) -> List[TableData]:
        """Extract all tables from HTML content"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = []
        
        for table_element in soup.find_all('table'):
            try:
                # Extract headers
                headers = []
                header_row = table_element.find('thead')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                else:
                    # Try first row as headers
                    first_row = table_element.find('tr')
                    if first_row:
                        headers = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
                
                # Extract rows
                rows = []
                tbody = table_element.find('tbody') or table_element
                
                for tr in tbody.find_all('tr'):
                    # Skip header row if already processed
                    if tr == table_element.find('tr') and headers:
                        continue
                    
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    
                    if not headers:
                        # Use first row as headers
                        headers = cells
                    elif len(cells) == len(headers):
                        row = {headers[i]: cells[i] for i in range(len(headers))}
                        rows.append(row)
                
                if rows:
                    table_data = TableData(
                        headers=headers,
                        rows=rows,
                        format=TableFormat.HTML,
                        metadata={
                            "table_index": len(tables),
                            "has_thead": header_row is not None,
                            "row_count": len(rows),
                            "column_count": len(headers)
                        }
                    )
                    tables.append(table_data)
                    
            except Exception as e:
                logger.warning(f"Failed to parse HTML table: {e}")
                continue
        
        return tables
    
    def parse_json_array(self, json_data: List[Dict[str, Any]]) -> TableData:
        """Convert JSON array to table format"""
        if not json_data:
            raise ValueError("Empty JSON array")
        
        # Extract all unique keys as headers
        headers = list(set(key for item in json_data for key in item.keys()))
        headers.sort()  # Consistent ordering
        
        # Normalize rows
        rows = []
        for item in json_data:
            row = {header: item.get(header, "") for header in headers}
            rows.append(row)
        
        return TableData(
            headers=headers,
            rows=rows,
            format=TableFormat.JSON_ARRAY,
            metadata={
                "original_count": len(json_data),
                "row_count": len(rows),
                "column_count": len(headers)
            }
        )