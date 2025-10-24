import os
import inspect
import logging
import logging.config
from pathlib import Path
from typing import List, Tuple, Union
import sys

############### LOGGING ###############

# --- Global Logging Setup ---
# opttional:   - exc_info=True for tracebacks: a single CSV-safe string (commas escaped, newlines flattened).
def setup_global_logger(
    script_name: str = None,
    cwd: Path = None,
    log_level: Union[int, str] = 'INFO',
    headers: Union[List[str], Tuple[str, ...]] = None,
    logger_name: str = 'pipeline'
) -> logging.Logger:
    """
    Set up a custom logger with console (ERROR/CRITICAL only) and file (all levels) handlers.
    Overwrites the log file each run. Supports custom headers and extras for script-specific logging.
    """
    # Map string log_level to int
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    # Get CWD if not provided
    if cwd is None:
        raise ValueError("setup_global_logger requires a 'cwd' argument (Path to log directory)")
    
    # Auto-detect script_name if not provided
    if script_name is None:
        frame = inspect.stack()[1]
        script_name = os.path.basename(frame.filename)
    
    # Generate log file path
    log_filename = f"a_{os.path.splitext(script_name)[0]}.log"
    log_path = cwd / log_filename  # Use Path for consistency
    
    # Validate headers: caller must supply headers (list/tuple). The first three
    # columns must be Date, Level, Msg (case-insensitive). No automatic schema
    # beyond those three is provided.
    if headers is None:
        raise ValueError("setup_global_logger requires a 'headers' argument (first three must be Date,Level,Msg)")
    if not isinstance(headers, (list, tuple)):
        raise ValueError("headers must be a list or tuple of column names")
    if len(headers) < 3:
        raise ValueError("headers must include at least the first three columns: Date, Level, Msg")
    # Normalize check for first three
    first_three = [h.strip().lower() for h in headers[:3]]
    if not (first_three[0] == 'date' and first_three[1] == 'level' and first_three[2] in ('msg', 'message')):
        raise ValueError("headers must start with ['Date','Level','Msg'|'Message'] in that order")

    # CSV-safe formatter that always emits columns in the declared order.
    class CSVFormatter(logging.Formatter):
        def __init__(self, columns, datefmt=None):
            super().__init__()
            self.columns = [str(c) for c in columns]
            self.datefmt = datefmt or '%m/%d'

        def _escape(self, v: str) -> str:
            if v is None:
                return ''
            s = str(v)
            # Double up quotes, wrap field in quotes to be safe
            s = s.replace('"', '""')
            return f'"{s}"'

        def format(self, record):
            # Ensure message formatted
            try:
                message = record.getMessage()
            except Exception:
                message = str(getattr(record, 'msg', ''))

            row = []
            for idx, col in enumerate(self.columns):
                key = col
                if idx == 0:  # Date
                    row.append(self._escape(self.formatTime(record, self.datefmt)))
                elif idx == 1:  # Level
                    row.append(self._escape(record.levelname))
                elif idx == 2:  # Msg
                    row.append(self._escape(message))
                else:
                    # Extras are expected to be set on the record via logging.extra
                    val = record.__dict__.get(key, '')
                    row.append(self._escape(val))
            return ','.join(row)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)  # Log everything to stdout
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    # Open file in overwrite mode and write header each run (truncate file)
    os.makedirs(log_path.parent, exist_ok=True)
    header_line = ','.join(str(h) for h in headers) + '\n'
    # Create the file handler in overwrite mode, then write the header using
    # the handler's open stream to avoid a second truncation.
    try:
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(CSVFormatter(headers, datefmt='%m-%d %H:%M'))
        # Write header to the handler's stream (file is already opened by handler)
        try:
            if getattr(file_handler, 'stream', None):
                file_handler.stream.write(header_line)
                file_handler.stream.flush()
        except Exception:
            # fallback: try direct write
            with open(log_path, 'w', encoding='utf-8') as hf:
                hf.write(header_line)
    except Exception as e:
        print(f"Warning: Could not create file handler for {log_path}: {e}")
        # Fall back to a no-op handler creation
        file_handler = logging.NullHandler()
    
    # Create named logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # Avoid root logger interference
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Enable exc_info by default for exceptions
    logger.exc_info = True

    # Log initialization using the CSV formatter (will produce a row)
    logger.info(f"Logger initialized: {log_path}")
    return logger


###################### DEBUG ############################
# Debug helpers to list/describe files discovered by a LlamaIndex reader
# and by a disk scan. Keep these for troubleshooting and remove/quiet them
# once discovery is verified.

from typing import List, Any
try:
    from llama_index.core import Document
except Exception:
    # llama_index is optional for some environments; fall back to Any so type checks still work
    Document = Any

def log_reader_docs(docs: List[Document], logger, max_ids: int = 20) -> None:
    """
    Log a truncated list of document identifiers returned by the reader.
    - docs: list of LlamaIndex Document objects
    - logger: configured logger instance
    - max_ids: how many ids to log before truncating
    """
    file_ids = []
    for d in docs:
        fid = getattr(d, "source", None) or getattr(d, "doc_id", None) or ((getattr(d, "metadata", None) or {}).get("file_path"))
        file_ids.append(str(fid) if fid else "<unknown>")
    if file_ids:
        display = ", ".join(file_ids[:max_ids])
        more = f", +{len(file_ids)-max_ids} more" if len(file_ids) > max_ids else ""
        logger.info("Found files (reader ids): " + display + more)
    else:
        logger.info("No docs returned by reader.")


def list_markdown_files_and_log(cwd: Path, logger, exts: List[str] = None, max_files: int = 50) -> List[Path]:
    """
    Scan disk under `cwd` for markdown files and log a truncated list of basenames.
    Returns the list of Path objects found.
    - cwd: Path to search under
    - exts: list of extensions to match (defaults to .md/.markdown/.mdx)
    - max_files: how many basenames to include in the log
    """
    if exts is None:
        exts = [".md", ".markdown", ".mdx"]
    md_paths = []
    try:
        for ext in exts:
            md_paths.extend(Path(cwd).rglob(f"*{ext}"))
        md_paths = sorted(set(md_paths))
        if md_paths:
            display_files = ", ".join([p.name for p in md_paths[:max_files]])
            more_files = f", +{len(md_paths)-max_files} more" if len(md_paths) > max_files else ""
            logger.info(f"Found markdown files on disk: {display_files}{more_files}")
        else:
            logger.info("No markdown files found on disk under CWD.")
    except Exception as e:
        logger.info(f"Could not list files on disk: {e}")
        md_paths = []
    return md_paths