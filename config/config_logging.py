"""
Módulo dedicado a configuração centralizada de logs.
"""

import logging
import sys
from pathlib import Path

logs_dir = Path(__file__).parent.parent / "logs"
logs_file_path = logs_dir / "app.log"
 
def setup_logging(level: int = logging.INFO, log_to_file: bool = True) -> None:
    """Configura o registro de logs para toda a aplicação."""
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        logs_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logs_file_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)