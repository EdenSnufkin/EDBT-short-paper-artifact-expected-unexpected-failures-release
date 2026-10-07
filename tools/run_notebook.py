"""Execute notebooks/paper_results.ipynb headlessly and write the outputs back into it."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "paper_results.ipynb"
nb = nbformat.read(nb_path, as_version=4)
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}}).execute()
nbformat.write(nb, nb_path)
print("executed", nb_path)
