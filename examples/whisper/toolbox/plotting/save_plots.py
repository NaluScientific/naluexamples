import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def save_plot(fig: go.Figure | plt.Figure, root_dir: Path | str = Path().cwd(), ext: str = "png"):
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    if not root_dir.exists():
        root_dir.mkdir(exist_ok=True)

    if isinstance(fig, go.Figure):
        name = _sanitize_filename(str(fig.layout.title.text))
        save_path = root_dir / f"{name}.{ext}"
        fig.write_image(str(save_path), ext)

    elif isinstance(fig, plt.Figure):
        name = _sanitize_filename(fig.get_suptitle())
        save_path = root_dir / f"{name}.{ext}"
        fig.savefig(str(save_path))
    else:
        raise TypeError(f"{type(fig)} is invalid. Type must be go.Figure or plt.Figure")