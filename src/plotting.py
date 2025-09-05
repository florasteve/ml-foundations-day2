from pathlib import Path
import matplotlib.pyplot as plt

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def savefig(name, dpi=150):
    plt.savefig(FIG_DIR / name, dpi=dpi, bbox_inches="tight")
