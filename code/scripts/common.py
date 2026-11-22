import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
DL = os.path.expanduser("~/mnt/Downloads")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
AGENTS = ["Claude_Code", "Copilot", "OpenAI_Codex", "Cursor", "Devin", "Google_Jules"]
LABEL = {"Claude_Code":"Claude Code","Copilot":"Copilot","OpenAI_Codex":"Codex","Cursor":"Cursor","Devin":"Devin","Google_Jules":"Jules"}
COLOR = dict(zip(AGENTS, ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300"]))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
import glob, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for _f in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc"): font_manager.fontManager.addfont(_f)
_fam = font_manager.FontProperties(fname=sorted(glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))[0]).get_name()
plt.rcParams.update({"font.family":_fam,"axes.unicode_minus":False,
  "figure.facecolor":SURF,"axes.facecolor":SURF,"axes.edgecolor":GRID,"axes.labelcolor":INK2,
  "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,
  "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True,
  "font.size":11,"axes.titlesize":13,"axes.titleweight":"bold","axes.titlecolor":INK,"savefig.dpi":160})
def pq(name): return f"read_parquet('{DL}/{name}.parquet')"
