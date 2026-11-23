"""
Cross-platform smoke test for euniverse (Linux, Windows, macOS).

Runs headless (Qt "offscreen" platform), needs no network and no data files:
it writes a small WCS-tagged TIFF, loads it through the real load path,
exercises annotation clean-up (sip), the look-up overlay switch-off, the
snapshot file-name format, and the CSV file names (no characters that are
illegal on Windows).

    python tests/smoke_test.py          (after  pip install .)
"""
import json
import os
import re
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["EUNIVERSE_NO_CLUSTERS"] = "1"        # no network access in the test
os.environ["EUNIVERSE_NO_DIAG"] = "1"
tmp = tempfile.mkdtemp(prefix="euniverse_smoke_")
os.environ["XDG_CACHE_HOME"] = os.environ["LOCALAPPDATA"] = tmp
os.environ["XDG_CONFIG_HOME"] = tmp

import numpy as np
import tifffile
from PyQt5.QtWidgets import QApplication

app = QApplication([])
from euniverse.annotations import Annotation                      # noqa: E402
from euniverse.euniverse import MainWindow                       # noqa: E402
from euniverse.image_exporter import ImageExporter               # noqa: E402
from euniverse.paths import cache_dir                            # noqa: E402

failures = []


def make_annotation():
    from PyQt5.QtWidgets import QGraphicsEllipseItem
    item = QGraphicsEllipseItem(90, 90, 20, 20)
    viewer.scene.addItem(item)
    return Annotation(item, 150.0, 2.0, "GL: test", 2.0)


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# --- a small tile with a WCS ---------------------------------------------------------------
pix = 0.1 / 3600
meta = dict(CRPIX1=100.5, CRPIX2=100.5, CRVAL1=150.0, CRVAL2=2.0, CTYPE1="RA---TAN", CTYPE2="DEC--TAN",
            CD1_1=-pix, CD1_2=0.0, CD2_1=0.0, CD2_2=pix, RADESYS="ICRS")
path = os.path.join(tmp, "TILE123456_smoke.tif")
tifffile.imwrite(path, (np.random.rand(200, 200, 3) * 60000).astype(np.uint16), description=json.dumps(meta))

win = MainWindow()
win.resize(900, 700)
win.show()
viewer, dock = win.viewer, win.control_widget


def pump(cond, seconds=30):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


viewer.filepath, viewer.dirpath = path, tmp
viewer.load_image(path)
check(pump(lambda: viewer.original_image is not None), "TIFF loads through the background thread")
check(viewer.original_image is not None and viewer.original_image.shape == (200, 200, 3), "image has the expected shape")
check(viewer.wcs is not None, "WCS is valid")

viewer.annotations.append(make_annotation())
try:
    viewer.clear_annotations()
    check(True, "clear_annotations() runs (uses sip.isdeleted)")
except Exception as e:                                           # ImportError on a broken sip
    check(False, f"clear_annotations() raised {e!r}")
try:
    viewer.simbad.clear()
    check(True, "look-up overlay clear() runs (uses sip.isdeleted)")
except Exception as e:
    check(False, f"simbad.clear() raised {e!r}")

# --- file names ----------------------------------------------------------------------------------
ILLEGAL = set('<>:"/\\|?*')
tag = ImageExporter._coord_tag(150.123456, 2.12345)
check(tag == "150.123456+2.12345" and not (set(tag) & ILLEGAL), f"coordinate tag {tag!r}")
check(ImageExporter._coord_tag(150.0, -1.5) == "150.000000-1.50000", "negative declination keeps its sign")

viewer.tileID = "TILE123456"
viewer.annotations.append(make_annotation())
before = set(os.listdir(tmp))
dock.on_save_targets()
new = sorted(set(os.listdir(tmp)) - before)
check(len(new) == 1 and new[0].endswith(".csv") and not (set(new[0]) & ILLEGAL),
      f"'save targets' file name is valid on Windows: {new}")

# --- cache location ---------------------------------------------------------------------------------
d = cache_dir("simbad")
check(os.path.isabs(d) and d.endswith(os.path.join("euniverse", "simbad")), f"cache dir {d}")

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED")
sys.exit(1 if failures else 0)
