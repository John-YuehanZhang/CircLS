"""Headless screenshot of a tqec block-graph .html (three.js / WebGL) with Chrome.

Set CHROME to the browser binary; otherwise a few usual locations are tried.
Returns True when the PNG was written.  Software WebGL (SwiftShader) is forced
so the 3D view renders without a GPU.
"""
import glob
import os
import pathlib
import shutil
import subprocess


def find_chrome():
    cands = [os.environ.get("CHROME")]
    cands += glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome"))
    cands += [shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")]
    for c in cands:
        if c and pathlib.Path(c).exists():
            return c
    return None


def snapshot(html_path, png_path, size=(1000, 700), budget_ms=10000):
    chrome = find_chrome()
    if chrome is None:
        return False
    html = pathlib.Path(html_path).resolve()
    cmd = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu-sandbox",
           "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
           "--enable-webgl", "--hide-scrollbars", f"--window-size={size[0]},{size[1]}",
           f"--virtual-time-budget={budget_ms}", f"--screenshot={pathlib.Path(png_path).resolve()}",
           html.as_uri()]
    try:
        subprocess.run(cmd, check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    return pathlib.Path(png_path).exists()
