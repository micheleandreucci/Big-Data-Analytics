# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None


a = Analysis(
    ['GUI_plot.py'],
    pathex=[],
    binaries=[],
    datas=[('uis/plot.ui', '.'), ('imgs/MoviesLens.png', '.'), ('resources/logoSmallGray.png', 'shap/plots/resources/'), ('resources/bundle.js', 'shap/plots/resources/'), ('inps/eda_df.csv', '.'), ('models/*.npy', '.'), ('models/*.joblib', '.')],
    hiddenimports=['sklearn.utils._typedefs', 'sklearn.neighbors._partition_nodes', 'matplotlib.backends.backend_tkagg'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GUI_plot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
