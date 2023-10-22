from setuptools import find_packages
from cx_Freeze import setup, Executable
import os
import sys


def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)
    return os.path.join(datadir, filename)


options = {
    'build_exe': {
        'includes': [
            'cx_Logging', 'idna',
        ],
        'packages': [
            'asyncio', 'flask', 'jinja2', 'dash', 'plotly', 'waitress', 'os', 'collections', 'xgboost', 'pickle', 'functools', 'webbrowser',
            'sklearn', 'matplotlib', 'pandas', 'numpy', 'dash_core_components', 'dash_html_components', 'pydoc', 'multiprocessing', 'grasia_dash_components',
            'dash_extensions', 'dash_bootstrap_components', 'os', 'base64', 'datetime', 'threading'
        ],
        'excludes': [
            'tkinter'
        ],
        "include_files": [
            'extratrees_flux.sav', 'model.py', 'assets/','extratrees_stock.sav', 'ico.ico', 'random_forest_flux.sav', 'random_forest_stock.sav',
            'scaler_flux.pkl', 'scaler_stock.pkl', 'xgboost_flux.sav', 'xgboost_stock.sav'
        ]
    }
}

executables = [
    Executable('server.py',
               base='console',
               targetName='Credit_Assurance.exe',
               icon = "ico.ico")
]

setup(
    name='Credit Assurance',
    packages=find_packages(),
    version='0.4.0',
    description='rig',
    executables=executables,
    options=options
)