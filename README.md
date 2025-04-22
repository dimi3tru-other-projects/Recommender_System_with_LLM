TSUM Recommender System
==============================

A recommender system for TSUM e-commerce platform to improve product recommendations.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical datasets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

# Quick Start: Local `venv` Environment

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd TSUM_recommender_system
   ```

2. **Create a Python virtual environment inside the project folder:**

   ```bash
   make create_environment
   ```

   This command:
   - Creates a local `venv` directory
   - Installs/upgrades `pip`, `setuptools`, and `wheel` inside it

3. **Activate the environment:**

   ```bash
   source venv/bin/activate
   ```

   > **Important**: You should keep this environment active in your terminal session whenever you want to install packages or run project code.

4. **Install dependencies:**

   ```bash
   make requirements
   ```

   By default, this will:
   - Run `test_environment` (which calls `python3 test_environment.py`)
   - Install packages from `requirements.txt`
   
   Ensure you have the environment activated so that packages are installed into `venv` rather than your system Python.

5. **(Optional) Generate the Dataset:**

   ```bash
   make data
   ```
   This command will execute `src/data/make_dataset.py`, using `data/raw` as an input and `data/processed` as an output directory.

6. **(Optional) Additional Make Commands:**

   - **Lint the code**:
     ```bash
     make lint
     ```
     Runs `flake8` on the `src/` folder (requires `flake8` in `requirements.txt`).
     
   - **Clean up `.pyc` files**:
     ```bash
     make clean
     ```
     Removes compiled Python files and `__pycache__` directories.

   - **Sync data to/from S3**:
     ```bash
     make sync_data_to_s3
     make sync_data_from_s3
     ```
     Adjust the `BUCKET` and `PROFILE` variables as needed:
     ```bash
     make sync_data_to_s3 BUCKET=my-bucket PROFILE=my-aws-profile
     ```

7. **Running Notebooks:**

   If you plan to use Jupyter notebooks:
   ```bash
   jupyter notebook notebooks/
   ```
   or
   ```bash
   jupyter lab
   ```
   Again, ensure you have installed Jupyter in your `venv` (e.g., `pip install jupyter`) and that the environment is activated.

---

## Tips for Mac MPS (Optional)

- If you use **PyTorch** on Apple Silicon with GPU acceleration, you might install it via:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/metal.html
  ```
  Then add `torch` to your `requirements.txt` for consistency.

---