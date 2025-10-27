# Windows Installation Guide (English)

This guide helps you install and configure Sora2WatermarkRemover on Windows.

## Prerequisites

1. **Python and Conda**: Install Miniconda or Anaconda from the official site: [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. **Git**: Optional, to clone this repository: [git-scm.com](https://git-scm.com/downloads).

## Method 1: Automatic install script (Python venv)

1. Ouvrez une invite de commande (CMD) dans le dossier du projet.

2. Exécutez le script d'installation:
   ```
   install_windows.bat
   ```

3. Follow the on-screen instructions.

## Method 2: Manual installation (Python venv)

Si le script automatique ne fonctionne pas, suivez ces étapes manuelles:

1. **Ouvrez une invite de commande** (CMD) ou PowerShell avec les droits administrateur.

2. **Navigate to the project folder**:
   ```
   cd chemin\vers\WatermarkRemover-AI
   ```

3. **Create a Python virtual environment**:
   ```
   python -m venv .venv
   ```

4. **Activate the environment**:
   ```
   .\.venv\Scripts\activate    # Windows
   source .venv/bin/activate     # macOS/Linux
   ```

5. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

6. **Download the LaMa model**:
   ```
   iopaint download --model lama
   ```

## Launching the application

After installation, launch the application:

1. **Activate the environment** (if not already active):
   ```
   conda activate py312aiwatermark
   ```

2. **Start the GUI application**:
   ```
   python remwmgui.py
   ```

## Usage

Once the application is running:

1. **Choose processing mode**:
   - Single image
   - Whole directory

2. **Select input and output paths**.

3. **Configure options**:
   - Overwrite existing files if needed
   - Make watermark regions transparent
   - Adjust maximum bounding box size for detection
   - Select output format (PNG, WEBP, JPG or original)

4. **Click "Start" to begin processing**.

## Common issues and solutions

### Issue: "Conda is not recognized as an internal or external command"
**Solution**: Ensure Conda is installed and its path is added to your environment PATH.

### Issue: Dependency installation failed
**Solution**: Run install commands individually and check specific error messages.

### Issue: The application does not start
**Solution**: Make sure the environment is activated: `conda activate py312aiwatermark`.

### Issue: LaMa model download fails
**Solution**: Ensure stable internet and retry: `iopaint download --model lama`.

## Support

If you encounter issues:
- Open an issue on the repository
- Check discussions for similar problems

---

Enjoy your new AI-powered watermark remover!