Project Setup Instructions for crayon_vectorizer

This repository is a Python project. The usual file used for project setup and instructions is a README file, typically named `README.md` for Markdown or `README.txt` for plain text. Here, this file is provided as plain text.

1. Clone the repository
   git clone <repo-url>
   cd crayon_vectorizer

2. Create a Python virtual environment
   python -m venv .venv

3. Activate the virtual environment
   PowerShell:
     .\.venv\Scripts\Activate.ps1
   Command Prompt:
     .\.venv\Scripts\activate
   Git Bash / WSL:
     source .venv/bin/activate

4. Install dependencies
   pip install -r requirements.txt

5. Verify the environment
   python -c "import cv2, numpy; print('opencv', cv2.__version__, 'numpy', numpy.__version__)"

6. Prepare your input files
   - `locked_palette_pipeline_config.json` is the shared configuration file used by the pipeline.
   - Put any image input or vector JSON input in the `input/` directory or use full paths.

7. Run the pipeline
   Example command:
     python run_locked_palette_pipeline.py --input input/<your_input_file> --config locked_palette_pipeline_config.json

   If you want output in a specific directory:
     python run_locked_palette_pipeline.py --input input/<your_input_file> --config locked_palette_pipeline_config.json --outdir output

8. Notes
   - `.gitignore` already ignores general generated assets like `output/` and `*.json`, but `locked_palette_pipeline_config.json` is explicitly tracked so it can be version controlled.
   - If you want a nicer long-form file with formatting, use `README.md` instead of plain text.
