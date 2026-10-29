#!/usr/bin/env python
"""Generate the counting_objects_with_LLMs notebooks.

For each scenario defined in config.toml this script:
  1. loads template.ipynb (the shared narrative skeleton);
  2. substitutes the scenario placeholders (title, intro, prompt, conclusion);
  3. expands the {{MODEL_CELLS}} sentinel into a "### <model>" header plus a
     code cell per model, reading the code from
     responses/<scenario>_<model_id>.py;
  4. executes the notebook with nbclient (errors allowed) to capture outputs;
  5. extracts each model's printed count and fills {{RESULTS_TABLE}};
  6. writes the finished notebook into Scripts/Jupyter/.

Paths are resolved relative to this file, so the script can be run from
anywhere. Run it inside the pixi environment so nbclient can find a kernel:

    pixi run --environment intro-ia python Scripts/llm_notebook/build_notebooks.py
"""

import copy
import os
import re
import tomllib
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell

HERE = Path(__file__).resolve().parent
NOTEBOOKS_DIR = HERE.parent / "Jupyter"
RESPONSES_DIR = HERE / "responses"


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def source_lines(text):
    if text.endswith("\n"):
        text = text[:-1]
    return [line + "\n" for line in text.split("\n")]


def cell_text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def set_cell_text(cell, text):
    cell["source"] = source_lines(text)


def blockquote(text):
    return "\n".join(("> " + line) if line else ">" for line in text.splitlines())


def build_model_cells(models, scenario_name):
    cells = []
    code_cells = {}
    for model in models:
        cells.append(new_markdown_cell("### " + model["label"]))
        code = read_text(RESPONSES_DIR / f"{scenario_name}_{model['id']}.py")
        code_cell = new_code_cell(code)
        cells.append(code_cell)
        code_cells[model["id"]] = code_cell
    return cells, code_cells


def extract_count(outputs):
    for out in outputs:
        if out.get("output_type") == "error":
            return "error"
    for out in outputs:
        if out.get("output_type") == "stream" and out.get("name") == "stdout":
            numbers = re.findall(r"\d+", out.get("text", ""))
            if numbers:
                return int(numbers[-1])
    return None


def results_table(models, counts):
    lines = ["| Model | Count |", "| --- | --- |"]
    for model in models:
        value = counts.get(model["id"])
        if value == "error":
            display = "error"
        elif value is None:
            display = "-"
        else:
            display = str(value)
        lines.append(f"| {model['label']} | {display} |")
    return "\n".join(lines)


def main():
    config = tomllib.loads(read_text(HERE / "config.toml"))
    models = config["models"]
    template = nbformat.read(HERE / "template.ipynb", as_version=4)

    # Run kernels from the notebooks directory so the notebook's ../../Data/...
    # paths resolve to the repo root.
    os.chdir(NOTEBOOKS_DIR)

    for scenario in config["scenarios"]:
        nb = copy.deepcopy(template)
        prompt = read_text(HERE / scenario["prompt_file"]).strip()

        model_cells, code_cells = build_model_cells(models, scenario["name"])

        new_cells = []
        for cell in nb.cells:
            text = cell_text(cell)
            if "{{MODEL_CELLS}}" in text:
                new_cells.extend(model_cells)
                continue
            if cell.cell_type == "markdown":
                mapping = {
                    "TITLE": scenario["title"],
                    "INTRO": scenario["intro"].strip(),
                    "PROMPT": blockquote(prompt),
                    "CONCLUSION": scenario["conclusion"].strip(),
                }
                for key, value in mapping.items():
                    text = text.replace("{{" + key + "}}", value)
                set_cell_text(cell, text)
            new_cells.append(cell)
        nb.cells = new_cells

        client = NotebookClient(
            nb, timeout=600, allow_errors=True, kernel_name="python3"
        )
        client.execute()

        counts = {
            mid: extract_count(code_cells[mid]["outputs"]) for mid in code_cells
        }

        for cell in nb.cells:
            if cell.cell_type == "markdown" and "{{RESULTS_TABLE}}" in cell_text(cell):
                text = cell_text(cell).replace(
                    "{{RESULTS_TABLE}}", results_table(models, counts)
                )
                set_cell_text(cell, text)

        out_path = NOTEBOOKS_DIR / scenario["output"]
        nbformat.write(nb, out_path)
        print(f"wrote {out_path.name}  counts={counts}")


if __name__ == "__main__":
    main()
