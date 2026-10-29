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


COUNT_LABEL_FIRST = re.compile(
    r"(?:count|cells?|nuclei|nucleus|objects?|n_cells)\b[^\d\n]{0,40}(\d+)",
    re.IGNORECASE,
)
COUNT_NUMBER_FIRST = re.compile(
    r"(\d+)[^\d\n]{0,10}(?:cells?|nuclei|nucleus|objects)\b",
    re.IGNORECASE,
)
COUNT_TABLE_HINT = re.compile(r"cells|count|nuclei|objects|filename", re.IGNORECASE)


def _plain_text(out):
    data = out.get("data", {})
    chunks = []
    for key in ("text/plain", "text/markdown"):
        text = data.get(key, "")
        chunks.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(chunk for chunk in chunks if chunk)


def _strip_dtype(text):
    # np.int32(7) -> (7) so the count, not the dtype digits, is captured.
    return re.sub(r"np\.(?:int|uint|float|bool)\d*", "", text)


def parse_markdown_cell_count(text):
    """Extract the 'Number of cells' value from a markdown table."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "|" not in line:
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        idx = next(
            (
                j
                for j, c in enumerate(cols)
                if "cell" in c.lower() or "count" in c.lower()
            ),
            None,
        )
        if idx is None:
            continue
        for data_line in lines[i + 1:]:
            if "|" not in data_line:
                continue
            data_cols = [c.strip() for c in data_line.strip().strip("|").split("|")]
            if len(data_cols) <= idx:
                continue
            if all(re.fullmatch(r"[-: ]*", c) for c in data_cols):
                continue  # separator row
            match = re.search(r"\d+", data_cols[idx])
            if match:
                return int(match.group())
    return None


def extract_count(outputs):
    for out in outputs:
        if out.get("output_type") == "error":
            return "error"

    stdout = _strip_dtype(
        "".join(
            out.get("text", "")
            for out in outputs
            if out.get("output_type") == "stream" and out.get("name") == "stdout"
        )
    )
    display = _strip_dtype(
        "".join(
            _plain_text(out)
            for out in outputs
            if out.get("output_type") in ("display_data", "execute_result")
        )
    )

    for text in (stdout, display):
        markdown_count = parse_markdown_cell_count(text)
        if markdown_count is not None:
            return markdown_count

    for text in (stdout, display):
        for pattern in (COUNT_LABEL_FIRST, COUNT_NUMBER_FIRST):
            matches = pattern.findall(text)
            if matches:
                return int(matches[-1])

    # Fallback for a printed/displayed table whose count sits in the last column.
    if COUNT_TABLE_HINT.search(display):
        numbers = re.findall(r"\d+", display)
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
