"""PyRepl multimodal image output example with seaborn.

Run:
    poetry run python examples/pyrepl_seaborn_multimodal_images.py
    uv run python examples/pyrepl_seaborn_multimodal_images.py

What this example demonstrates:
1. Code executed inside PyRepl can generate multiple image artifacts.
2. Direct ``repl.execute(...)`` exposes those artifacts in ``result["artifacts"]``.
3. The ``execute_code`` tool converts image artifacts into a multimodal tool
   return shaped as ``(summary: str, images: list[ImgPath | ImgUrl])``.

This example does not require an LLM provider. It requires the repo's dev
plotting dependencies: seaborn, pandas, and matplotlib.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import textwrap
from pathlib import Path

from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.type import ImgPath, ImgUrl


WORKSPACE = Path(__file__).resolve().parent / "_generated" / "pyrepl_seaborn"
REQUIRED_PLOT_PACKAGES = ("seaborn", "pandas", "matplotlib")


PLOT_CODE = r'''
from pathlib import Path

from IPython.display import Image, display

import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

output_dir = Path(".")

rng = random.Random(2026)
days = ["Thu", "Fri", "Sat", "Sun"]
times = ["Lunch", "Dinner"]
sexes = ["Female", "Male"]

rows = []
for index in range(180):
    day = days[index % len(days)]
    time = times[(index // 3) % len(times)]
    sex = sexes[(index // 5) % len(sexes)]
    party_size = 1 + (index % 6)
    bill_base = 14 + days.index(day) * 3.2 + party_size * 2.6
    total_bill = round(bill_base + rng.random() * 18, 2)
    tip_rate = 0.13 + (0.03 if time == "Dinner" else 0.0) + rng.random() * 0.08
    tip = round(total_bill * tip_rate + math.sin(index / 9) * 0.35, 2)
    rows.append(
        {
            "total_bill": total_bill,
            "tip": max(tip, 0.5),
            "time": time,
            "sex": sex,
            "day": day,
            "size": party_size,
        }
    )

tips = pd.DataFrame(rows)
print("Using seaborn with a local synthetic tips-like dataset")

plot_specs = [
    (
        "tips_scatter.png",
        lambda: sns.scatterplot(
            data=tips,
            x="total_bill",
            y="tip",
            hue="time",
            style="sex",
        ),
        "Scatter: total bill vs tip",
    ),
    (
        "tips_box.png",
        lambda: sns.boxplot(data=tips, x="day", y="total_bill", hue="time"),
        "Box plot: bill distribution by day",
    ),
    (
        "tips_hist.png",
        lambda: sns.histplot(data=tips, x="tip", hue="sex", kde=True),
        "Histogram: tip distribution",
    ),
]

saved_paths = []
for filename, draw, title in plot_specs:
    plt.figure(figsize=(7, 4))
    draw()
    plt.title(title)
    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=160)
    plt.close()
    saved_paths.append(path)

print("Generated images:")
for path in saved_paths:
    print(path.resolve())

for path in saved_paths:
    display(Image(filename=str(path)))
'''


def ensure_plot_dependencies() -> bool:
    missing = [
        package
        for package in REQUIRED_PLOT_PACKAGES
        if importlib.util.find_spec(package) is None
    ]
    if not missing:
        return True

    missing_text = ", ".join(missing)
    print(
        "This example requires the dev plotting dependencies "
        f"({missing_text} missing).\n"
        "Install the repo dev dependencies and run again:\n"
        "  poetry install --with dev\n"
        "If you are maintaining the dependency list, add seaborn with:\n"
        "  poetry add --group dev seaborn",
        file=sys.stderr,
    )
    return False


def describe_artifacts(artifacts: list[dict]) -> None:
    print(f"artifact count: {len(artifacts)}")
    for index, artifact in enumerate(artifacts, start=1):
        print(
            f"  {index}. type={artifact.get('type')} "
            f"source={artifact.get('source')} "
            f"mime={artifact.get('mime_type')} "
            f"path={artifact.get('path') or artifact.get('url')}"
        )


def describe_tool_return(result: object) -> None:
    if isinstance(result, tuple) and len(result) == 2:
        summary, images = result
        print("execute_code returned a multimodal tuple")
        print("summary preview:")
        print(textwrap.indent(str(summary).split("\n")[0], "  "))
        if isinstance(images, list):
            print(f"image payload count: {len(images)}")
            for index, image in enumerate(images, start=1):
                if isinstance(image, ImgPath):
                    print(f"  {index}. ImgPath: {image.path}")
                elif isinstance(image, ImgUrl):
                    print(f"  {index}. ImgUrl: {image.url[:80]}")
                else:
                    print(f"  {index}. unexpected image payload: {type(image).__name__}")
        return

    print("execute_code returned a text-only result")
    print(result)


async def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    repl = PyRepl(working_directory=WORKSPACE)

    print("=" * 80)
    print("Direct PyRepl.execute(...) artifact capture")
    print("=" * 80)
    direct_result = await repl.execute(PLOT_CODE)
    print(f"success: {direct_result['success']}")
    print("stdout:")
    print(textwrap.indent(direct_result["stdout"].strip(), "  "))
    describe_artifacts(direct_result["artifacts"])

    print("\n" + "=" * 80)
    print("execute_code tool multimodal return")
    print("=" * 80)
    execute_tool = next(tool for tool in repl.toolset if tool.name == "execute_code")
    tool_result = await execute_tool.run(PLOT_CODE)
    describe_tool_return(tool_result)

    print("\nGenerated files are under:")
    print(f"  {WORKSPACE}")


if __name__ == "__main__":
    if not ensure_plot_dependencies():
        raise SystemExit(1)
    asyncio.run(main())
