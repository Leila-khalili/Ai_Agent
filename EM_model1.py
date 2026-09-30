import os
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

from GPy.models import GPRegression

from sklearn.gaussian_process import GaussianProcessRegressor

model = GaussianProcessRegressor()


from datetime import datetime

# Clear terminal
os.system("cls" if os.name == "nt" else "clear")

#from smolagents import CodeAgent, InferenceClientModel, tool
from smolagents import  CodeAgent, LiteLLMModel, tool

# --------------------------------------------------
# Tool 1: Calculate
# --------------------------------------------------

@tool
def calculate(expression: str) -> str:
    """
    Calculate a simple mathematical expression.

    Args:
        expression: A mathematical expression such as "2 + 3 * 4".
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


# --------------------------------------------------
# Tool 2: Validate result
# --------------------------------------------------

@tool
def validate_result(
    reference_value: float,
    computed_value: float,
    tolerance: float
) -> str:
    """
    Compare a computed result with a reference value.

    Args:
        reference_value: The value reported in the reference paper.
        computed_value: The value calculated by the agent.
        tolerance: Maximum acceptable relative error in percent.
    """

    if reference_value == 0:
        return "Cannot calculate relative error because reference value is zero."

    error = (
        abs(computed_value - reference_value)
        / abs(reference_value)
        * 100
    )

    if error <= tolerance:
        status = "PASS"
    else:
        status = "FAIL"

    return (
        f"Reference value: {reference_value}\n"
        f"Computed value: {computed_value}\n"
        f"Relative error: {error:.2f}%\n"
        f"Validation: {status}"
    )


# --------------------------------------------------
# Tool 3: Parse PDF
# --------------------------------------------------

@tool
def parse_paper(file_path: str) -> str:
    """
    Extract text from a research paper PDF using PyMuPDF.

    Args:
        file_path: Path to the PDF file.
    """
    try:
        import fitz

        doc = fitz.open(file_path)

        pages = []

        for i, page in enumerate(doc):
            text = page.get_text()

            pages.append(
                f"--- Page {i + 1} ---\n{text}"
            )

        return "\n\n".join(pages)

    except Exception as e:
        return f"PDF parsing error: {e}"



# --------------------------------------------------
# AI Model
# --------------------------------------------------

#model = InferenceClientModel(
    #model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

model = LiteLLMModel(
    model_id="ollama_chat/qwen3:8b",
    api_base="http://localhost:11434",
    api_key="ollama",
    num_ctx=8192,
    temperature=0.0,
    seed=42,
)

# --------------------------------------------------
# Agent
# --------------------------------------------------

agent = CodeAgent(
    tools=[
        calculate,
        validate_result,
        parse_paper
    ],
    model=model
)


# --------------------------------------------------
# Run Agent
# --------------------------------------------------

answer = agent.run(
    "You must use the parse_paper tool to read the PDF file "
    "'paper.pdf'. "
    "After reading the paper, provide a structured analysis with "
    "the following sections:\n"
    "1. Main objective\n"
    "2. Algorithm or numerical method\n"
    "3. Governing equations\n"
    "4. Assumptions\n"
    "5. Parameters and material properties\n"
    "6. Boundary conditions\n"
    "7. Reported numerical results\n"
    "8. Information needed to reproduce the results."
)


# --------------------------------------------------
# Print result in Terminal
# --------------------------------------------------

print("\n")
print("=" * 70)
print("AGENT REPORT")
print("=" * 70)
print()

print(answer)

print()
print("=" * 70)
print("END OF REPORT")
print("=" * 70)


# --------------------------------------------------
# Save result as PDF
# --------------------------------------------------
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.units import cm


pdf_filename = "agent_report.pdf"

doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=A4,
    rightMargin=2 * cm,
    leftMargin=2 * cm,
    topMargin=2 * cm,
    bottomMargin=2 * cm
)


styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    spaceAfter=20
)

body_style = ParagraphStyle(
    "ReportBody",
    parent=styles["BodyText"],
    fontSize=10,
    leading=14,
    spaceAfter=8
)


story = []

# Title
story.append(
    Paragraph(
        "AI Research Agent Report",
        title_style
    )
)

# Date and time
current_time = datetime.now().strftime(
    "%Y-%m-%d %H:%M"
)

story.append(
    Paragraph(
        f"Generated: {current_time}",
        body_style
    )
)

story.append(Spacer(1, 10))


# Convert Agent output into PDF paragraphs
text = str(answer)

# Split output into lines
lines = text.split("\n")

for line in lines:

    line = line.strip()

    if not line:
        story.append(Spacer(1, 6))
        continue

    # Escape special HTML characters
    line = (
        line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )

    story.append(
        Paragraph(
            line,
            body_style
        )
    )


# Build PDF
doc.build(story)


print()
print(f"PDF report saved as: {pdf_filename}") 
