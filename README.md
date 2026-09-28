# COBOL Quality QLoRA Lab

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![QLoRA](https://img.shields.io/badge/QLoRA-4--Bit%20Fine--Tuning-8A2BE2?style=for-the-badge)
![COBOL](https://img.shields.io/badge/COBOL-00599C?style=for-the-badge)
![GnuCOBOL](https://img.shields.io/badge/GnuCOBOL-Compiler-6B7280?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## Overview

**COBOL Quality QLoRA Lab** is an experimental machine-learning and software-quality framework for investigating how fine-tuned language models can generate and evaluate test cases for legacy COBOL programs.

The project combines **parameter-efficient fine-tuning**, **structured data curation**, **COBOL program execution**, **oracle validation**, and **mutation testing** into a reproducible evaluation workflow.

Rather than judging generated tests only from their textual appearance, the benchmark executes those tests against real COBOL programs and measures whether they can detect intentionally introduced program mutations.

This creates a bridge between **LLM-based code generation** and **execution-based software testing**.

---

## What This Project Does

The system is designed around a simple question:

> Can a language model generate COBOL test cases that are actually useful for finding program defects?

To investigate this, the project provides components for:

- Preparing and curating COBOL training data
- Fine-tuning language models using QLoRA
- Producing structured test-suite outputs
- Validating generated test cases against a reference COBOL implementation
- Compiling COBOL programs with GnuCOBOL
- Executing generated tests
- Running tests against mutated versions of programs
- Measuring mutation-based test effectiveness
- Producing machine-readable evaluation reports

The benchmark therefore evaluates **behavioral effectiveness**, not just whether a model produces syntactically plausible output.

---

## Core Approach

The project follows a pipeline built around four major stages:

### 1. Data Ingestion and Curation

Raw COBOL examples are processed into structured training material.

The data-quality layer is responsible for preparing examples in a form suitable for model training and evaluation. Curation helps ensure that the resulting examples follow the expected schema and contain the information required by downstream components.

This separation between data preparation and model training makes the workflow easier to reproduce and inspect.

---

### 2. Parameter-Efficient Fine-Tuning

The modeling component uses **QLoRA** to adapt a language model while reducing the memory requirements associated with full model fine-tuning.

QLoRA combines low-rank adaptation with quantized model weights, allowing experiments with large language models using substantially fewer trainable parameters than traditional full-parameter fine-tuning.

The implementation is designed around the Hugging Face/PyTorch ecosystem and provides the model-side foundation for COBOL test-generation experiments.

---

### 3. Execution-Based Benchmarking

Generated model output is parsed as structured JSON containing COBOL test cases.

The benchmark validates the generated suite before executing it.

Each generated case is checked against the reference implementation to determine whether its expected output is valid. Invalid oracle cases are excluded from mutation evaluation.

The benchmark also protects execution with configurable timeouts so that an individual test case cannot run indefinitely.

---

### 4. Mutation Testing

The benchmark uses a reference COBOL implementation together with mutated versions of the program.

A generated test case is considered capable of **killing a mutant** when the mutant:

- Produces incorrect output
- Returns a non-zero exit status
- Times out during execution

The resulting mutation score is calculated as:

```text
mutation score =
mutants killed / eligible mutants
```

This gives the project an execution-based metric for determining whether generated tests can distinguish correct behavior from intentionally altered behavior.

---

## Why Mutation Testing?

Traditional code-generation evaluation can focus heavily on textual similarity or syntax.

That can miss an important question:

**Does the generated test actually detect incorrect program behavior?**

Mutation testing addresses this by introducing controlled changes into a program and checking whether the generated tests can detect those changes.

For this project, a high-quality test suite should do more than look reasonable. It should exercise meaningful program behavior strongly enough to expose the injected defects.

---

## COBOL Evaluation Pipeline

The benchmark uses **GnuCOBOL** to compile the reference program and mutant programs.

The evaluation process is approximately:

```text
Generated Model Output
        │
        ▼
Parse JSON Test Suite
        │
        ▼
Validate Test Cases
        │
        ▼
Compile Reference Program
        │
        ▼
Run Generated Tests
        │
        ▼
Validate Expected Outputs
        │
        ▼
Compile Mutant Programs
        │
        ▼
Execute Valid Tests Against Mutants
        │
        ▼
Identify Killed Mutants
        │
        ▼
Calculate Mutation Score
        │
        ▼
Generate Evaluation Report
```

The implementation creates isolated temporary execution environments for benchmark runs and records execution information such as return codes, normalized output, errors, timeouts, and execution duration.

---

## Generated Test Suite Validation

Generated model output must contain a JSON object with a `cases` collection.

The benchmark validates that:

- The output can be parsed as JSON
- A test-case collection exists
- At least one test case is provided
- Test-case identifiers are unique
- Each test case conforms to the expected COBOL case representation

Malformed model output is reported as a parsing failure rather than being silently treated as a successful evaluation.

This makes model failures distinguishable from benchmark execution failures.

---

## Reference Oracle Validation

Before mutation testing begins, each generated test case is executed against the reference implementation.

A test case is considered valid when:

- The reference program does not time out
- The reference program exits successfully
- The actual output matches the expected output

Only oracle-valid cases are used for mutant evaluation.

This prevents incorrect expected outputs from artificially affecting the mutation score.

---

## Mutation Score

The benchmark reports several useful evaluation values, including:

- Number of generated test cases
- Number of oracle-valid cases
- Number of oracle-invalid cases
- Number of eligible mutants
- Number of killed mutants
- Mutation score
- Individual mutant results
- Cases responsible for killing each mutant
- Model-output parsing errors when applicable

A simplified example report looks like:

```json
{
  "generated_case_count": 10,
  "oracle_valid_count": 9,
  "oracle_invalid_count": 1,
  "mutant_count": 5,
  "mutants_killed": 4,
  "mutation_score": 0.8
}
```

The exact values above are illustrative; benchmark results should be generated from an actual run rather than assumed.

---

## Model Evaluation Philosophy

The project separates **generation quality** from **execution quality**.

A model can produce output that looks structurally correct while still generating weak tests.

For that reason, the benchmark does not stop at JSON validation.

The generated tests must survive multiple stages:

1. Structural validation
2. Reference-program validation
3. COBOL execution
4. Mutation execution
5. Mutation detection

This makes the evaluation more closely connected to practical software-testing behavior.

---

## QLoRA Fine-Tuning

The model layer uses **QLoRA-style parameter-efficient adaptation** rather than requiring every model parameter to be updated.

This approach is useful for experimentation because it can reduce the computational and memory requirements associated with fine-tuning.

The implementation is built around modern machine-learning tooling, including:

- PyTorch
- Hugging Face Transformers
- Parameter-efficient fine-tuning
- Quantized model weights
- Low-rank adaptation

The objective is to specialize a language model toward the project's structured COBOL testing task while keeping the fine-tuning process comparatively lightweight.

---

## Data Quality

The project treats training data as a first-class component of the machine-learning pipeline.

The data-quality layer provides functionality for ingesting and curating COBOL examples before they reach the model-training stage.

This is important because structured test generation depends not only on the model architecture but also on the quality, consistency, and validity of the examples used during training.

A simplified conceptual flow is:

```text
Raw COBOL Data
      │
      ▼
Ingestion
      │
      ▼
Validation
      │
      ▼
Curation
      │
      ▼
Structured Training Examples
      │
      ▼
QLoRA Fine-Tuning
```

---

## Benchmark Programs

The repository includes a reference COBOL program and a deliberately buggy/mutated counterpart for benchmark experimentation.

The reference program represents the expected behavior, while the buggy or mutated implementation provides an altered behavior that generated tests can potentially expose.

This provides a concrete execution target for measuring test-suite effectiveness.

---

## Error and Timeout Handling

The benchmark includes explicit handling for execution failures.

A COBOL test execution can be marked as unsuccessful when the program:

- Times out
- Exits with an error
- Produces output that does not match the expected result

The evaluation code also limits execution time and captures standard output and error information, making failures easier to diagnose.

---

## Example Evaluation Workflow

After producing a model-generated test suite, the benchmark can evaluate it conceptually as follows:

```python
result = score_generated_suite(
    task=task,
    model_output=model_output,
    timeout_seconds=2.0,
)
```

The resulting report can then be written to JSON for further analysis:

```python
write_report(
    result,
    "reports/evaluation.json",
)
```

The benchmark implementation exposes these evaluation operations through the Python code in the repository.

---

## Reproducibility

The project is structured to keep the major experimental components separated:

- Data preparation
- Model adaptation
- Test generation
- COBOL execution
- Mutation evaluation
- Result reporting

This separation makes it possible to modify one part of the experiment without having to redesign the entire pipeline.

For reproducible experiments, use the project's actual dependency and configuration files when setting up the environment.

---

## Requirements

The evaluation workflow requires a Python environment together with the machine-learning dependencies used by the project.

For COBOL execution, the benchmark requires the **GnuCOBOL `cobc` compiler** to be available on the system.

The benchmark explicitly checks for the compiler before attempting compilation and reports an error when it is unavailable.

Typical system requirements include:

- Python
- PyTorch
- Hugging Face Transformers
- PEFT/QLoRA tooling
- GnuCOBOL
- A suitable environment for model fine-tuning

Use the repository's dependency/configuration files as the authoritative source for exact versions and installation commands.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/matrixwhisper/cobol-quality-qlora-lab.git
cd cobol-quality-qlora-lab
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the project's Python dependencies using the repository's dependency specification.

Before running the COBOL benchmark, verify that GnuCOBOL is installed:

```bash
cobc --version
```

If `cobc` is not available on your PATH, the benchmark cannot compile the reference or mutant programs.

---

## Running the Benchmark

The benchmark is designed to evaluate a generated JSON test suite against a COBOL task.

At a high level:

```python
from benchmark.pipeline import score_generated_suite

result = score_generated_suite(
    task=task,
    model_output=model_output,
)

print(result)
```

The evaluator parses the model output, validates the generated cases, executes them against the reference implementation, and then evaluates valid cases against the available mutants.

---

## Evaluation Output

A successful evaluation produces structured information that can be stored as JSON.

The report can contain information such as:

```text
Task ID
Generated test count
Oracle-valid test count
Oracle-invalid test count
Mutant count
Killed mutant count
Mutation score
Per-mutant results
Killed-by test cases
Parsing errors
```

This makes the benchmark suitable for automated experiments and downstream analysis.

---

## Technical Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation and orchestration |
| PyTorch | Machine-learning framework |
| Hugging Face Transformers | Language-model tooling |
| PEFT / QLoRA | Parameter-efficient fine-tuning |
| COBOL | Target programming language |
| GnuCOBOL | COBOL compilation and execution |
| JSON | Structured model outputs and evaluation reports |
| Mutation Testing | Test-suite effectiveness evaluation |

---

## Research Direction

This project explores the intersection of:

- Large language models
- Parameter-efficient fine-tuning
- Legacy software
- Automated test generation
- Program execution
- Mutation testing
- Data quality
- Software engineering evaluation

The broader goal is to investigate whether language models can be adapted to generate **execution-validated tests for legacy COBOL systems**, rather than evaluating generated code solely through textual or syntactic metrics.

---

## Limitations

This repository is an experimental framework, and benchmark results depend on the model, training data, generated test suites, COBOL tasks, mutants, and execution environment used for a particular experiment.

A mutation score should therefore be interpreted in the context of the specific benchmark rather than as a universal measure of software quality.

The benchmark also depends on GnuCOBOL being correctly installed and available to the execution environment.

---

## Contributing

Contributions are welcome.

Potential areas for improvement include:

- Additional COBOL benchmark tasks
- More diverse mutation operators
- Improved data-curation strategies
- Additional model architectures
- More comprehensive evaluation metrics
- Better test-suite generation strategies
- Reproducibility improvements
- Additional automated validation

When contributing, keep generated test outputs and benchmark results reproducible and clearly document changes to the evaluation methodology.

---

## License

See the repository's license file for the applicable licensing terms.

---

## Acknowledgements

This project builds on the broader open-source ecosystem surrounding:

- Python
- PyTorch
- Hugging Face
- PEFT/QLoRA
- GnuCOBOL
- Mutation testing
- Large language model research

---

## Author

**Idris**

GitHub: https://github.com/matrixwhisper
