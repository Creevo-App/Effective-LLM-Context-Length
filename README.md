# Effective-LLM-Context-Length

We measure the effective context length of LLMs on mathematical reasoning tasks using the AIME (American Invitational Mathematics Examination) dataset.

## Overview

This project evaluates how different amounts of context padding affect LLM performance on mathematical problems. By adding random word padding to prompts, we can study how models perform as context length increases, helping identify optimal context usage and performance degradation thresholds.

We find that for 2.5 flash the performance starts to degrade around 64k tokens (-15% accuracy), and completely degrade around 128k tokens (-30% accuracy).

For 2.5 pro the performance starts to degrade around 128k tokens (-5-10% accuracy), yet completely degrade around 256k tokens (-40% accuracy).

We expect for both models the models become more or less unusable for techincal tasks after 256-512k tokens.

## 🚀 Quick Start

### Prerequisites

1. **Python Virtual Environment**: This project uses a virtual environment located at `venv/`
2. **API Key**: You need a Google Gemini API key in a `.env` file:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

### Installation

```bash
# Install dependencies (using the virtual environment)
./venv/bin/python -m pip install google-genai python-dotenv matplotlib numpy tenacity
```

### Running an Evaluation

1. **Choose your model** by editing the `model` variable in `run_gemini_script.py`:

   ```python
   model = "gemini-2.5-pro"  # or "gemini-2.5-flash"
   ```

2. **Run the evaluation**:

   ```bash
   ./venv/bin/python run_gemini_script.py
   ```

3. **Generate visualizations**:
   ```bash
   ./venv/bin/python visualize.py
   ```

## 📊 What It Does

### Evaluation Process

- **Dataset**: 30 AIME 2025 mathematical problems
- **Token Padding Sizes**: [0, 256, 1024, 4096, 8192, 16384, 32K, 64K, 128K, 256K, (512K)] Note after 128k it gets very expensive.
- **Multiple Runs**: 10 runs per problem-token combination for statistical reliability
- **Parallel Processing**: Uses 16 workers for efficient evaluation
- **Total Evaluations**: 3,100 individual API calls (30 problems × 11 token sizes × 10 runs)

## 📈 Results Structure

Results are organized by model in separate directories:

```
gemini-2.5-pro/
├── line_chart.png                                   # Performance trend
├── line_chart_multisample.png                       # Std error if each question is an indepdent sample (not really the case though).

gemini-2.5-flash/
├── aime_2025_token_padding_evaluation_results.json
├── line_chart.png
└── bar_chart.png
```

## 🔍 Key Findings

Based on our evaluations, we typically observe:

1. **Optimal Range**: For 2.5 pro, performance holds strong
3. **Statistical Significance**: Large context sizes (>128K) show statistically significant degradation
4. **Model Differences**: Different models show varying sensitivity to context length

## 📁 Project Structure

- **`run_gemini_script.py`**: Main evaluation script with multiprocessing
- **`visualize.py`**: Creates line charts and bar charts with error bars
- **`download_aime_dataset.py`**: Downloads the AIME dataset
- **`aime_2025_dataset.json`**: Mathematical problems dataset
- **`venv/`**: Python virtual environment with dependencies

## ⚙️ Configuration

### Evaluation Parameters

```python
# In run_gemini_script.py
num_runs = 10         # Runs per problem-token combination
max_workers = 16       # Parallel processing threads
token_sizes = [0, 256, 1024, 4096, 8192, 16384, 32000, 64000, 128000, 256000] # with 512000 for 2.5-flash.
```

### Visualization Settings

- Error bars show **3× standard error** for better visibility
- Automatic log scaling for large token ranges
- Professional styling with confidence intervals

## 🛠️ Technical Details

### API Integration

- Uses the new `google-genai` package
- Thread-local clients for parallel processing
- Automatic retry logic with exponential backoff
- Thinking budget of 1024 tokens for reasoning

### Statistical Methods

- **Standard Error**: σ/√n where n = number of problems
- **Confidence Intervals**: ±3 standard errors (≈99.7% confidence)