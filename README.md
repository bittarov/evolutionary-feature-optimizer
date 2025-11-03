# DataOptima - Evolutionary Feature Optimization Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Intelligent Feature Selection using Bio-Inspired Evolutionary Algorithms**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [API](#api) • [Architecture](#architecture)

</div>

---

## 📋 Overview

**DataOptima** is a powerful web-based platform that leverages evolutionary algorithms to automatically discover optimal feature subsets for machine learning models. By mimicking natural selection processes, it intelligently explores feature combinations to maximize model accuracy while minimizing dimensionality.

### Key Highlights

- 🧬 **Bio-Inspired Intelligence**: Uses genetic algorithms (selection, crossover, mutation) to evolve optimal solutions
- 📊 **Comprehensive Benchmarking**: Compares evolutionary results against 4 traditional methods
- 🎯 **Optimal Selection**: Automatically balances accuracy and feature economy
- 🌐 **User-Friendly Interface**: No coding required - simply upload your dataset
- ⚡ **Performance Optimized**: Reduces dimensionality by 30-70% while maintaining accuracy

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Evolutionary Optimization** | Bio-inspired genetic algorithm that evolves feature subsets through generations |
| **Multi-Method Comparison** | Benchmarks against Statistical Test, Information Theory, Recursive Elimination, and Tree-Based methods |
| **Interactive Visualization** | Real-time charts showing evolution progress, method comparison, and feature distribution |
| **CSV/Excel Support** | Upload datasets in CSV, XLSX, or XLS formats |
| **Configurable Parameters** | Customize population size, generations, and mutation rate |
| **Cross-Validation** | Uses 5-fold cross-validation for robust accuracy estimation |

### Algorithm Comparison

| Method | Approach | Strengths |
|--------|----------|-----------|
| **Evolutionary** | Genetic algorithm with selection, crossover, mutation | Explores complex feature interactions, finds non-obvious combinations |
| **Statistical Test** | ANOVA F-test | Fast, detects linear relationships, statistically rigorous |
| **Information Theory** | Mutual Information | Captures non-linear dependencies, information-theoretic basis |
| **Recursive Elimination** | Backward elimination with Random Forest | Iterative refinement, model-aware selection |
| **Tree-Based** | Random Forest feature importance | Captures feature interactions, ensemble-based approach |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/evolutionary-feature-optimizer.git
cd evolutionary-feature-optimizer
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python server.py
```

The application will be available at `http://localhost:5000`

---

## 📖 Usage

### Web Interface

1. **Access the Platform**: Navigate to `http://localhost:5000`
2. **Upload Dataset**: Click on "Try Demo" and upload your CSV/Excel file
3. **Configure Parameters** (optional):
   - Population Size: 20-100 (default: 30)
   - Generations: 10-50 (default: 20)
   - Mutation Rate: 0.0-0.3 (default: 0.1)
4. **Start Optimization**: Click "Start Optimization" and wait for results
5. **Analyze Results**: Review accuracy metrics, feature selection, and comparison charts

### Python API

```python
from engines import EvolutionOptimizer, BaselineSelector
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop(columns=['target']).select_dtypes(include=[np.number]).values
y = df['target'].values

# Run evolutionary optimization
optimizer = EvolutionOptimizer(
    X, y,
    population_size=50,
    generations=30,
    mutation_rate=0.1
)

selected_features, accuracy, history = optimizer.evolve()

print(f"Selected {len(selected_features)} features")
print(f"Accuracy: {accuracy:.4f}")

# Compare with baseline methods
baseline = BaselineSelector(X, y)
comparison = baseline.compare_all(selected_features, accuracy)
```

---

## 🔌 API Endpoints

### POST `/api/preview`

Preview dataset structure before optimization.

**Request:**
- `file`: CSV or Excel file (multipart/form-data)

**Response:**
```json
{
  "columns": ["feature1", "feature2", ...],
  "total_rows": 150,
  "total_columns": 5
}
```

### POST `/api/optimize`

Run feature optimization on uploaded dataset.

**Request:**
- `file`: CSV or Excel file (multipart/form-data)
- `target_column`: Target column name (optional, defaults to last column)
- `population_size`: Integer (default: 50)
- `generations`: Integer (default: 30)
- `mutation_rate`: Float (default: 0.1)

**Response:**
```json
{
  "success": true,
  "dataset_info": {
    "total_rows": 150,
    "total_features": 4,
    "selected_features": 2
  },
  "evolutionary_optimization": {
    "accuracy": 0.95,
    "feature_count": 2,
    "selected_indices": [0, 2],
    "feature_names": ["feature1", "feature3"],
    "evolution_history": [...]
  },
  "baseline_comparison": {
    "methods": [...],
    "accuracy_scores": [...],
    "feature_counts": [...]
  }
}
```

---

## 🏗️ Architecture

### Project Structure

```
evolutionary-feature-optimizer/
├── server.py                 # Flask application and API endpoints
├── wsgi.py                   # WSGI entry point for production
├── requirements.txt          # Python dependencies
├── engines/
│   ├── __init__.py
│   ├── evolution_optimizer.py    # Evolutionary algorithm implementation
│   └── baseline_selector.py      # Traditional feature selection methods
├── templates/
│   ├── home.html            # Landing page
│   └── demo.html            # Interactive demo interface
├── storage/                  # Temporary file storage
└── README.md
```

### Key Components

#### EvolutionOptimizer

Implements genetic algorithm for feature selection:

- **Initialization**: Creates random population of feature subsets
- **Fitness Evaluation**: Uses cross-validated accuracy with penalty for excessive features
- **Selection**: Roulette wheel selection based on fitness
- **Crossover**: Single-point crossover between parent chromosomes
- **Mutation**: Random bit-flip mutation
- **Elitism**: Preserves best individual across generations

#### BaselineSelector

Provides traditional feature selection methods:

- **Statistical Test**: ANOVA F-test (linear relationships)
- **Information Theory**: Mutual Information (non-linear dependencies)
- **Recursive Elimination**: RFE with Random Forest
- **Tree-Based**: Random Forest feature importance

---

## 🧪 How It Works

### Evolutionary Algorithm Flow

```
1. Initialize Population
   └─> Generate random feature subsets (30-70% of features)

2. Evaluation Phase
   └─> Calculate fitness (cross-validated accuracy - penalty)

3. Selection Phase
   └─> Select parents using roulette wheel selection

4. Crossover Phase
   └─> Create offspring by combining parent features

5. Mutation Phase
   └─> Randomly flip feature selection bits

6. Elitism
   └─> Preserve best individual

7. Repeat for N generations
   └─> Return best feature subset
```

### Fitness Function

```
fitness = cross_val_accuracy - (feature_count / total_features) * 0.1
```

This formula balances:
- **Accuracy**: Higher is better
- **Parsimony**: Fewer features is better (penalty term)

---

## 📊 Example Results

### Typical Performance

| Metric | Value |
|--------|-------|
| **Dimensionality Reduction** | 30-70% |
| **Accuracy Improvement** | Maintained or improved |
| **Feature Count Reduction** | Significant reduction while preserving predictive power |
| **Processing Time** | 10-60 seconds (depends on dataset size) |

### Comparison Example

```
Method                  Accuracy    Features    Reduction
─────────────────────────────────────────────────────────
Evolutionary            95.2%       2          75.0%
Statistical Test        93.8%       3          62.5%
Information Theory      94.5%       2          75.0%
Recursive Elimination   92.1%       4          50.0%
Tree-Based              91.5%       5          37.5%
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):

```env
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
UPLOAD_FOLDER=storage
MAX_CONTENT_LENGTH=16777216
```

### Algorithm Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `population_size` | 50 | 20-100 | Number of individuals per generation |
| `generations` | 30 | 10-50 | Number of evolution cycles |
| `mutation_rate` | 0.1 | 0.0-0.3 | Probability of feature bit mutation |
| `crossover_rate` | 0.8 | 0.5-1.0 | Probability of parent crossover |

---

## 🚢 Deployment

### Production Setup with Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "wsgi:app"]
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Data processing with [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/)
- Visualization with [Chart.js](https://www.chartjs.org/)

---

## 📧 Contact

For questions, suggestions, or support, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ using Evolutionary Algorithms**

⭐ Star this repo if you find it useful!

</div>

