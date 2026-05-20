# Installation Guide - Crop Advisory System

## Quick Start (No Dependencies)
The main system (`demo_crop_system.py`) uses **only Python standard library** and requires no additional packages:

```bash
# Run immediately - no installation needed!
python3 demo_crop_system.py
```

## Advanced Features Installation

### Option 1: Minimal Installation (Recommended)
For the advanced ML features with Random Forest and Regression:

```bash
# Install minimal required packages
pip install -r requirements-minimal.txt

# Or install individually:
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Option 2: Full Installation
For complete development environment with all optional features:

```bash
# Install all packages
pip install -r requirements.txt
```

### Option 3: Virtual Environment (Recommended for Development)
```bash
# Create virtual environment
python3 -m venv crop_env

# Activate virtual environment
# On Linux/Mac:
source crop_env/bin/activate
# On Windows:
crop_env\Scripts\activate

# Install packages
pip install -r requirements-minimal.txt

# Run the system
python3 crop_recommendation.py
```

## System Compatibility

### ✅ Works Without Any Installation:
- `demo_crop_system.py` - Main comprehensive system
- `simple_crop_system.py` - Basic version

### 📦 Requires Packages:
- `crop_recommendation.py` - Advanced ML models version

## Troubleshooting

### If you get "ModuleNotFoundError":
```bash
# Make sure you're in the right directory
cd crop_advisory_system

# Install minimal requirements
pip install pandas numpy scikit-learn matplotlib seaborn

# Try running again
python3 crop_recommendation.py
```

### For WSL/Linux Users:
```bash
# If pip3 is not found
sudo apt update
sudo apt install python3-pip

# Then install packages
pip3 install -r requirements-minimal.txt
```

### For Windows Users:
```bash
# Use pip instead of pip3
pip install -r requirements-minimal.txt
```

## Package Versions
Tested with:
- Python 3.8+
- pandas 2.0+
- numpy 1.24+
- scikit-learn 1.3+
- matplotlib 3.7+
- seaborn 0.12+

## No Internet? No Problem!
The `demo_crop_system.py` file works completely offline using only Python's built-in libraries. Perfect for demonstration purposes!