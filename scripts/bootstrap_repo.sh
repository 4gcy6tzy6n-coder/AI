#!/bin/bash

# Post Transformer AI Repository Bootstrap Script

set -e

echo "Bootstrapping Post Transformer AI repository..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/raw data/interim data/processed
mkdir -p artifacts/logs artifacts/checkpoints artifacts/traces

# Create .env file if not exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Post Transformer AI Environment Variables
DEBUG=true
LOG_LEVEL=INFO
DATA_DIR=./data
ARTIFACTS_DIR=./artifacts
EOF
fi

echo "Bootstrap complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run tests: make test"
echo "3. Start development!"
