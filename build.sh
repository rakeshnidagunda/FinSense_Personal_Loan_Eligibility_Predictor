#!/bin/bash
set -e
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Retraining model to match installed scikit-learn version..."
python retrain.py

echo "Build complete!"
