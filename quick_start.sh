#!/bin/bash

# Face Recognition System - Quick Start Script
# This script helps you quickly set up and run the face recognition system.

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${YELLOW}==================================${NC}"
echo -e "${YELLOW}  Face Recognition System Setup    ${NC}"
echo -e "${YELLOW}==================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 is required but not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}pip3 is required but not installed. Please install pip3.${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${GREEN}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "\n${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate  # Linux/Mac
# On Windows, use: .\venv\Scripts\activate

# Upgrade pip
echo -e "\n${GREEN}Upgrading pip and setuptools...${NC}"
pip install --upgrade pip setuptools wheel

# Install dependencies
echo -e "\n${GREEN}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install development dependencies
echo -e "\n${GREEN}Installing development dependencies...${NC}"
pip install -e ".[dev]"

# Create necessary directories
echo -e "\n${GREEN}Setting up directories...${NC}"
mkdir -p data/raw data/processed models

# Download pre-trained face detection model (if not exists)
if [ ! -f "models/haarcascade_frontalface_default.xml" ]; then
    echo -e "\n${GREEN}Downloading pre-trained face detection model...${NC}"
    wget -P models/ https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
fi

# Download sample dataset (if needed)
if [ "$1" == "--download-sample" ]; then
    echo -e "\n${GREEN}Downloading sample dataset...${NC}"
    mkdir -p data/raw
    # Add code to download sample dataset if available
    echo -e "${YELLOW}Sample dataset download not yet implemented. Please add images manually.${NC}"
fi

# Check if there's any training data
if [ -z "$(ls -A data/raw 2>/dev/null 2>&1)" ]; then
    echo -e "\n${YELLOW}No training data found in data/raw/ directory.${NC}"
    echo -e "You can add students and capture images using the web interface."
    echo -e "\nTo add training data manually, use this structure:"
    echo -e "  data/raw/person1/image1.jpg"
    echo -e "  data/raw/person1/image2.jpg"
    echo -e "  data/raw/person2/image1.jpg"
    echo -e "  ..."
fi

echo -e "\n${GREEN}Setup complete!${NC}"

# Function to open URL in default browser
open_browser() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$1"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "$1"
    else
        echo "Please open this URL in your browser: $1"
    fi
}

# Ask about opening the notebook
read -p "Do you want to open the Jupyter notebook? (y/n): " open_notebook
if [[ $open_notebook =~ ^[Yy]$ ]]; then
    echo -e "\n${GREEN}Starting Jupyter notebook...${NC}"
    jupyter notebook --no-browser --port=8889 &
    sleep 2
    open_browser "http://localhost:8889/tree"
    echo -e "Jupyter notebook is now running in your browser."
    exit 0
fi

# Ask about training the model
read -p "Do you want to train the face recognition model? (y/n): " train_model
if [[ $train_model =~ ^[Yy]$ ]]; then
    echo -e "\n${GREEN}Starting model training...${NC}"
    python src/train_face_recognition.py
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}Model training completed successfully!${NC}"
    else
        echo -e "\n${YELLOW}Model training failed. Please check the error messages above.${NC}"
    fi
fi

# Ask about running the demo application
read -p "Do you want to run the demo application? (y/n): " run_demo
if [[ $run_demo =~ ^[Yy]$ ]]; then
    echo -e "\n${GREEN}Starting the demo application...${NC}"
    echo -e "Press Ctrl+C to stop the application when done.\n"
    streamlit run app.py
else
    echo -e "\n${YELLOW}You can start the application later by running:${NC}"
    echo -e "   source venv/bin/activate"
    echo -e "   streamlit run app.py"
fi

echo -e "\n${GREEN}All done!${NC}"
