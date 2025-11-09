#!/bin/bash

# Face Recognition System - Quick Start Script
# This script helps you quickly set up and run the face recognition system.

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print section header
print_section() {
    echo -e "\n${BLUE}==> $1${NC}"
}

# Function to print status
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[✓] $1${NC}"
    else
        echo -e "${RED}[✗] $1${NC}"
        exit 1
    fi
}

# Function to check directory structure
check_directories() {
    local dirs=("data/raw" "data/processed" "models" "reports/figures" "notebooks" "src")
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    touch notebooks/.gitkeep
}

# Print header
echo -e "${YELLOW}==================================${NC}"
echo -e "${YELLOW}  Face Recognition System Setup    ${NC}"
echo -e "${YELLOW}==================================${NC}"

# Check system requirements
print_section "Checking system requirements"

# Check Python
if ! command_exists python3; then
    echo -e "${RED}Python 3 is required but not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi
python_version=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
echo -e "${GREEN}Python $python_version detected${NC}"

# Check pip
if ! command_exists pip3; then
    echo -e "${YELLOW}pip3 not found. Attempting to install...${NC}"
    python3 -m ensurepip --upgrade || { echo -e "${RED}Failed to install pip. Please install it manually.${NC}"; exit 1; }
fi
echo -e "${GREEN}pip $(pip3 --version | awk '{print $2}') detected${NC}"

# Setup virtual environment
print_section "Setting up virtual environment"
if [ ! -d "venv" ]; then
    echo -e "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    echo -e "Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\nActivating virtual environment..."
source venv/bin/activate

# Upgrade pip and setuptools
echo -e "\nUpgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install dependencies
print_section "Installing dependencies"
pip install -r requirements.txt || { echo -e "${RED}Failed to install dependencies${NC}"; exit 1; }
pip install -e ".[dev]" || echo -e "${YELLOW}Warning: Could not install development dependencies${NC}"

# Setup directory structure
print_section "Setting up directory structure"
check_directories
print_status "Directory structure verified"

# Check for training data
if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo -e "\n${YELLOW}No training data found in data/raw/ directory.${NC}"
    echo -e "You can add students and capture images using the web interface."
    echo -e "\nTo add training data manually, use this structure:"
    echo -e "  data/raw/person1/image1.jpg"
    echo -e "  data/raw/person1/image2.jpg"
    echo -e "  data/raw/person2/image1.jpg"
fi

# Function to open URL in default browser
open_browser() {
    local url=$1
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$url"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "$url" || echo -e "${YELLOW}Could not open browser. Please visit: $url${NC}"
    else
        echo -e "\nPlease open this URL in your browser: $url"
    fi
}

# Main menu
while true; do
    echo -e "\n${BLUE}=== Face Recognition System ===${NC}"
    echo "1. Open Jupyter Notebook"
    echo "2. Train Model"
    echo "3. Run Web Application"
    echo "4. Exit"
    read -p "Choose an option (1-4): " choice

    case $choice in
        1)
            print_section "Setting up Jupyter Notebook"
            pip install -q ipykernel jupyter
            python -m ipykernel install --user --name=face_recognition_system \
                --display-name "Python (Face Recognition System)"
            echo -e "\n${GREEN}Starting Jupyter Notebook...${NC}"
            echo -e "Please select 'Python (Face Recognition System)' as the kernel"
            jupyter notebook notebooks/
            ;;
        2)
            print_section "Training Model"
            if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
                echo -e "${YELLOW}No training data found. Please add images to data/raw/ first.${NC}"
                continue
            fi
            python src/train_face_recognition.py
            if [ $? -eq 0 ]; then
                echo -e "\n${GREEN}Model training completed successfully!${NC}"
            else
                echo -e "\n${RED}Model training failed. Please check the error messages above.${NC}"
            fi
            ;;
        3)
            print_section "Starting Web Application"
            echo -e "${GREEN}Starting Streamlit application...${NC}"
            echo -e "Press Ctrl+C to stop the application when done.\n"
            streamlit run app.py
            ;;
        4)
            echo -e "\n${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}Invalid option. Please choose 1-4.${NC}"
            ;;
    esac
done