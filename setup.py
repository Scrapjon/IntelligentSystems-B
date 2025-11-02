import setuptools
from setuptools import setup, find_packages
import io # <-- Re-added io IMPORT for robustness

# Function to read the requirements from requirements.txt
def read_requirements():
    """Reads requirements from the requirements.txt file."""
    try:
        with open('requirements.txt') as f:
            return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        # Fallback to hardcoded list if requirements.txt is missing
        return [
            'numpy',
            'opencv-python',  # For cv2
            'Pillow',         # For PIL
            'torch',          # PyTorch
            'torchvision',    # PyTorch vision library for MNIST
            # tkinter is usually built-in, but dependencies are listed here
        ]

# Define long description robustly to prevent metadata-generation-failed error
try:
    with io.open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'A simple GUI application for drawing digits and predicting them using trained models.'


setup(
    name='digit-recognition-app',
    version='1.0.0',
    description='A Digit Recognition Application using PyTorch and traditional machine learning methods.',
    long_description=long_description, # <-- Now using the robustly read variable
    author='Oliver Moloney',
    author_email='104273068@student.swin.edu.au',
    url='https://github.com/Scrapjon/IntelligentSystems-B',
    
    # Automatically finds all packages/folders containing an __init__.py file
    packages=find_packages(),
    
    # Dependencies required for the package to run
    install_requires=read_requirements(),
    
    # Classification tags to categorize your package (optional but recommended)
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Education'
    ],
    
    # Define an entry point to run the GUI directly from the command line
    # This assumes your main execution file is gui.py inside the IntelligentSystems-B-main folder
    entry_points={
        'console_scripts': [
            # Running 'digit-app' in the terminal will execute the main method in gui.py
            'digit-app = IntelligentSystems-B-main.gui:main_run',
        ],
    },
    
    # Include non-code files (like model files) in the package
    # This assumes the ImageRecognition/model/ folder exists and contains .pth files.
    package_data={
        'IntelligentSystems-B-main': ['ImageRecognition/model/*.pth'],
    },
    include_package_data=True,
    
    python_requires='>=3.8',
)

# --- NOTE ---
# You need to define a main_run() function in IntelligentSystems-B-main/gui.py
# that initializes and starts the Tkinter event loop for the 'digit-app' command to work.
