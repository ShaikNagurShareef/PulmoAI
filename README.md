# PulmoAI: Vision Language Transformer for Clinical Findings from Chest X-Rays

## Project Description

In this project, we designed a Vision Language Transformer to generate clinical findings from multi-view Chest X-Ray images. This innovative approach aims to improve medical decision-making and patient care by providing automated and accurate analysis of radiographic images.

## Project Structure

The project is organized into the following directories and files:

- **Icons**: Contains icon images used in the application.
- **service**: Includes the backend service code for handling model inference and other business logic.
- **static**: Stores static files such as CSS, JavaScript, and images.
- **templates**: Contains HTML templates for rendering the web interface.
- **Main.py**: The main entry point for the Flask application.
- **README.md**: This file, providing an overview of the project.
- **test.ipynb**: Jupyter notebook for testing and demonstrating the model and its capabilities.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Flask
- PyTorch or TensorFlow (depending on the framework used for the Vision Language Transformer)
- Jupyter Notebook (optional, for `test.ipynb`)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/VisionLanguageTransformer.git
    cd VisionLanguageTransformer
    ```

2. **Create a virtual environment and activate it**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the model**:
    - Download the pre-trained Vision Language Transformer model and place it in the appropriate directory as specified in `service` code.
    - Ensure you have the necessary Chest X-Ray image dataset for inference.

5. **Run the application**:
    ```bash
    python Main.py
    ```

### Usage

Once the application is running, navigate to `http://localhost:5000` in your web browser. You can upload Chest X-Ray images through the web interface, and the application will generate clinical findings based on the Vision Language Transformer model.

### Testing and Demonstration

To test and demonstrate the model's capabilities, open the `test.ipynb` notebook using Jupyter Notebook:
```bash
jupyter notebook test.ipynb
```
Follow the instructions in the notebook to run test cases and visualize the results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

