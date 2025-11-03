from pathlib import Path
from gui import DigitDrawingApp, start_app, ModelType
from ImageRecognition.models import evaluate_models
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from enum import Enum
import os
import re

def generate_and_save_report_plots(
    accuracy: float, 
    model_type: ModelType, 
    report_text: str,
    save_dir: str = 'evaluation_plots'
):
    """
    Parses a scikit-learn classification report and generates visualization plots.

    Args:
        accuracy (float): The overall single-digit test accuracy (e.g., 0.9869).
        model_type (ModelType): The model type (e.g., ModelType.CNN).
        report_text (str): The classification report string.
        save_dir (str): Directory where the plots will be saved.
    """
    
    model_name = model_type.value
    
    # 1. Prepare Directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 2. Parse the Classification Report into a DataFrame
    # We use StringIO to treat the string as a file and read it into a DataFrame
    # The report contains unneeded header/footer lines; we skip them.
    
    # Clean up and load the report text
    # The scikit-learn report format is tricky; we capture the main digit rows.
    report_data = report_text.split('\n')
    data_lines = [line.strip() for line in report_data if line.strip().startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))]
    
    if not data_lines:
        print("Error: Could not parse classification report lines.")
        return
        
    # Combine relevant lines and read into DataFrame
    # 2. Parse the Classification Report into a DataFrame
    # Use StringIO to treat the string as a file
    report_io = StringIO(report_text)

    # Read the entire report, separating columns by 2 or more spaces.
    # This is much more reliable than trying to filter lines by starting digit.
    df = pd.read_csv(
    report_io, 
    sep=r'\s{2,}', # Separator: 2 or more spaces
    engine='python',
    header=0,
    skipinitialspace=True)

    # Rename the first column, which contains the labels/metrics names
    df.rename(columns={df.columns[0]: 'label'}, inplace=True)
    df.set_index('label', inplace=True)
    # Remove rows that are entirely NaN (often blank lines or headers)
    df.dropna(how='all', inplace=True)
    
    # Fallback if 'precision' not parsed correctly
    if 'precision' not in df.columns:
        rows = []
        pattern = re.compile(
            r'^(?P<label>(\d|macro avg|weighted avg|micro avg))\s+'
            r'(?P<precision>\d\.\d+)\s+'
            r'(?P<recall>\d\.\d+)\s+'
            r'(?P<f1_score>\d\.\d+)\s+'
            r'(?P<support>\d+)$'
        )

        for line in report_text.splitlines():
            line = line.strip()
            m = pattern.match(line)
            if m:
                rows.append(m.groupdict())

        if rows:
            df = pd.DataFrame(rows)
            df.rename(columns={'f1_score': 'f1-score'}, inplace=True)
            df.set_index('label', inplace=True)
            for col in ['precision', 'recall', 'f1-score']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['support'] = pd.to_numeric(df['support'], errors='coerce')
        else:
            print(f" Could not parse classification report for {model_name}")
            return
    


    # 3. Filter the DataFrame for Digit Classes and Aggregate Averages
    try:
        # Filter for the 10 digit classes (indices '0' through '9')
        # We use df.index.str.isdigit() to robustly find the digit rows
        df.index = df.index.astype(str)
        class_df = df[df.index.str.isdigit()].copy()
        
        # Convert metric columns to float, handling potential parsing errors
        for col in ['precision', 'recall', 'f1-score']:
            class_df[col] = pd.to_numeric(class_df[col], errors='coerce')
            
    except Exception as e:
        print(f"Error during DataFrame filtering: {e}")
        print("Check the format of your classification report text.")
        return
    
    # Check if the class_df is empty before plotting
    if class_df.empty:
        print(f"Error: Parsed class data is empty. Cannot plot for {model_name}.")
        return
    plt.figure(figsize=(10, 6))
    class_df['f1-score'].plot(kind='bar', color='skyblue') 
    try:
        weighted_f1 = df.loc['weighted avg', 'f1-score']
    except KeyError:
        weighted_f1 = accuracy # Fallback if 'weighted avg' row was missed

    plt.title(f'F1-Score per Digit Class for {model_name}')
    plt.xlabel('Digit Class')
    plt.ylabel('F1-Score')
    plt.ylim(0.85, 1.0) # Set a sensible Y limit for high-accuracy models
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot1_path = os.path.join(save_dir, f'{model_name}_f1_per_class.png')
    plt.savefig(plot1_path)
    plt.close()
    print(f"✅ Saved F1-Score per class plot to {plot1_path}")
        
        
    # --- Visualization 2: Summary Metrics (Accuracy vs. Weighted F1) ---
        
    # Get the weighted F1-Score from the DataFrame
    try:
        weighted_f1 = df.loc['weighted avg', 'f1-score']
    except KeyError:
        # Fallback if parsing missed the footer
        weighted_f1 = accuracy # Using accuracy as a fallback proxy
        print("Warning: Could not find 'weighted avg' F1 score in report; using accuracy.")
        
    summary_data = {
        'Metric': ['Accuracy', 'Weighted F1-Score'],
        'Value': [accuracy / 100.0, weighted_f1]
    }
    summary_df = pd.DataFrame(summary_data)
    
    plt.figure(figsize=(6, 5))
    plt.bar(summary_df['Metric'], summary_df['Value'], color=['teal', 'coral'])
    plt.title(f'Overall Performance for {model_name}')
    plt.ylabel('Score')
    plt.ylim(0.9, 1.0) # Set a sensible Y limit
    
    plot2_path = os.path.join(save_dir, f'{model_name}_summary_metrics.png')
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved summary metrics plot to {plot2_path}")


def generate_accuracy_plot(
    accuracy_data: dict[ModelType, float], 
    save_dir: str = 'evaluation_plots'
):
    """
    Generates a bar chart comparing the overall accuracy of multiple models 
    and saves the plot.

    Args:
        accuracy_data (dict[ModelType, float]): A dictionary mapping the 
            ModelType to its overall test accuracy (e.g., 0.9869).
        save_dir (str): Directory where the plot will be saved.
    """
    
    model_names = [mt.value.upper() for mt in accuracy_data.keys()] # Changed to .upper() for labels
    accuracies = [v / 100.0 for v in accuracy_data.values()]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(8, 6))
    
    bars = plt.bar(model_names, accuracies, color=['teal', 'coral', 'skyblue', 'mediumseagreen'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0005, # Adjusted text position slightly
                 f'{yval:.2%}', ha='center', va='bottom')

    plt.title('Comparison of Overall Model Accuracy')
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy Score')
    
    # --- IMPROVEMENTS FOR SCALE ---
    # 1. Dynamically adjust Y-axis start based on min accuracy, or set a tight range
    min_acc = min(accuracies) if accuracies else 0.90
    # Set ylim to start slightly below the lowest accuracy, or at a fixed sensible value
    plt.ylim(max(0.975, min_acc - 0.005), 1.002) # Start closer to results, go slightly above 1.00
    
    # 2. Add more noticeable Y-axis ticks and grid lines
    # Generate ticks every 0.001 (0.1%) or 0.002 (0.2%) within the range
    start_tick = int(plt.ylim()[0] * 1000) / 1000.0 # Round down to nearest 0.001
    end_tick = int(plt.ylim()[1] * 1000) / 1000.0 # Round up to nearest 0.001
    
    # Create ticks at every 0.001 (0.1%) interval
    yticks = [i / 1000.0 for i in range(int(start_tick * 1000), int(end_tick * 1000) + 3, 1)] # Generate ticks
    
    plt.yticks(yticks) # Apply the custom ticks
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Keep grid lines
    # --- END IMPROVEMENTS ---
    
    plot_path = os.path.join(save_dir, 'overall_model_accuracy_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved accuracy comparison plot to {plot_path}")


app, app_thread = start_app()
try:
    with open("result_doc.txt", "w", encoding="utf-8") as f:
        results = ("#"*5)+" TESTING "+("#"*5)+("\n"*3)
        print(results)
        accuracy_map = {}
        for accuracy, model_type in app.run_tests():
            to_add = f"{model_type}: {accuracy}% accuracy\n\n"
            accuracy_map[model_type] = accuracy
            print(to_add)
            results += to_add
        evaluation_header = ("#"*5)+" EVALUATION "+("#"*5)+("\n"*3)
        print(evaluation_header)
        for result, model_type, report in evaluate_models({
            "cnn": app.image_rec.models[ModelType.CNN],
            "mlp": app.image_rec.models[ModelType.MLP],
            "svc": app.image_rec.models[ModelType.SVC]
            }):
            print(result)
            results += result
            key_match: dict[str, ModelType] = {
                "cnn": ModelType.CNN,
                "mlp": ModelType.MLP,
                "svc": ModelType.SVC
            }
            df = pd.DataFrame(report)
            df.to_csv(f"reports/{model_type}_classification_report.csv")
            accuracy = accuracy_map[key_match[model_type.lower()]]
            model_key = key_match[model_type.lower()]
        f.write(results)
except Exception as e:
        print(e)