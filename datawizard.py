import glob
import os
from typing import Any, Dict, List, Optional

import langchain
import langchain_openai
import matplotlib
import numpy as np
import openai
import pandas as pd
import seaborn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Verify imports
print("✅ All libraries imported successfully!")
print(f"📦 Pandas version: {pd.__version__}")
print(f"📦 LangChain version: {langchain.__version__}")
print(f"📦 NumPy version: {np.__version__}")
print("-" * 50)


# ============================================
# STEP 3: Create LangChain Tools
# ============================================
from langchain_core.tools import tool


#Tool1 - List csv files
@tool
def list_csv_files() -> Optional[List[str]]:
  """
  Lists all CSV files names in the local directory.
  Returns:
      Optional[List[str]]: A list of CSV file names or None if no files found.
      If no csv files are found, returns None.
  """
  csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))
  if not csv_files:
      return None
  return [os.path.basename(file) for file in csv_files]

# Test the tool
print("\n🔧 Testing Tool 1: list_csv_files")
print("Tool Name:", list_csv_files.name)
print("Tool Description:", list_csv_files.description)
print("Tool Arguments:", list_csv_files.args)
print("\n📁 CSV files found:", list_csv_files.invoke({}))
print("-" * 50)

# ============================================
# STEP 4: Dataset Caching Tool
# ============================================

DATAFRAME_CACHE = {}

@tool
def preload_datasets(paths: List[str]) -> str:
   """
    loads the csv files into a global cache if not already loaded.
    
    this function helps to efficiently manage datasets by loading them once
    and storing them in memory for future use. Without caching, you would
    waste tokens describing datasets contents repetatedly in agent responses.

    Args:
      paths: A list of file paths to CSV files.

    Returns:
     A message summarizing which datasets were loaded or already cached.
   """

   loaded = []
   cached = []

   for path in paths:
    if path not in DATAFRAME_CACHE:
      DATAFRAME_CACHE[path] = pd.read_csv(path)
      loaded.append(path)
    else:
      cached.append(path)

    return (
      f"Loaded datasets: {loaded}\n"
      f"Already cached: {cached}"
    )
    
# Test the tool
print("\n🔧 Testing Tool 2: preload_datasets")
result = preload_datasets.invoke({"paths": ["classification-dataset.csv", "regression-dataset.csv"]})
print(result)
print(f"\n💾 Datasets in cache: {list(DATAFRAME_CACHE.keys())}")
print("-" * 50)

@tool
def get_dataset_summaries(dataset_paths: List[str]) -> List[Dict[str,Any]]:
  """
  Analyzes multiple datasets and returns summaries for each.

  Args:
    dataset_paths (List[str]): A list of file paths to csv datasets.
  Returns:
    List[Dict[str,Any]]: 
     A list of summaries, one per dataset, each containing:
      - "file_name": The path of the dataset file.
      - "column_names": A list of column names in the dataset.
      - "data_types": A dictionary mapping column names of their data types.
  """
  summaries = []
  for path in dataset_paths:
    if path not in DATAFRAME_CACHE:
      DATAFRAME_CACHE[path] = pd.read_csv(path)

    df = DATAFRAME_CACHE[path]
    summary = {
      "file_name": path,
      "column_names": df.columns.tolist(),
      "data_types": df.dtypes.astype(str).to_dict(),
    }
    summaries.append(summary)
  
  return summaries
  
# Test the tool
print("\n🔧 Testing Tool 3: get_dataset_summaries")
summaries = get_dataset_summaries.invoke({
    "dataset_paths": ["classification-dataset.csv", "regression-dataset.csv"]
})

for summary in summaries:
    print(f"\n📊 Dataset: {summary['file_name']}")
    print(f"   Columns: {summary['column_names']}")
    print(f"   Data Types: {summary['data_types']}")
print("-" * 50)

# ============================================
# STEP 5: DataFrame Method Execution Tool
# ============================================

@tool
def run_method_on_dataframe(file_name: str, method: str) -> str:
  """
  Execute given method on a DataFrame and return the result on a dataset that has been loaded and
  cached using 'preload_datasets'.

  Args:
   file_name (str): the path or name of the dataset in the global cache.
   method (str): the name of the method to call on DataFrame. Only no-argument methods are supported
   (e.g. 'head', 'tail', 'describe', 'info')
  Returns:
    str: the output of the method as a formatted string, or an error message if dataset is not
    found or the method is invalid.
  Example:
    run_method_on_dataframe(file_name="data.csv", method="info")
  """
  # try to get DataFrame from cache or load it if not already cached
  if file_name not in DATAFRAME_CACHE:
    try:
      DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
    except FileNotFoundError:
      return f"DataFrame '{file_name}' not found in the cache or on disk."
    except Exception as e:
      return f"Error loading '{file_name}: {str(e)}"
  df = DATAFRAME_CACHE[file_name]

  # returns the callabale function from the df object by the name (method)
  # so func should be a callable function if it exists on df
  func = getattr(df, method, None)
  if not callable(func):
    return f"'{method}' is not a valid method of DataFrame"
  
  try:
    result = func()
    return str(result)
  except Exception as e:
    return "Error calling '{method}' on '{file_name}' : {str(e)}"


# Test the tool
result = run_method_on_dataframe.invoke({
  "file_name": 'regression-dataset.csv',
  "method": 'describe'
})


print(result)


# ============================================
# STEP 5: Model evaluation tools
# ============================================

@tool
def evaluate_classification_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    """
    Train and evaluate a classifier on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the classification target.
    Returns:
        Dict[str, float]: A dictionary with the model's accuracy score.
    """
    # Try to get the DataFrame from cache, or load it if not already cached
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
    
    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc}

@tool
def evaluate_regression_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    """
    Train and evaluate a regression model on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the regression target.
    Returns:
        Dict[str, float]: A dictionary with R² score and Mean Squared Error.
    """
    # Try to get the DataFrame from cache, or load it if not already cached
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
    
    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "r2_score": r2,
        "mean_squared_error": mse
    }