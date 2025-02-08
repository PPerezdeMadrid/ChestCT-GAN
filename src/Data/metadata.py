import pandas as pd

def load_metadata(csv_path):
    """Load the metadata CSV file and return the dataframe."""
    try:
        metadata = pd.read_csv(csv_path)
        print(f"Successfully loaded metadata from {csv_path}")
        return metadata
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"File is empty: {csv_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error parsing the CSV file: {csv_path}")
        return None

def display_metadata_summary(metadata):
    """Display summary information and the first few rows of the metadata."""
    if metadata is not None:
        print("\n--- Metadata Summary ---")
        print(f"Columns: {metadata.columns.tolist()}")
        print(f"Number of records: {len(metadata)}")
        print("\nFirst 5 rows of the metadata:")
        print(metadata.head())

        # Display basic statistics (if applicable)
        print("\nBasic Statistics:")
        print(metadata.describe(include='all'))
    else:
        print("No metadata available to display.")

def filter_valid_study_descriptions(metadata):
    """Filter the metadata for valid study descriptions."""
    valid_study_descriptions = ["Chest", "ThoraxAThoraxRoutine Adult", "Chest 3D IMR", "Chest 3D"]
    filtered_metadata = metadata[metadata['Study Description'].isin(valid_study_descriptions)]
    
    print(f"\nFound {len(filtered_metadata)} valid study description records.")
    return filtered_metadata

def get_unique_study_id(metadata):
    """Get unique values from the 'Study UID' column."""
    if metadata is not None:
        unique_values = metadata['Study UID'].unique()
        print("\nUnique values in 'Study UID':")
        for value in unique_values:
            print(value)
    else:
        print("No metadata available to extract unique values.")

def get_unique_file_location(metadata):
    """Get unique values from the 'File Location' column."""
    if metadata is not None:
        unique_values = metadata['File Location'].unique()
        print("\nUnique values in 'File Location':")
        for value in unique_values:
            print(value)
    else:
        print("No metadata available to extract unique values.")


if __name__ == "__main__":
    # Specify the path to your metadata CSV file
    metadata_path = '../../../../ChestCT-NBIA/manifest-1608669183333/metadata.csv'  # Update this path
    
    # Load the metadata
    metadata = load_metadata(metadata_path)
    
    # Display summary of the metadata
    # display_metadata_summary(metadata)
    
    # Filter for valid study descriptions
    # valid_metadata = filter_valid_study_descriptions(metadata)
    # display_metadata_summary(valid_metadata)
    
    # Get unique values from the 'Study Description' column
    get_unique_study_id(metadata)
    get_unique_file_location(metadata)
