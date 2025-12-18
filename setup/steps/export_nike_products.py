"""
Script to read Nike CSV file and save to MongoDB.
Includes only specific columns: url, name, sub_title, brand, color, price, currency, description, images
"""
import pandas as pd
from pathlib import Path
from pymongo import MongoClient, UpdateOne
from typing import List
from setup.config import SetupConfig


def __read_csv(file_path: str, include_columns: List[str]) -> pd.DataFrame:
    """
    Read Nike CSV file and select only specified columns.

    Args:
        file_path: Path to the CSV file
        include_columns: List of columns to include

    Returns:
        DataFrame with selected columns
    """
    print(f"Reading CSV file from {file_path}...")
    df = pd.read_csv(file_path)

    print(f"Total rows: {len(df)}")
    print(f"Available columns: {df.columns.tolist()}")

    # Select only the specified columns that exist in the dataframe
    available_columns = [col for col in include_columns if col in df.columns]
    missing_columns = [col for col in include_columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: The following columns are not in the CSV: {missing_columns}")

    selected_df = df[available_columns].copy()

    # Parse images column into array of strings
    if 'images' in selected_df.columns:
        print("Parsing 'images' column into array format...")
        selected_df['images'] = selected_df['images'].apply(
            lambda x: [url.strip() for url in str(x).split('|')] if pd.notna(x) else []
        )

    print(f"Selected columns: {available_columns}")

    return selected_df

def __save_to_mongodb(df: pd.DataFrame,
                    connection_string: str,
                    database_name: str,
                    collection_name: str) -> None:
    """
    Save data to MongoDB collection.

    Args:
        df: DataFrame to save
        connection_string: MongoDB connection string
        database_name: Name of the database
        collection_name: Name of the collection
    """
    print(f"Connecting to MongoDB...")
    client = MongoClient(connection_string)

    # Get database and collection
    db = client[database_name]
    collection = db[collection_name]

    # Convert DataFrame to list of dictionaries
    records = df.to_dict('records')

    print(f"Upserting {len(records)} documents into {database_name}.{collection_name}...")

    if records:
        # Upsert documents using url as unique identifier
        operations = [
            UpdateOne(
                {"url": record["url"]},  # Filter by url
                {"$set": record},         # Update with full record
                upsert=True               # Insert if not exists
            )
            for record in records
        ]
        result = collection.bulk_write(operations)
        print(f"Successfully upserted documents - Matched: {result.matched_count}, "
              f"Modified: {result.modified_count}, Inserted: {result.upserted_count}")
    else:
        print("No records to upsert")

    client.close()
    print("MongoDB connection closed")


def export_nike_products(config: SetupConfig):
    # Get path relative to this script's location
    script_dir = Path(__file__).parent
    data_file = script_dir / "data" / "nike_data.csv"

    # Read data
    selected_df = __read_csv(str(data_file), [
        "url",
        "name",
        "sub_title",
        "brand",
        "color",
        "price",
        "currency",
        "description",
        "images"
    ])

    # Save to MongoDB
    __save_to_mongodb(
        selected_df,
        connection_string=config.MONGO_URI,
        database_name=config.DATABASE_NAME,
        collection_name=config.PRODUCTS_COLLECTION
    )

    print("\n✓ Export process completed successfully!")
