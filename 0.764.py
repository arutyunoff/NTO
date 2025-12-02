import os
import sys
import time
import argparse
import joblib
import gc
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from tqdm.notebook import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

class constants:
    # --- FILENAMES ---
    TRAIN_FILENAME = "train.csv"
    TEST_FILENAME = "test.csv"
    USER_DATA_FILENAME = "users.csv"
    BOOK_DATA_FILENAME = "books.csv"
    BOOK_GENRES_FILENAME = "book_genres.csv"
    GENRES_FILENAME = "genres.csv"
    BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
    SUBMISSION_FILENAME = "submission.csv"
    TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
    BERT_EMBEDDINGS_FILENAME = "bert_embeddings.pkl"
    BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
    PROCESSED_DATA_FILENAME = "processed_features.parquet"

    # --- COLUMN NAMES ---
    # Main columns
    COL_USER_ID = "user_id"
    COL_BOOK_ID = "book_id"
    COL_TARGET = "rating"
    COL_SOURCE = "source"
    COL_PREDICTION = "rating_predict"
    COL_HAS_READ = "has_read"
    COL_TIMESTAMP = "timestamp"

    # Feature columns (newly created)
    F_USER_MEAN_RATING = "user_mean_rating"
    F_USER_RATINGS_COUNT = "user_ratings_count"
    F_BOOK_MEAN_RATING = "book_mean_rating"
    F_BOOK_RATINGS_COUNT = "book_ratings_count"
    F_AUTHOR_MEAN_RATING = "author_mean_rating"
    F_BOOK_GENRES_COUNT = "book_genres_count"

    # Metadata columns from raw data
    COL_GENDER = "gender"
    COL_AGE = "age"
    COL_AUTHOR_ID = "author_id"
    COL_PUBLICATION_YEAR = "publication_year"
    COL_LANGUAGE = "language"
    COL_PUBLISHER = "publisher"
    COL_AVG_RATING = "avg_rating"
    COL_GENRE_ID = "genre_id"
    COL_DESCRIPTION = "description"

    # --- VALUES ---
    VAL_SOURCE_TRAIN = "train"
    VAL_SOURCE_TEST = "test"

    # --- MAGIC NUMBERS ---
    MISSING_CAT_VALUE = "-1"
    MISSING_NUM_VALUE = -1
    PREDICTION_MIN_VALUE = 0
    PREDICTION_MAX_VALUE = 10

# %%
class config:
    # --- DIRECTORIES ---
    # Настраиваем пути для работы в текущей директории (Colab/Kaggle)
    ROOT_DIR = Path(".")
    DATA_DIR = ROOT_DIR
    RAW_DATA_DIR = DATA_DIR
    INTERIM_DATA_DIR = DATA_DIR
    PROCESSED_DATA_DIR = DATA_DIR
    OUTPUT_DIR = ROOT_DIR / "output"
    MODEL_DIR = OUTPUT_DIR / "models"
    SUBMISSION_DIR = OUTPUT_DIR / "submissions"

    # --- PARAMETERS ---
    RANDOM_STATE = 42
    TARGET = constants.COL_TARGET

    # --- TEMPORAL SPLIT CONFIG ---
    # 0.8 means 80% of data points (by timestamp) go to train, 20% to validation
    TEMPORAL_SPLIT_RATIO = 0.8

    # --- TRAINING CONFIG ---
    EARLY_STOPPING_ROUNDS = 50
    MODEL_FILENAME = "lgb_model.txt"

    # --- TF-IDF PARAMETERS ---
    TFIDF_MAX_FEATURES = 500
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95
    TFIDF_NGRAM_RANGE = (1, 2)

    # --- BERT PARAMETERS ---
    BERT_MODEL_NAME = constants.BERT_MODEL_NAME
    BERT_BATCH_SIZE = 8
    BERT_MAX_LENGTH = 512
    BERT_EMBEDDING_DIM = 768
    BERT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BERT_GPU_MEMORY_FRACTION = 0.75

    # --- FEATURES ---
    CAT_FEATURES = [
        constants.COL_USER_ID,
        constants.COL_BOOK_ID,
        constants.COL_GENDER,
        constants.COL_AGE,
        constants.COL_AUTHOR_ID,
        constants.COL_PUBLICATION_YEAR,
        constants.COL_LANGUAGE,
        constants.COL_PUBLISHER,
    ]

    # --- MODEL PARAMETERS ---
    LGB_PARAMS = {
        "objective": "rmse",
        "metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "n_jobs": -1,
        "seed": RANDOM_STATE,
        "boosting_type": "gbdt",
    }

    LGB_FIT_PARAMS = {
        "eval_metric": "rmse",
        "callbacks": [],
    }

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category" and "datetime" not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    return df



def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category" and "datetime" not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    return df


# --- TEMPORAL SPLIT UTILS ---

def temporal_split_by_date(
    df: pd.DataFrame, split_date: pd.Timestamp, timestamp_col: str = constants.COL_TIMESTAMP
) -> Tuple[pd.Series, pd.Series]:
    """Splits DataFrame into train and validation sets based on absolute date threshold."""
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    train_mask = df[timestamp_col] <= split_date
    val_mask = df[timestamp_col] > split_date

    if train_mask.sum() == 0:
        raise ValueError(f"No records found with timestamp <= {split_date}.")
    if val_mask.sum() == 0:
        raise ValueError(f"No records found with timestamp > {split_date}.")

    if train_mask.sum() > 0 and val_mask.sum() > 0:
        max_train_timestamp = df.loc[train_mask, timestamp_col].max()
        min_val_timestamp = df.loc[val_mask, timestamp_col].min()

        if min_val_timestamp <= max_train_timestamp:
            raise ValueError(
                f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
                f"is not greater than max train timestamp ({max_train_timestamp})."
            )

    return train_mask, val_mask


def get_split_date_from_ratio(
    df: pd.DataFrame, ratio: float, timestamp_col: str = constants.COL_TIMESTAMP
) -> pd.Timestamp:
    """Calculates split date based on ratio of data points."""
    if not 0 < ratio < 1:
        raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    sorted_timestamps = df[timestamp_col].sort_values()
    threshold_index = int(len(sorted_timestamps) * ratio)

    return sorted_timestamps.iloc[threshold_index]

# %%
# --- DATA LOADING UTILS ---

def load_and_merge_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw data files and merges them into a single DataFrame."""
    print("Loading data...")

    dtype_spec: Dict[str, Any] = {
        constants.COL_USER_ID: "int32",
        constants.COL_BOOK_ID: "int32",
        constants.COL_TARGET: "float32",
        constants.COL_GENDER: "category",
        constants.COL_AGE: "float32",
        constants.COL_AUTHOR_ID: "int32",
        constants.COL_PUBLICATION_YEAR: "float32",
        constants.COL_LANGUAGE: "category",
        constants.COL_PUBLISHER: "category",
        constants.COL_AVG_RATING: "float32",
        constants.COL_GENRE_ID: "int16",
    }

    train_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TRAIN_FILENAME,
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k in [constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_TARGET]
        },
        parse_dates=[constants.COL_TIMESTAMP],
    )

    initial_count = len(train_df)
    train_df = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    print(f"Filtered training data: {initial_count} -> {len(train_df)} rows (only has_read=1)")

    test_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TEST_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_USER_ID, constants.COL_BOOK_ID]},
    )
    user_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.USER_DATA_FILENAME,
        dtype={
            k: v for k, v in dtype_spec.items() if k in [constants.COL_USER_ID, constants.COL_GENDER, constants.COL_AGE]
        },
    )
    book_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME,
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k in [
                constants.COL_BOOK_ID, constants.COL_AUTHOR_ID, constants.COL_PUBLICATION_YEAR,
                constants.COL_LANGUAGE, constants.COL_AVG_RATING, constants.COL_PUBLISHER,
            ]
        },
    )
    book_genres_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_BOOK_ID, constants.COL_GENRE_ID]},
    )
    genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.GENRES_FILENAME)
    book_descriptions_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DESCRIPTIONS_FILENAME,
        dtype={constants.COL_BOOK_ID: "int32"},
    )

    print("Data loaded. Merging datasets...")

    train_df[constants.COL_SOURCE] = constants.VAL_SOURCE_TRAIN
    test_df[constants.COL_SOURCE] = constants.VAL_SOURCE_TEST
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    combined_df = combined_df.merge(user_data_df, on=constants.COL_USER_ID, how="left")
    book_data_df = book_data_df.drop_duplicates(subset=[constants.COL_BOOK_ID])
    combined_df = combined_df.merge(book_data_df, on=constants.COL_BOOK_ID, how="left")

    print(f"Merged data shape: {combined_df.shape}")
    return combined_df, book_genres_df, genres_df, book_descriptions_df

# %% [markdown]
# ## 3. Генерация признаков (Feature Engineering)
# Код разделен на логические блоки: агрегации, жанры, TF-IDF, BERT.

# %%
def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features."""
    print("Adding aggregate features...")

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")

# %%
def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book."""
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")

# %%
def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds TF-IDF features from book descriptions."""
    print("Adding text features (TF-IDF)...")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    train_books = train_df[constants.COL_BOOK_ID].unique()
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    if vectorizer_path.exists():
        print(f"Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("Fitting TF-IDF vectorizer on training descriptions...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )

    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")
    tfidf_matrix = vectorizer.transform(df_descriptions)

    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    print(f"Added {len(tfidf_feature_names)} TF-IDF features.")
    return df_with_tfidf

# %%
def add_bert_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds BERT embeddings from book descriptions."""
    print("Adding text features (BERT embeddings)...")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME

    if embeddings_path.exists():
        print(f"Loading cached BERT embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing BERT embeddings (this may take a while)...")
        print(f"Using device: {config.BERT_DEVICE}")

        if config.BERT_DEVICE == "cuda":
            torch.cuda.set_per_process_memory_fraction(config.BERT_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.BERT_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        model.to(config.BERT_DEVICE)
        model.eval()

        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")
        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()

        embeddings_dict = {}
        num_batches = (len(descriptions) + config.BERT_BATCH_SIZE - 1) // config.BERT_BATCH_SIZE

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing BERT batches", unit="batch"):
                start_idx = batch_idx * config.BERT_BATCH_SIZE
                end_idx = min(start_idx + config.BERT_BATCH_SIZE, len(descriptions))
                batch_descriptions = descriptions[start_idx:end_idx]
                batch_book_ids = book_ids[start_idx:end_idx]

                encoded = tokenizer(
                    batch_descriptions, padding=True, truncation=True,
                    max_length=config.BERT_MAX_LENGTH, return_tensors="pt"
                )
                encoded = {k: v.to(config.BERT_DEVICE) for k, v in encoded.items()}
                outputs = model(**encoded)

                attention_mask = encoded["attention_mask"]
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                batch_embeddings = mean_pooled.cpu().numpy()

                for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                    embeddings_dict[book_id] = embedding

                if config.BERT_DEVICE == "cuda":
                    time.sleep(0.2)

        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved BERT embeddings to {embeddings_path}")

    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            embeddings_list.append(np.zeros(config.BERT_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)
    bert_feature_names = [f"bert_{i}" for i in range(config.BERT_EMBEDDING_DIM)]
    bert_df = pd.DataFrame(embeddings_array, columns=bert_feature_names, index=df.index)
    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)
    print(f"Added {len(bert_feature_names)} BERT features.")
    return df_with_bert

# %%
def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values using a defined strategy."""
    print("Handling missing values...")

    global_mean = train_df[config.TARGET].mean()
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    return df

# %%
def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    """Runs the full feature engineering pipeline."""
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    df = add_bert_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)

    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df

# %% [markdown]
# ## 4.1. Подготовка данных (Prepare Step)
# Загружаем данные и генерируем базовые признаки (без агрегаций, чтобы избежать утечек).

# %%
def prepare_data() -> None:
    """Processes raw data and saves prepared features to processed directory."""
    print("=" * 60)
    print("Data Preparation Pipeline")
    print("=" * 60)

    merged_df, book_genres_df, _, descriptions_df = load_and_merge_data()

    # Apply feature engineering WITHOUT aggregates
    featured_df = create_features(merged_df, book_genres_df, descriptions_df, include_aggregates=False)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    print(f"\nSaving processed data to {processed_path}...")
    featured_df.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")
    print("Processed data saved successfully!")

# %%
prepare_data()

# %% [markdown]
# ## 4.2. Обучение модели (Train Step)
# Обучаем LightGBM, используя Temporal Split.

# %%
def train() -> None:
    """Runs the model training pipeline with temporal split (Memory Optimized)."""
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}.")

    print(f"Loading prepared data from {processed_path}...")

    # OPTIMIZATION: Filter at load time or immediately after
    try:
        featured_df = pd.read_parquet(
            processed_path,
            engine="pyarrow",
            filters=[(constants.COL_SOURCE, '==', constants.VAL_SOURCE_TRAIN)]
        )
    except Exception:
        featured_df = pd.read_parquet(processed_path, engine="pyarrow")
        featured_df = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN]

    print(f"Loaded {len(featured_df):,} rows. Optimizing memory...")
    featured_df = reduce_mem_usage(featured_df)
    gc.collect()

    # Use featured_df as train_set
    train_set = featured_df

    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(f"Timestamp column '{constants.COL_TIMESTAMP}' not found.")

    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    # Split data
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Cleanup parent dataframe
    del train_set, featured_df
    gc.collect()

    # Verify temporal correctness
    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
    print(f"Max train timestamp: {max_train_timestamp}")
    print(f"Min validation timestamp: {min_val_timestamp}")

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError("Temporal split validation failed.")
    print("✅ Temporal split validation passed")

    print("\nComputing aggregate features on train split only...")
    # Compute aggregates
    train_split_with_agg = add_aggregate_features(train_split, train_split)
    val_split_with_agg = add_aggregate_features(val_split, train_split)

    del train_split, val_split
    gc.collect()

    print("Handling missing values...")
    # Use train_split_with_agg for filling stats (same target stats as train_split)
    train_split_final = handle_missing_values(train_split_with_agg, train_split_with_agg)
    val_split_final = handle_missing_values(val_split_with_agg, train_split_with_agg)

    del train_split_with_agg, val_split_with_agg
    gc.collect()

    # Define features
    exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP]
    features = [col for col in train_split_final.columns if col not in exclude_cols]
    non_feature_object_cols = train_split_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    print(f"Training features: {len(features)}")

    X_train = train_split_final[features]
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features]
    y_val = val_split_final[config.TARGET]

    # Cleanup before training
    del train_split_final, val_split_final
    gc.collect()

    print("\nTraining LightGBM model...")

    # Use lgb.Dataset with free_raw_data=True for memory optimization
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=config.CAT_FEATURES, free_raw_data=True)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=config.CAT_FEATURES, reference=train_data, free_raw_data=True)

    callbacks = [
        lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        config.LGB_PARAMS,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Evaluate
    print("\nEvaluating on validation set...")
    val_preds = model.predict(X_val)

    # Clip predictions
    val_preds = np.clip(val_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)

    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")


# %%
# ЗАПУСК ОБУЧЕНИЯ
train()

# %% [markdown]
# ## 4.3. Генерация предсказаний (Predict Step)
# Генерируем сабмит для тестовой выборки.

# %%
def predict() -> None:
    """Generates and saves predictions for the test set."""
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}.")

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")

    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    print("\nComputing aggregate features on all train data...")
    test_set_with_agg = add_aggregate_features(test_set.copy(), train_set)

    print("Handling missing values...")
    test_set_final = handle_missing_values(test_set_with_agg, train_set)

    exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP]
    features = [col for col in test_set_final.columns if col not in exclude_cols]
    non_feature_object_cols = test_set_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_test = test_set_final[features]

    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    print(f"\nLoading model from {model_path}...")
    model = lgb.Booster(model_file=str(model_path))

    print("Generating predictions...")
    test_preds = model.predict(X_test)
    clipped_preds = np.clip(test_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)

    submission_df = test_set[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    submission_df[constants.COL_PREDICTION] = clipped_preds

    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file created at: {submission_path}")

# %%
predict()

# %% [markdown]
# ## 4.4. Валидация формата (Validate Step)
# Проверяем, что файл сабмита соответствует требованиям.

# %%
def validate() -> None:
    """Validates the structure and format of the submission file."""
    print("Validating submission file...")
    try:
        test_df = pd.read_csv(config.RAW_DATA_DIR / constants.TEST_FILENAME)
        sub_df = pd.read_csv(config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME)

        assert len(sub_df) == len(test_df), f"Length mismatch. Expected {len(test_df)}, got {len(sub_df)}."
        assert not sub_df[constants.COL_PREDICTION].isna().any(), "Missing values in prediction."

        test_keys = test_df[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy().set_index([constants.COL_USER_ID, constants.COL_BOOK_ID])
        sub_keys = sub_df[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy().set_index([constants.COL_USER_ID, constants.COL_BOOK_ID])
        assert test_keys.index.equals(sub_keys.index), "User/Book pairs do not match test set."

        assert sub_df[constants.COL_PREDICTION].between(constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE).all(), "Predictions out of range."

        print("\nValidation successful! Submission file is valid.")
    except Exception as e:
        print(f"Validation failed: {e}")

# %%
validate()

# %% [markdown]
# ## 5. Оценка качества (Evaluation)
# Локальная проверка качества, если вы создали test выборку и для неё файл с правильными ответами solution.csv

# %%
def validate_submission_format_for_eval(df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    """Validate submission file format against solution."""
    if df.empty:
        raise ValueError("Submission file is empty")

    required_cols = {"user_id", "book_id", "rating_predict"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}. Expected: {required_cols}")

    if df.shape[0] != solution_df.shape[0]:
        raise ValueError(f"Row count mismatch: {df.shape[0]} in submission, {solution_df.shape[0]} expected")

def calculate_stage1_metrics(merged_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate RMSE, MAE, and Score metrics."""
    if merged_df.empty:
        return {"Score": 0.0, "RMSE": 0.0, "MAE": 0.0}

    y_true = merged_df["rating"]
    y_pred = merged_df["rating_predict"].clip(0, 10)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    range_width = 10.0
    mae_norm = mae / range_width
    rmse_norm = rmse / range_width
    score = 1 - (0.5 * rmse_norm + 0.5 * mae_norm)

    return {"Score": score, "RMSE": rmse, "MAE": mae}

def evaluate_submission(submission_path: str, solution_path: str) -> Optional[Dict[str, float]]:
    """Main evaluation function."""
    print(f"Evaluating {submission_path} against {solution_path}...")
    try:
        submission = pd.read_csv(submission_path)
        solution = pd.read_csv(solution_path)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
        return None

    try:
        validate_submission_format_for_eval(submission, solution)
    except ValueError as e:
        print(f"Validation error: {e}")
        return None

    solution_public = solution[solution["stage"] == "public"].copy()
    solution_private = solution[solution["stage"] == "private"].copy()

    public_merged = submission.merge(solution_public, on=["user_id", "book_id"], how="inner")
    private_merged = submission.merge(solution_private, on=["user_id", "book_id"], how="inner")

    public_metrics = calculate_stage1_metrics(public_merged)
    private_metrics = calculate_stage1_metrics(private_merged)

    print("--- Public ---")
    for metric, value in public_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\n--- Private ---")
    for metric, value in private_metrics.items():
        print(f"{metric}: {value:.6f}")

    return {
        "public_score": public_metrics["Score"],
        "private_score": private_metrics["Score"],
    }

# %%
# Пример запуска оценки (раскомментируйте, если у вас есть solution.csv)
solution_file = "solution.csv"
submission_file = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME
if os.path.exists(solution_file):
    evaluate_submission(submission_file, solution_file)

# %%



