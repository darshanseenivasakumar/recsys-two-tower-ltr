"""Canonical column names used by every stage of the system.

Training, evaluation and serving all import these constants instead of typing string
literals. A renamed column then fails loudly everywhere at once, rather than silently
diverging between the training and serving code (one source of training-serving skew).
"""

from typing import Final

# Identifiers. *_id = the dataset's raw ID, *_idx = our contiguous 0..N-1 index
# (what embedding tables and sparse matrices are indexed by).
USER_ID: Final = "user_id"
ITEM_ID: Final = "item_id"
USER_IDX: Final = "user_idx"
ITEM_IDX: Final = "item_idx"

# Interaction columns.
RATING: Final = "rating"
TIMESTAMP: Final = "timestamp"  # seconds since the Unix epoch (UTC)
IS_POSITIVE: Final = "is_positive"  # implicit-feedback label: rating >= threshold

# Item metadata.
TITLE: Final = "title"
GENRES: Final = "genres"  # list[str]
YEAR: Final = "year"  # release year parsed from the title; nullable

# User metadata (MovieLens-1M only; absent for ML-25M).
GENDER: Final = "gender"
AGE: Final = "age"  # ML-1M age *bucket* code (1, 18, 25, ...), categorical, not numeric
OCCUPATION: Final = "occupation"  # categorical code 0..20
