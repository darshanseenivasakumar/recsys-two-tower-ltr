"""Feature construction shared by training, offline evaluation and online serving.

Every model input is produced by a function in this package. Training and serving import
the same functions, so they cannot drift apart (training-serving skew).
"""
