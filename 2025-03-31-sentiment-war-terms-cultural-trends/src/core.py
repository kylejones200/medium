"""Core functions for sentiment analysis of war terms and cultural trends."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_sentiment_score(text_data: pd.Series, positive_words: list, negative_words: list) -> pd.Series:
    """Calculate sentiment score from text data."""
    scores = []
    for text in text_data:
        if pd.isna(text):
            scores.append(0)
            continue
        text_lower = str(text).lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        score = (positive_count - negative_count) / max(len(text_lower.split()), 1)
        scores.append(score)
    return pd.Series(scores, index=text_data.index)

def plot_sentiment_trend(sentiment: pd.Series, title: str, output_path: Path):
 """Plot sentiment trend over time """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if hasattr(sentiment.index, '__len__') and len(sentiment.index) > 0:
        if hasattr(sentiment.index[0], 'year'):
            ax.plot(sentiment.index, sentiment.values, color="#4A90A4", linewidth=1.2)
            ax.set_xlabel("Date")
        else:
            ax.plot(sentiment.values, color="#4A90A4", linewidth=1.2)
            ax.set_xlabel("Time")
    else:
        ax.plot(sentiment.values, color="#4A90A4", linewidth=1.2)
        ax.set_xlabel("Time")
    
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Sentiment Score")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_word_frequency(word_counts: Dict[str, int], title: str, output_path: Path, top_n: int = 10):
 """Plot word frequency """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*sorted_words)
    
    ax.barh(words, counts, color="#4A90A4", alpha=0.7, edgecolor='none')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

