"""
Data utilities for fake news detection system
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
from datasets import load_dataset
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of fake news datasets"""
    
    def __init__(self, data_dir: str = "/app/ml_components/data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_liar_dataset(self) -> pd.DataFrame:
        """Load LIAR dataset from Hugging Face"""
        try:
            dataset = load_dataset("liar")
            
            # Combine train, validation, and test sets
            all_data = []
            for split in ['train', 'validation', 'test']:
                split_data = dataset[split].to_pandas()
                split_data['split'] = split
                all_data.append(split_data)
            
            df = pd.concat(all_data, ignore_index=True)
            
            # Map labels to binary (fake/real)
            # LIAR has 6-way classification: pants-fire, false, barely-true, half-true, mostly-true, true
            fake_labels = ['pants-fire', 'false', 'barely-true']
            df['is_fake'] = df['label'].isin(fake_labels).astype(int)
            
            logger.info(f"Loaded LIAR dataset: {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading LIAR dataset: {e}")
            return pd.DataFrame()
    
    def load_fakenews_dataset(self) -> pd.DataFrame:
        """Load fake news dataset from Hugging Face"""
        try:
            dataset = load_dataset("GonzaloA/fake_news")
            df = dataset['train'].to_pandas()
            
            # Map labels to binary
            df['is_fake'] = (df['label'] == 'FAKE').astype(int)
            
            logger.info(f"Loaded fake news dataset: {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading fake news dataset: {e}")
            return pd.DataFrame()
    
    def create_sample_multimodal_data(self) -> pd.DataFrame:
        """Create sample multimodal data for demonstration"""
        sample_data = {
            'text': [
                "Breaking: Scientists discover cure for aging in laboratory mice",
                "Local weather forecast shows sunny skies for the weekend",
                "SHOCKING: Celebrity caught in scandal that will amaze you",
                "Research shows importance of regular exercise for health",
                "You won't believe what happened next in this incredible story"
            ],
            'image_path': [
                None,  # No image
                None,  # No image
                None,  # No image
                None,  # No image
                None   # No image
            ],
            'is_fake': [1, 0, 1, 0, 1],  # 1 = fake, 0 = real
            'source': ['sample', 'sample', 'sample', 'sample', 'sample']
        }
        
        df = pd.DataFrame(sample_data)
        logger.info(f"Created sample multimodal dataset: {len(df)} samples")
        return df
    
    def combine_datasets(self) -> pd.DataFrame:
        """Combine multiple datasets into a unified format"""
        datasets = []
        
        # Load LIAR dataset
        liar_df = self.load_liar_dataset()
        if not liar_df.empty:
            liar_processed = pd.DataFrame({
                'text': liar_df['statement'],
                'image_path': None,
                'is_fake': liar_df['is_fake'],
                'source': 'liar',
                'additional_info': liar_df.apply(lambda x: {
                    'subject': x.get('subject', ''),
                    'speaker': x.get('speaker', ''),
                    'context': x.get('context', '')
                }, axis=1)
            })
            datasets.append(liar_processed)
        
        # Load fake news dataset
        fakenews_df = self.load_fakenews_dataset()
        if not fakenews_df.empty:
            fakenews_processed = pd.DataFrame({
                'text': fakenews_df['text'],
                'image_path': None,
                'is_fake': fakenews_df['is_fake'],
                'source': 'fakenews',
                'additional_info': fakenews_df.apply(lambda x: {
                    'title': x.get('title', ''),
                    'subject': x.get('subject', ''),
                    'date': x.get('date', '')
                }, axis=1)
            })
            datasets.append(fakenews_processed)
        
        # Add sample data
        sample_df = self.create_sample_multimodal_data()
        sample_processed = pd.DataFrame({
            'text': sample_df['text'],
            'image_path': sample_df['image_path'],
            'is_fake': sample_df['is_fake'],
            'source': sample_df['source'],
            'additional_info': [{}] * len(sample_df)
        })
        datasets.append(sample_processed)
        
        if datasets:
            combined_df = pd.concat(datasets, ignore_index=True)
            logger.info(f"Combined dataset: {len(combined_df)} total samples")
            
            # Save combined dataset
            save_path = os.path.join(self.data_dir, 'combined_dataset.csv')
            combined_df.to_csv(save_path, index=False)
            logger.info(f"Saved combined dataset to {save_path}")
            
            return combined_df
        else:
            logger.warning("No datasets could be loaded")
            return pd.DataFrame()
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the dataset"""
        stats = {
            'total_samples': len(df),
            'fake_samples': (df['is_fake'] == 1).sum(),
            'real_samples': (df['is_fake'] == 0).sum(),
            'sources': df['source'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'has_images': df['image_path'].notna().sum()
        }
        
        stats['fake_ratio'] = stats['fake_samples'] / stats['total_samples']
        
        return stats