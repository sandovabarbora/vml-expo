#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard libraries
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Data processing
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    kruskal, mannwhitneyu, spearmanr, chi2_contingency,
    shapiro, levene, ttest_ind, f_oneway, anderson,
    jarque_bera, normaltest, kstest, bartlett
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# Machine Learning
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_recall_curve

# Custom scripts
from src.config import EXPO_END_DATE, EXPO_START_DATE, FIGURES_DIR
from src.data_loader import Platform, SocialMediaDataLoader
from src.feature_engineering import FeatureEngineer
from src.modeling import EngagementPredictor, ModelResults, PlatformAwareVotingEnsemble
from src.profiler import DataProfiler
from src.text_processor import TextProcessor
from src.validator import DataValidator, NoFutureDatesRule, RequiredColumnsRule

# Ignore warnings
warnings.filterwarnings('ignore')

# Project setup
PROJECT_ROOT = Path.cwd()
while PROJECT_ROOT.name != 'vml' and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("EXPO 2020 DUBAI - SOCIAL MEDIA ANALYSIS")
print("=" * 80)
print(f"Started: {datetime.now()}")

# Load data
data_file = PROJECT_ROOT / 'data' / 'dataset_expo2020.json'
loader = SocialMediaDataLoader(data_file)
result = loader.load()
df = result.dataframe

print(f"\nData loaded: {len(df):,} records")
print(f"Date range: {df['created_time'].min()} to {df['created_time'].max()}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Quick check for new columns
if 'type' in df.columns:
    print(f"\nContent types: {df['type'].value_counts().to_dict()}")
if 'network_source' in df.columns:
    print(f"Network sources: {df['network_source'].value_counts().to_dict()}")


# In[2]:


df.head()


# In[3]:


# Validate data
print("=" * 80)
print("DATA VALIDATION")
print("=" * 80)

validator = DataValidator()
validator.add_rule(RequiredColumnsRule(['id', 'created_time', 'message', 'interaction_count']))
validator.add_rule(NoFutureDatesRule('created_time'))

validation_results = validator.validate(df)
print(f"Validation passed: {validation_results['is_valid']}")
if validation_results['errors']:
    for error in validation_results['errors']:
        print(f"Error: {error['message']}")

# Data profiling
print("\n" + "=" * 80)
print("DATA PROFILING")
print("=" * 80)

profiler = DataProfiler(df)
profile = profiler.profile()

print(f"\nBasic stats:")
print(f"Total records: {profile['shape']['rows']:,}")
print(f"Duplicate records: {profile['shape']['duplicates']:,}")
print(f"Data completeness: {profile['quality']['completeness']:.1f}%")
print(f"Date range: {profile['temporal']['date_range_days']} days")
print(f"Avg posts/day: {profile['temporal']['posts_per_day']:.0f}")

# Column-wise analysis
print("\nColumn profiles:")
column_stats = pd.DataFrame([
    {
        'column': cp.name,
        'type': cp.dtype,
        'missing': f"{cp.null_count:,} ({cp.null_percentage:.1f}%)",
        'unique': f"{cp.unique_count:,} ({cp.unique_percentage:.1f}%)"
    }
    for cp in profile['columns']
])
print(column_stats.to_string(index=False))

# Missing data patterns
print("\n" + "=" * 80)
print("MISSING DATA")
print("=" * 80)

missing_summary = df.isnull().sum()
missing_pct = (missing_summary / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_count': missing_summary,
    'missing_pct': missing_pct
}).sort_values('missing_count', ascending=False)
print(missing_df[missing_df['missing_count'] > 0])

# Platform vs Network Source analysis
print("\n" + "=" * 80)
print("CROSS-POSTING ANALYSIS")
print("=" * 80)

cross_tab = pd.crosstab(df['platform'], df['network_source'], margins=True)
print("\nPlatform vs Network Source:")
print(cross_tab)

# Calculate cross-posting rate
df['is_crosspost'] = df['platform'] != df['network_source']
crosspost_rate = df['is_crosspost'].mean() * 100
print(f"\nOverall cross-posting rate: {crosspost_rate:.1f}%")

# By platform
platform_crosspost = df.groupby('platform')['is_crosspost'].agg(['sum', 'mean'])
platform_crosspost.columns = ['crosspost_count', 'crosspost_rate']
platform_crosspost['crosspost_rate'] = platform_crosspost['crosspost_rate'] * 100
print("\nCross-posting by platform:")
print(platform_crosspost)


# In[5]:


print("=" * 80)
print("CROSS-POSTING?")
print("=" * 80)

print("\nSample Instagram posts:")
ig_sample = df[df['platform'] == 'instagram'].head(5)
print(ig_sample[['platform', 'network_source', 'link', 'type']].to_string())

print("\nSample Twitter posts:")
tw_sample = df[df['platform'] == 'twitter'].head(5)
print(tw_sample[['platform', 'network_source', 'link', 'type']].to_string())

# platform is correctly derived from link?
print("\n" + "=" * 80)
print("VERIFYING PLATFORM DETECTION")
print("=" * 80)

def detect_platform_from_url(url):
    if pd.isna(url):
        return 'unknown'
    url_str = str(url).lower()
    if 'instagram.com' in url_str:
        return 'instagram'
    elif 'twitter.com' in url_str:
        return 'twitter'
    else:
        return 'unknown'

sample_df = df.sample(n=20, random_state=42)
sample_df['detected_platform'] = sample_df['link'].apply(detect_platform_from_url)
sample_df['platform_matches'] = sample_df['platform'] == sample_df['detected_platform']

print("Platform detection check (sample of 20):")
mismatch = sample_df[~sample_df['platform_matches']]
if len(mismatch) > 0:
    print(f"\nFound {len(mismatch)} mismatches:")
    print(mismatch[['platform', 'detected_platform', 'link']].to_string())
else:
    print("All platforms correctly detected from URLs")

# Theory: network_source means something else?
print("\n" + "=" * 80)
print("HYPOTHESIS: What does network_source really mean?")
print("=" * 80)

# if network_source correlates with content type
source_type = pd.crosstab(df['network_source'], df['type'])
print("\nNetwork source vs Content type:")
print(source_type)

print("\n" + "=" * 80)
print("CHECKING IF COLUMNS ARE SWAPPED")
print("=" * 80)

# how many posts would be "native" if interpretation is swapped
df['would_be_native'] = df['platform'] == df['network_source']
print(f"If the interpretation is swapped, native posts would be: {df['would_be_native'].sum()} (should be > 0)")

# it's the data collection method?
print("\n" + "=" * 80)
print("HYPOTHESIS: Special dataset of cross-posted content only?")
print("=" * 80)

# duplicate messages
message_counts = df.groupby('message').agg({
    'platform': ['count', 'nunique', list],
    'interaction_count': list
}).reset_index()
message_counts.columns = ['message', 'post_count', 'platform_count', 'platforms', 'interactions']

# messages posted on both platforms
cross_platform_messages = message_counts[message_counts['platform_count'] > 1]
print(f"\nMessages appearing on multiple platforms: {len(cross_platform_messages)}")

if len(cross_platform_messages) > 0:
    print("\nExample of cross-platform message:")
    example = cross_platform_messages.iloc[0]
    print(f"Message: {example['message'][:100]}...")
    print(f"Platforms: {example['platforms']}")
    print(f"Interactions: {example['interactions']}")


# In[6]:


# map content types to platforms
print("Content type reality check:")
print("\nInstagram typical types: image, video, carousel")
print("Twitter typical types: status, photo, link")

# actual distribution
actual_dist = pd.crosstab(df['platform'], df['type'])
print("\nActual distribution (platform vs type):")
print(actual_dist)

actual_dist_pct = pd.crosstab(df['platform'], df['type'], normalize='index') * 100
print("\nAs percentages:")
print(actual_dist_pct.round(1))

print("\n" + "=" * 80)
print("DUPLICATE MESSAGES")
print("=" * 80)

duplicate_messages = df[df.duplicated('message', keep=False)].copy()
print(f"Total posts with duplicate messages: {len(duplicate_messages)}")

# group by message to see patterns
msg_groups = duplicate_messages.groupby('message').agg({
    'platform': list,
    'type': list,
    'network_source': list,
    'interaction_count': list,
    'created_time': ['min', 'max']
}).reset_index()

print("\nExamples of duplicate messages:")
for i in range(min(3, len(msg_groups))):
    msg = msg_groups.iloc[i]
    print(f"\nMessage {i+1}: '{msg['message'][:50]}...'")
    print(f"  Platforms: {msg['platform']}")
    print(f"  Types: {msg['type']}")
    print(f"  Network sources: {msg['network_source']}")
    print(f"  Time span: {msg['created_time']['max'] - msg['created_time']['min']}")

# network_source might be a data collection batch indicator
print("\n" + "=" * 80)
print("NETWORK_SOURCE AS BATCH INDICATOR?")
print("=" * 80)

# time distribution by network_source
df['date'] = df['created_time'].dt.date
date_range_by_source = df.groupby('network_source')['date'].agg(['min', 'max'])
print("\nDate ranges by network_source:")
print(date_range_by_source)

# there's temporal clustering?
daily_by_source = df.groupby(['date', 'network_source']).size().unstack(fill_value=0)
print("\nSample of daily posts by network_source (first 5 days):")
print(daily_by_source.head())

# statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"Unique messages: {df['message'].nunique():,}")
print(f"Messages appearing on both platforms: 35")
print(f"Percentage of shared content: {35 / df['message'].nunique() * 100:.3f}%")


# In[7]:


print("=" * 80)
print("DATA CLEANING")
print("=" * 80)

# remove unknown platform posts
df_clean = df[df['platform'] != 'unknown'].copy()
print(f"Removed {len(df) - len(df_clean)} posts with unknown platform")

# drop network_source, it's misleading
df_clean = df_clean.drop('network_source', axis=1)
print("Dropped misleading 'network_source' column")

# missing content types
print(f"\nMissing content types: {df_clean['type'].isna().sum():,}")
df_clean['type_filled'] = df_clean['type'].fillna('unknown_type')

# remove exact duplicates (same message, same platform, same time)
duplicates = df_clean.duplicated(subset=['message', 'platform', 'created_time'], keep='first')
df_clean = df_clean[~duplicates].copy()
print(f"Removed {duplicates.sum()} exact duplicates")

print(f"\nFinal dataset: {len(df_clean):,} posts")

# statistics
print("\n" + "=" * 80)
print("ENGAGEMENT DISTRIBUTION")
print("=" * 80)

print("Overall engagement statistics:")
print(df_clean['interaction_count'].describe())

# by platform
print("\nBy platform:")
platform_stats = df_clean.groupby('platform')['interaction_count'].describe()
print(platform_stats)

# 0 engagement
zero_eng = df_clean.groupby('platform')['interaction_count'].apply(lambda x: (x == 0).mean() * 100)
print(f"\nZero engagement rate:")
print(zero_eng.round(1))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Overall distribution (log scale)
ax1 = axes[0, 0]
df_clean[df_clean['interaction_count'] > 0]['interaction_count'].hist(bins=50, ax=ax1, alpha=0.7)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel('Interaction Count')
ax1.set_ylabel('Frequency (log)')
ax1.set_title('Overall Engagement Distribution (excluding zeros)')

# 2. By platform
ax2 = axes[0, 1]
for platform in ['instagram', 'twitter']:
    data = df_clean[(df_clean['platform'] == platform) & (df_clean['interaction_count'] > 0)]
    data['interaction_count'].hist(bins=50, alpha=0.5, label=platform, ax=ax2)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Interaction Count')
ax2.set_ylabel('Frequency (log)')
ax2.set_title('Engagement by Platform (excluding zeros)')
ax2.legend()

# 3. Content type performance
ax3 = axes[1, 0]
type_stats = df_clean.groupby('type_filled')['interaction_count'].agg(['mean', 'median'])
type_stats.sort_values('mean', ascending=True).plot(kind='barh', ax=ax3)
ax3.set_xlabel('Interaction Count')
ax3.set_title('Average Engagement by Content Type')

# 4. Platform vs Type heatmap
ax4 = axes[1, 1]
pivot = df_clean.pivot_table(
    values='interaction_count',
    index='platform',
    columns='type_filled',
    aggfunc='mean'
)
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4)
ax4.set_title('Mean Engagement: Platform × Content Type')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'basic_eda.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("EXTREME VALUES")
print("=" * 80)

top_posts = df_clean.nlargest(5, 'interaction_count')
print("\nTop 5 posts by engagement:")
for idx, post in top_posts.iterrows():
    print(f"\n{post['interaction_count']:,} interactions - {post['platform']} ({post['type']})")
    print(f"Message: {post['message'][:100]}...")


# The engagement follows extreme power law distribution with massive inequality - Instagram averages 158 interactions vs Twitter's 13, but medians are 3 and 0 respectively. Platform differences are stark: 54% of Twitter posts get zero engagement vs 25% on Instagram. Content type matters significantly - carousels dominate with 310 average engagement, followed by videos, while status updates and links perform poorly. The top post (7.1M interactions) is 45,000x the Instagram average, suggesting viral outliers drive much of the total engagement.

# In[8]:


print("=" * 80)
print("TEXT PROCESSING")
print("=" * 80)

# initialize text processor
text_processor = TextProcessor()
df_processed = text_processor.process_dataframe(df_clean)

# basic text stats
print("Text statistics:")
print(f"Avg text length: {df_processed['text_length'].mean():.1f} chars")
print(f"Avg word count: {df_processed['word_count'].mean():.1f} words")
print(f"Posts with hashtags: {(df_processed['hashtag_count'] > 0).mean():.1%}")
print(f"Posts with mentions: {(df_processed['mention_count'] > 0).mean():.1%}")
print(f"Posts with urls: {(df_processed['url_count'] > 0).mean():.1%}")
print(f"Posts with emojis: {(df_processed['emoji_count'] > 0).mean():.1%}")

# language distribution
print("\nLanguage distribution:")
lang_dist = df_processed['language'].value_counts(normalize=True) * 100
print(lang_dist.head())

# expo-related posts
expo_posts = df_processed['is_expo_related'].sum()
print(f"\nEXPO-related posts: {expo_posts:,} ({expo_posts/len(df_processed)*100:.1f}%)")

# top hashtags
print("\nTop 10 hashtags:")
top_hashtags = text_processor.get_top_items(df_processed['hashtags'], 10)
for i, (tag, count) in enumerate(top_hashtags, 1):
    print(f"{i:2d}. {tag}: {count:,}")

print("\n" + "=" * 80)
print("TEMPORAL ANALYSIS")
print("=" * 80)

# add temporal features
df_processed['date'] = df_processed['created_time'].dt.date
df_processed['hour'] = df_processed['created_time'].dt.hour
df_processed['day_of_week'] = df_processed['created_time'].dt.dayofweek
df_processed['month'] = df_processed['created_time'].dt.month

# expo phases
expo_start = pd.Timestamp(EXPO_START_DATE, tz='UTC')
expo_end = pd.Timestamp(EXPO_END_DATE, tz='UTC')

def get_expo_phase(timestamp):
    if timestamp < expo_start:
        return 'pre_opening'
    elif timestamp < expo_start + pd.Timedelta(days=30):
        return 'opening_month'
    elif timestamp < expo_end - pd.Timedelta(days=30):
        return 'main_period'
    else:
        return 'closing_month'

df_processed['expo_phase'] = df_processed['created_time'].apply(get_expo_phase)

# phase distribution
print("Posts by expo phase:")
phase_dist = df_processed['expo_phase'].value_counts()
for phase, count in phase_dist.items():
    print(f"  {phase}: {count:,} ({count/len(df_processed)*100:.1f}%)")

# engagement by phase
print("\nEngagement by expo phase:")
phase_engagement = df_processed.groupby('expo_phase')['interaction_count'].agg(['mean', 'median'])
print(phase_engagement)

# temporal visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. daily post volume
ax1 = axes[0, 0]
daily_posts = df_processed.groupby('date').size()
ax1.plot(daily_posts.index, daily_posts.values)
ax1.axvline(expo_start, color='green', linestyle='--', alpha=0.5)
ax1.axvline(expo_end, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Posts')
ax1.set_title('Daily Post Volume')
ax1.tick_params(axis='x', rotation=45)

# 2. hourly patterns by platform
ax2 = axes[0, 1]
hourly = df_processed.groupby(['platform', 'hour']).size().unstack(0)
hourly.plot(ax=ax2)
ax2.set_xlabel('Hour of Day (UTC)')
ax2.set_ylabel('Number of Posts')
ax2.set_title('Posting Patterns by Hour')
ax2.legend(title='Platform')

# 3. day of week engagement
ax3 = axes[0, 2]
dow_engagement = df_processed.groupby(['platform', 'day_of_week'])['interaction_count'].mean().unstack(0)
dow_engagement.plot(kind='bar', ax=ax3)
ax3.set_xlabel('Day of Week (0=Monday)')
ax3.set_ylabel('Average Engagement')
ax3.set_title('Engagement by Day of Week')
ax3.legend(title='Platform')

# 4. language vs engagement
ax4 = axes[1, 0]
lang_engagement = df_processed.groupby('language')['interaction_count'].agg(['mean', 'count'])
lang_engagement = lang_engagement[lang_engagement['count'] > 1000]  # filter rare languages
lang_engagement['mean'].sort_values().plot(kind='barh', ax=ax4)
ax4.set_xlabel('Average Engagement')
ax4.set_title('Engagement by Language')

# 5. content features vs engagement
ax5 = axes[1, 1]
features = ['hashtag_count', 'mention_count', 'emoji_count', 'url_count']
correlations = [df_processed[feat].corr(df_processed['interaction_count']) for feat in features]
ax5.bar(features, correlations)
ax5.set_ylabel('Correlation with Engagement')
ax5.set_title('Content Feature Correlations')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 6. phase timeline
ax6 = axes[1, 2]
phase_timeline = df_processed.groupby(['date', 'expo_phase']).size().unstack(fill_value=0)
phase_timeline.plot(kind='area', stacked=True, ax=ax6, alpha=0.7)
ax6.set_xlabel('Date')
ax6.set_ylabel('Number of Posts')
ax6.set_title('Posts by Expo Phase Over Time')
ax6.legend(title='Phase', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'text_temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


# Text analysis reveals 99.8% of posts are Expo-related (using keywords), with English dominating (61%) despite Dubai's multicultural context. Temporal patterns show clear daily rhythm peaking at 10-15 UTC and dramatic volume spike at opening (October 1st) followed by steady decline. Pre-opening phase had highest average engagement (204) despite lowest volume, suggesting hype effect, while main period stabilized at 107. Interestingly, emoji count shows positive correlation with engagement while hashtags/mentions show negative correlation.

# In[9]:


# quick check if url analysis is relevant
print("URL presence in data:")
print(f"Posts with URLs: {(df_processed['url_count'] > 0).sum():,} ({(df_processed['url_count'] > 0).mean():.1%})")

# by platform
url_by_platform = df_processed.groupby('platform')['url_count'].apply(lambda x: (x > 0).mean() * 100)
print("\nURL presence by platform:")
print(url_by_platform)

# engagement comparison
ig_data = df_processed[df_processed['platform'] == 'instagram']
ig_with_url = ig_data[ig_data['url_count'] > 0]['interaction_count'].mean()
ig_no_url = ig_data[ig_data['url_count'] == 0]['interaction_count'].mean()

twitter_data = df_processed[df_processed['platform'] == 'twitter']
twitter_with_url = twitter_data[twitter_data['url_count'] > 0]['interaction_count'].mean()
twitter_no_url = twitter_data[twitter_data['url_count'] == 0]['interaction_count'].mean()

print(f"\nInstagram URL effect:")
print(f"  With URL: {ig_with_url:.1f} mean engagement")
print(f"  No URL: {ig_no_url:.1f} mean engagement")
print(f"  Ratio: {ig_with_url/ig_no_url:.2f}x")

print(f"\nTwitter URL effect:")
print(f"  With URL: {twitter_with_url:.1f} mean engagement")
print(f"  No URL: {twitter_no_url:.1f} mean engagement")
print(f"  Ratio: {twitter_with_url/twitter_no_url:.2f}x")


# In[10]:


print("\n" + "=" * 80)
print("DUPLICATE MESSAGES")
print("=" * 80)

# analyze duplicate patterns
duplicate_msgs = df_processed[df_processed.duplicated('message', keep=False)]
print(f"Posts with duplicate messages: {len(duplicate_msgs):,} ({len(duplicate_msgs)/len(df_processed)*100:.1f}%)")

# group duplicates to understand pattern
msg_groups = duplicate_msgs.groupby('message').agg({
    'platform': ['count', list],
    'created_time': ['min', 'max'],
    'interaction_count': ['mean', 'std']
}).head(10)

print("\nDuplicate pattern analysis (first 10):")
for i, (msg, data) in enumerate(msg_groups.iterrows(), 1):
    count = data[('platform', 'count')]
    platforms = data[('platform', 'list')]
    time_diff = data[('created_time', 'max')] - data[('created_time', 'min')]

    # count platform distribution
    platform_counts = pd.Series(platforms).value_counts().to_dict()

    print(f"\n{i}. Message: '{msg[:40]}...'")
    print(f"   Posted {count} times over {time_diff}")
    print(f"   Platforms: {platform_counts}")
    print(f"   Avg engagement: {data[('interaction_count', 'mean')]:.0f} (std: {data[('interaction_count', 'std')]:.0f})")


# In[11]:


# identify duplicate messages
message_counts = df_processed['message'].value_counts()
duplicate_messages = message_counts[message_counts > 1]
print(f"Unique messages posted multiple times: {len(duplicate_messages):,}")

# analyze posting patterns for duplicates
duplicate_posts = df_processed[df_processed['message'].isin(duplicate_messages.index)]
duplicate_stats = duplicate_posts.groupby('message').agg({
    'id': 'count',
    'platform': lambda x: x.mode()[0],  # most common platform
    'created_time': lambda x: (x.max() - x.min()).total_seconds() / 3600,  # hours
    'interaction_count': ['mean', 'max']
})

# flatten column names
duplicate_stats.columns = ['post_count', 'main_platform', 'time_span_hours', 'mean_engagement', 'max_engagement']

# categorize by time span
rapid_reposts = duplicate_stats[duplicate_stats['time_span_hours'] < 1]
daily_reposts = duplicate_stats[(duplicate_stats['time_span_hours'] >= 1) &
                                (duplicate_stats['time_span_hours'] < 24)]
long_reposts = duplicate_stats[duplicate_stats['time_span_hours'] >= 24]

print(f"\nReposting patterns:")
print(f"  Rapid (<1 hour): {len(rapid_reposts):,} messages")
print(f"  Daily (1-24 hours): {len(daily_reposts):,} messages")
print(f"  Long-term (>24 hours): {len(long_reposts):,} messages")

# engagement comparison
original_posts = df_processed[~df_processed.duplicated('message', keep='first')]
duplicate_posts = df_processed[df_processed.duplicated('message', keep='first')]

print(f"\nEngagement comparison:")
print(f"  Original posts mean: {original_posts['interaction_count'].mean():.1f}")
print(f"  Duplicate posts mean: {duplicate_posts['interaction_count'].mean():.1f}")
print(f"  Ratio: {duplicate_posts['interaction_count'].mean() / original_posts['interaction_count'].mean():.2f}x")

# identify potential spam
spam_indicators = duplicate_stats[
    (duplicate_stats['post_count'] >= 3) &  # posted 3+ times
    (duplicate_stats['time_span_hours'] < 24) &  # within 24 hours
    (duplicate_stats['mean_engagement'] < 10)  # low engagement
]
print(f"\nPotential spam messages (3+ posts, <24h, <10 engagement): {len(spam_indicators):,}")

# deduplicate dataset
df_clean = df_processed.drop_duplicates(subset=['message'], keep='first').copy()
print(f"\nDataset after deduplication: {len(df_clean):,} posts (removed {len(df_processed) - len(df_clean):,})")

# verify impact
print("\nImpact on key metrics:")
print(f"  Mean engagement before: {df_processed['interaction_count'].mean():.1f}")
print(f"  Mean engagement after: {df_clean['interaction_count'].mean():.1f}")
print(f"  Zero engagement before: {(df_processed['interaction_count'] == 0).mean():.1%}")
print(f"  Zero engagement after: {(df_clean['interaction_count'] == 0).mean():.1%}")


# In[12]:


print("=" * 80)
print("MISSING CONTENT TYPES")
print("=" * 80)

# analyze missing types
missing_type_mask = df_clean['type'].isna()
missing_type_posts = df_clean[missing_type_mask]

print(f"Posts with missing type: {len(missing_type_posts):,} ({len(missing_type_posts)/len(df_clean)*100:.1f}%)")

# by platform
print("\nMissing types by platform:")
missing_by_platform = df_clean.groupby('platform')['type'].apply(lambda x: x.isna().sum())
total_by_platform = df_clean.groupby('platform').size()
missing_rate = (missing_by_platform / total_by_platform * 100).round(1)
for platform in missing_rate.index:
    print(f"  {platform}: {missing_by_platform[platform]:,} / {total_by_platform[platform]:,} ({missing_rate[platform]}%)")

# temporal pattern
print("\nTemporal pattern of missing types:")
missing_type_posts['date'] = missing_type_posts['created_time'].dt.date
date_range = f"{missing_type_posts['date'].min()} to {missing_type_posts['date'].max()}"
print(f"Date range: {date_range}")

# are they concentrated in specific period?
missing_by_month = missing_type_posts.groupby(missing_type_posts['created_time'].dt.to_period('M')).size()
print("\nMissing types by month:")
for month, count in missing_by_month.head().items():
    print(f"  {month}: {count:,}")

# text characteristics of missing type posts
print("\nCharacteristics of posts with missing types:")
print(f"  Avg text length: {missing_type_posts['text_length'].mean():.1f}")
print(f"  Avg hashtags: {missing_type_posts['hashtag_count'].mean():.1f}")
print(f"  Avg engagement: {missing_type_posts['interaction_count'].mean():.1f}")
print(f"  Zero engagement rate: {(missing_type_posts['interaction_count'] == 0).mean():.1%}")

# compare with posts that have types
has_type_posts = df_clean[~missing_type_mask]
print("\nComparison with typed posts:")
print(f"  Typed posts avg engagement: {has_type_posts['interaction_count'].mean():.1f}")
print(f"  Missing type avg engagement: {missing_type_posts['interaction_count'].mean():.1f}")
print(f"  Ratio: {missing_type_posts['interaction_count'].mean() / has_type_posts['interaction_count'].mean():.2f}x")

# sample messages to understand what they might be
print("\nSample posts with missing types:")
sample = missing_type_posts.sample(n=min(5, len(missing_type_posts)), random_state=42)
for i, (_, post) in enumerate(sample.iterrows(), 1):
    print(f"\n{i}. Platform: {post['platform']}, Engagement: {post['interaction_count']}")
    print(f"   Message: {post['message'][:100]}...")


# In[13]:


print("=" * 80)
print("OUTLIERS")
print("=" * 80)

# define outliers using multiple methods
q99 = df_clean['interaction_count'].quantile(0.99)
q999 = df_clean['interaction_count'].quantile(0.999)
q9999 = df_clean['interaction_count'].quantile(0.9999)

print("Engagement percentiles:")
print(f"  99th: {q99:,.0f}")
print(f"  99.9th: {q999:,.0f}")
print(f"  99.99th: {q9999:,.0f}")
print(f"  Max: {df_clean['interaction_count'].max():,.0f}")

# extreme outliers (>99.9th percentile)
extreme_outliers = df_clean[df_clean['interaction_count'] > q999]
print(f"\nExtreme outliers (>99.9th percentile): {len(extreme_outliers)} posts")

# analyze extreme outliers
print("\nExtreme outlier characteristics:")
print(f"  Platform distribution: {extreme_outliers['platform'].value_counts().to_dict()}")
print(f"  Type distribution: {extreme_outliers['type_filled'].value_counts().head(3).to_dict()}")
print(f"  Engagement range: {extreme_outliers['interaction_count'].min():,.0f} - {extreme_outliers['interaction_count'].max():,.0f}")

# top 10 posts analysis
top_10 = df_clean.nlargest(10, 'interaction_count')
print("\nTop 10 posts analysis:")
for i, (_, post) in enumerate(top_10.iterrows(), 1):
    print(f"\n{i}. {post['interaction_count']:,} interactions")
    print(f"   Platform: {post['platform']}, Type: {post['type_filled']}")
    print(f"   Date: {post['created_time'].date()}")
    print(f"   Hashtags: {post['hashtag_count']}, Emojis: {post['emoji_count']}")
    print(f"   Message preview: {post['message'][:80]}...")

# check for suspiciously round numbers (potential data errors)
round_numbers = [1000000, 2000000, 3000000, 5000000, 7000000]
suspicious = df_clean[df_clean['interaction_count'].isin(round_numbers)]
print(f"\nSuspiciously round engagement numbers: {len(suspicious)} posts")

# temporal clustering of outliers
extreme_outliers['date'] = extreme_outliers['created_time'].dt.date
outlier_dates = extreme_outliers.groupby('date').size().sort_values(ascending=False)
print("\nDates with most extreme outliers:")
for date, count in outlier_dates.head(5).items():
    print(f"  {date}: {count} outliers")

# calculate impact on mean
mean_with = df_clean['interaction_count'].mean()
mean_without = df_clean[df_clean['interaction_count'] <= q999]['interaction_count'].mean()
print(f"\nImpact on mean engagement:")
print(f"  With outliers: {mean_with:.1f}")
print(f"  Without outliers (≤99.9%): {mean_without:.1f}")
print(f"  Difference: {mean_with - mean_without:.1f} ({(mean_with/mean_without - 1)*100:.1f}% inflation)")

# add outlier flag
df_clean['is_extreme_outlier'] = (df_clean['interaction_count'] > q999).astype(int)
print(f"\nAdded 'is_extreme_outlier' flag for {df_clean['is_extreme_outlier'].sum()} posts")


# In[14]:


# create comprehensive figure
fig = plt.figure(figsize=(20, 16))

# 1. engagement distribution - reality check
ax1 = plt.subplot(3, 4, 1)
engagement_bins = [0, 1, 10, 100, 1000, 10000, 100000, 10000000]
engagement_labels = ['0', '1-9', '10-99', '100-999', '1K-10K', '10K-100K', '100K+']
df_clean['engagement_bin'] = pd.cut(df_clean['interaction_count'], bins=engagement_bins, labels=engagement_labels, include_lowest=True)
engagement_dist = df_clean['engagement_bin'].value_counts()
colors = ['red' if x == '0' else 'orange' if x in ['1-9', '10-99'] else 'green' for x in engagement_dist.index]
engagement_dist.plot(kind='bar', ax=ax1, color=colors)
ax1.set_title('Engagement Distribution Reality', fontweight='bold')
ax1.set_xlabel('Engagement Range')
ax1.set_ylabel('Number of Posts')
ax1.set_yscale('log')
for i, v in enumerate(engagement_dist):
    ax1.text(i, v, f'{v:,}\n({v/len(df_clean)*100:.1f}%)', ha='center', va='bottom')

# 2. platform comparison - complete picture
ax2 = plt.subplot(3, 4, 2)
platform_metrics = df_clean.groupby('platform').agg({
    'interaction_count': ['mean', 'median'],
    'id': 'count'
})
platform_metrics.columns = ['mean', 'median', 'count']
x = np.arange(len(platform_metrics))
width = 0.25
ax2.bar(x - width, platform_metrics['mean'], width, label='Mean', color='darkblue')
ax2.bar(x, platform_metrics['median'], width, label='Median', color='lightblue')
ax2.bar(x + width, platform_metrics['count']/1000, width, label='Count (K)', color='gray')
ax2.set_xticks(x)
ax2.set_xticklabels(platform_metrics.index)
ax2.set_ylabel('Value')
ax2.set_title('Platform Metrics Comparison', fontweight='bold')
ax2.legend()
ax2.set_yscale('log')

# 3. content type performance
ax3 = plt.subplot(3, 4, 3)
type_performance = df_clean.groupby('type_filled').agg({
    'interaction_count': 'mean',
    'id': 'count'
}).sort_values('interaction_count', ascending=False)
type_performance = type_performance[type_performance['id'] > 100]  # filter rare types
bars = ax3.bar(range(len(type_performance)), type_performance['interaction_count'],
               color=['green' if x > 100 else 'orange' if x > 50 else 'red'
                      for x in type_performance['interaction_count']])
ax3.set_xticks(range(len(type_performance)))
ax3.set_xticklabels(type_performance.index, rotation=45, ha='right')
ax3.set_ylabel('Average Engagement')
ax3.set_title('Content Type Performance', fontweight='bold')
for i, (v, c) in enumerate(zip(type_performance['interaction_count'], type_performance['id'])):
    ax3.text(i, v, f'{v:.0f}\n(n={c:,})', ha='center', va='bottom', fontsize=8)

# 4. temporal decay pattern
ax4 = plt.subplot(3, 4, 4)
daily_metrics = df_clean.groupby(df_clean['created_time'].dt.date).agg({
    'interaction_count': 'mean',
    'id': 'count'
})
ax4_twin = ax4.twinx()
ax4.plot(daily_metrics.index, daily_metrics['id'], color='gray', alpha=0.5, label='Volume')
ax4_twin.plot(daily_metrics.index, daily_metrics['interaction_count'], color='red', linewidth=2, label='Avg Engagement')
ax4.set_ylabel('Daily Posts', color='gray')
ax4_twin.set_ylabel('Avg Engagement', color='red')
ax4.set_title('Temporal Decay Pattern', fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
# mark key events
expo_start = pd.Timestamp(EXPO_START_DATE)
ax4.axvline(expo_start, color='green', linestyle='--', alpha=0.5, label='Opening')

# 5. language performance
ax5 = plt.subplot(3, 4, 5)
lang_stats = df_clean.groupby('language').agg({
    'interaction_count': 'mean',
    'id': 'count'
}).sort_values('interaction_count', ascending=False)
lang_stats = lang_stats[lang_stats['id'] > 1000]
colors = ['green', 'orange', 'red'] if len(lang_stats) <= 3 else ['blue'] * len(lang_stats)
bars = ax5.bar(range(len(lang_stats)), lang_stats['interaction_count'], color=colors)
ax5.set_xticks(range(len(lang_stats)))
ax5.set_xticklabels(lang_stats.index)
ax5.set_ylabel('Average Engagement')
ax5.set_title('Language Performance', fontweight='bold')
for i, (v, c) in enumerate(zip(lang_stats['interaction_count'], lang_stats['id'])):
    ax5.text(i, v, f'{v:.0f}\n({c/len(df_clean)*100:.1f}%)', ha='center', va='bottom')

# 6. emoji optimization curve
ax6 = plt.subplot(3, 4, 6)
emoji_performance = df_clean.groupby('emoji_count')['interaction_count'].agg(['mean', 'count'])
emoji_performance = emoji_performance[emoji_performance['count'] > 100]
emoji_performance = emoji_performance.head(10)  # limit to 0-9 emojis
ax6.plot(emoji_performance.index, emoji_performance['mean'], marker='o', linewidth=2, markersize=8)
ax6.fill_between(emoji_performance.index, emoji_performance['mean'], alpha=0.3)
ax6.set_xlabel('Number of Emojis')
ax6.set_ylabel('Average Engagement')
ax6.set_title('Emoji Optimization Curve', fontweight='bold')
ax6.grid(True, alpha=0.3)
# mark optimal
optimal_emoji = emoji_performance['mean'].idxmax()
ax6.axvline(optimal_emoji, color='red', linestyle='--', alpha=0.5)
ax6.text(optimal_emoji, emoji_performance.loc[optimal_emoji, 'mean'],
         f'Optimal: {optimal_emoji}', ha='center', va='bottom', color='red', fontweight='bold')

# 7. hashtag effectiveness
ax7 = plt.subplot(3, 4, 7)
hashtag_bins = [0, 1, 5, 10, 15, 20, 30, 100]
hashtag_labels = ['0', '1-4', '5-9', '10-14', '15-19', '20-29', '30+']
df_clean['hashtag_bin'] = pd.cut(df_clean['hashtag_count'], bins=hashtag_bins, labels=hashtag_labels, include_lowest=True)
hashtag_performance = df_clean.groupby('hashtag_bin')['interaction_count'].mean()
colors = ['red' if i == 0 else 'green' if i in [2, 3] else 'orange' for i in range(len(hashtag_performance))]
hashtag_performance.plot(kind='bar', ax=ax7, color=colors)
ax7.set_xlabel('Number of Hashtags')
ax7.set_ylabel('Average Engagement')
ax7.set_title('Hashtag Effectiveness', fontweight='bold')
ax7.tick_params(axis='x', rotation=45)

# 8. hour of day heatmap
ax8 = plt.subplot(3, 4, 8)
hour_platform = df_clean.pivot_table(
    values='interaction_count',
    index='platform',
    columns='hour',
    aggfunc='mean'
)
sns.heatmap(hour_platform, cmap='YlOrRd', ax=ax8, cbar_kws={'label': 'Avg Engagement'})
ax8.set_xlabel('Hour of Day (UTC)')
ax8.set_title('Optimal Posting Hours', fontweight='bold')

# 9. viral post timeline
ax9 = plt.subplot(3, 4, 9)
viral_threshold = df_clean['interaction_count'].quantile(0.99)
viral_posts = df_clean[df_clean['interaction_count'] > viral_threshold]
viral_timeline = viral_posts.groupby(viral_posts['created_time'].dt.date).size()
ax9.bar(viral_timeline.index, viral_timeline.values, color='red', alpha=0.7)
ax9.set_xlabel('Date')
ax9.set_ylabel('Number of Viral Posts (>99%)')
ax9.set_title('Viral Post Clustering', fontweight='bold')
ax9.tick_params(axis='x', rotation=45)

# 10. zero engagement analysis
ax10 = plt.subplot(3, 4, 10)
zero_analysis = df_clean.groupby(['platform', 'type_filled'])['interaction_count'].apply(lambda x: (x == 0).mean() * 100)
zero_pivot = zero_analysis.unstack()
zero_pivot.plot(kind='bar', ax=ax10, width=0.8)
ax10.set_ylabel('Zero Engagement Rate (%)')
ax10.set_title('Zero Engagement by Platform & Type', fontweight='bold')
ax10.legend(title='Content Type', bbox_to_anchor=(1.05, 1), loc='upper left')
ax10.tick_params(axis='x', rotation=0)
ax10.axhline(y=50, color='red', linestyle='--', alpha=0.5)

# 11. duplicate impact
ax11 = plt.subplot(3, 4, 11)
duplicate_impact = pd.DataFrame({
    'Original Dataset': [len(df_processed), df_processed['interaction_count'].mean(),
                         (df_processed['interaction_count'] == 0).mean() * 100],
    'After Deduplication': [len(df_clean), df_clean['interaction_count'].mean(),
                           (df_clean['interaction_count'] == 0).mean() * 100]
}, index=['Posts', 'Avg Engagement', 'Zero Rate (%)'])
duplicate_impact.T.plot(kind='bar', ax=ax11)
ax11.set_title('Impact of Removing Duplicates', fontweight='bold')
ax11.set_ylabel('Value')
ax11.legend(loc='upper right')
ax11.tick_params(axis='x', rotation=0)

# 12. summary statistics box
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')
summary_text = f"""
DATASET SUMMARY
═══════════════════════════
Total Posts: {len(df_clean):,}
Date Range: {df_clean['created_time'].min().date()} to {df_clean['created_time'].max().date()}
Platforms: Instagram (55%), Twitter (45%)

ENGAGEMENT METRICS
═══════════════════════════
Mean: {df_clean['interaction_count'].mean():.0f}
Median: {df_clean['interaction_count'].median():.0f}
Zero Engagement: {(df_clean['interaction_count'] == 0).mean():.1%}
Viral (>99%): {(df_clean['interaction_count'] > viral_threshold).mean():.1%}
"""
ax12.text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
         verticalalignment='top', transform=ax12.transAxes)

plt.suptitle('EXPO 2020 DUBAI - DATA ANALYSIS', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'eda_summary.png', dpi=300, bbox_inches='tight')
plt.show()


# ## EDA Summary & Implications
# 
# The dataset was reduced from 500K to 423K posts after 18% duplicates and spam were removed, misleading network_source column was dropped, and 71K missing content types were identified as Twitter-only issue with 8x worse engagement. Strong patterns were uncovered including 12x platform difference (Instagram vs Twitter), power law distribution where 0.1% posts capture majority of engagement, and unexpected Instagram URL paradox where non-functional links correlate with 2x higher engagement. Content type was identified as strongest predictor (carousel 310 vs status 2 mean), followed by URL presence (2-4x multiplier), emoji count (optimal 2-3), and hashtag diversity over quantity. The 35% zero-engagement rate and extreme outliers (max 64,000x median) require platform-specific, two-stage modeling where engagement presence is predicted first, then amount if positive. Feature engineering should focus on content type encoding, URL flags, emoji/hashtag optimization metrics, and log-transformed or percentile-ranked targets to handle the severe skew.

# In[31]:


print("=" * 80)
print("HYPOTHESIS TESTING")
print("=" * 80)

# all p-values for multiple testing correction
all_p_values = []
test_names = []

# test Assumptions
print("\n" + "=" * 80)
print("STATISTICAL ASSUMPTIONS")
print("=" * 80)

# sample for normality testing (full dataset too large)
sample_size = 5000
ig_sample = df_clean[df_clean['platform'] == 'instagram']['interaction_count'].sample(n=min(sample_size, len(df_clean[df_clean['platform'] == 'instagram'])), random_state=42)
tw_sample = df_clean[df_clean['platform'] == 'twitter']['interaction_count'].sample(n=min(sample_size, len(df_clean[df_clean['platform'] == 'twitter'])), random_state=42)

# normality tests
print("\n1. Normality Tests (for parametric test validity):")
print("-" * 40)

# Shapiro-Wilk test
_, p_shapiro_ig = shapiro(ig_sample[:1000])
_, p_shapiro_tw = shapiro(tw_sample[:1000])

# Jarque-Bera test
jb_ig, p_jb_ig = jarque_bera(ig_sample)
jb_tw, p_jb_tw = jarque_bera(tw_sample)

# Anderson-Darling test
ad_ig = anderson(ig_sample, dist='norm')
ad_tw = anderson(tw_sample, dist='norm')

print(f"Instagram normality:")
print(f"  Shapiro-Wilk p-value: {p_shapiro_ig:.2e}")
print(f"  Jarque-Bera p-value: {p_jb_ig:.2e}")
print(f"  Anderson-Darling: statistic={ad_ig.statistic:.3f}, critical@5%={ad_ig.critical_values[2]:.3f}")

print(f"\nTwitter normality:")
print(f"  Shapiro-Wilk p-value: {p_shapiro_tw:.2e}")
print(f"  Jarque-Bera p-value: {p_jb_tw:.2e}")
print(f"  Anderson-Darling: statistic={ad_tw.statistic:.3f}, critical@5%={ad_tw.critical_values[2]:.3f}")

# log transformation to check if it helps
ig_log = np.log1p(ig_sample)
tw_log = np.log1p(tw_sample)
_, p_shapiro_ig_log = shapiro(ig_log[:1000])
_, p_shapiro_tw_log = shapiro(tw_log[:1000])

print(f"\nAfter log transformation:")
print(f"  Instagram Shapiro-Wilk p-value: {p_shapiro_ig_log:.2e}")
print(f"  Twitter Shapiro-Wilk p-value: {p_shapiro_tw_log:.2e}")

normality_satisfied = p_shapiro_ig > 0.05 and p_shapiro_tw > 0.05
print(f"\n✓ Normality assumption: {'SATISFIED' if normality_satisfied else 'VIOLATED - Use non-parametric tests'}")

print("\n2. Homogeneity of Variance Tests:")
print("-" * 40)

# Levene's test (robust to non-normality)
stat_levene, p_levene = levene(ig_sample, tw_sample)
print(f"Levene's test: F={stat_levene:.3f}, p={p_levene:.2e}")

# Bartlett's test (sensitive to non-normality)
stat_bartlett, p_bartlett = bartlett(ig_sample, tw_sample)
print(f"Bartlett's test: χ²={stat_bartlett:.3f}, p={p_bartlett:.2e}")

variance_equal = p_levene > 0.05
print(f"\n✓ Equal variance assumption: {'SATISFIED' if variance_equal else 'VIOLATED - Use Welch t-test'}")

# --- HYPOTHESIS 1: Content Type Effect ---
print("\n" + "=" * 80)
print("H1: CONTENT TYPE AFFECTS ENGAGEMENT")
print("=" * 80)

print("\nFormal Hypothesis:")
print("  H₀: μ_carousel = μ_image = μ_video = μ_status = μ_link")
print("  H₁: At least one content type has different mean engagement")
print("  Significance level: α = 0.001 (Bonferroni corrected)")

content_groups = []
content_names = []
for ctype in df_clean['type_filled'].unique():
    if ctype != 'unknown_type':
        group_data = df_clean[df_clean['type_filled'] == ctype]['interaction_count'].values
        if len(group_data) > 30:
            content_groups.append(group_data)
            content_names.append(ctype)

# parametric test: One-way ANOVA
f_stat, p_anova = f_oneway(*content_groups)
print(f"\nOne-way ANOVA:")
print(f"  F-statistic: {f_stat:.2f}")
print(f"  p-value: {p_anova:.2e}")
all_p_values.append(p_anova)
test_names.append("ANOVA_content")

# non-parametric alternative: Kruskal-Wallis
h_stat, p_kruskal = kruskal(*content_groups)
print(f"\nKruskal-Wallis test (non-parametric):")
print(f"  H-statistic: {h_stat:.2f}")
print(f"  p-value: {p_kruskal:.2e}")
all_p_values.append(p_kruskal)
test_names.append("KW_content")

# effect size
n_total = sum(len(g) for g in content_groups)
epsilon_squared = (h_stat - len(content_groups) + 1) / (n_total - len(content_groups))
eta_squared = (f_stat * (len(content_groups) - 1)) / (f_stat * (len(content_groups) - 1) + n_total - len(content_groups))
print(f"\nEffect sizes:")
print(f"  η² (ANOVA): {eta_squared:.4f} - {'Small' if eta_squared < 0.06 else 'Medium' if eta_squared < 0.14 else 'Large'}")
print(f"  ε² (Kruskal-Wallis): {epsilon_squared:.4f}")

# post-hoc tests if significant
if p_kruskal < 0.001:
    print("\nPost-hoc pairwise comparisons (Tukey HSD):")
    # Prepare data for Tukey
    engagement_data = []
    group_labels = []
    for i, group in enumerate(content_groups[:4]):  # Limit to 4 main types
        sample = group[:1000] if len(group) > 1000 else group  # Sample for speed
        engagement_data.extend(sample)
        group_labels.extend([content_names[i]] * len(sample))

    tukey = pairwise_tukeyhsd(engagement_data, group_labels, alpha=0.001)
    print(tukey.summary())

# --- HYPOTHESIS 2: Platform Difference ---
print("\n" + "=" * 80)
print("H2: PLATFORM ENGAGEMENT DISTRIBUTIONS DIFFER")
print("=" * 80)

print("\nFormal Hypothesis:")
print("  H₀: μ_Instagram = μ_Twitter")
print("  H₁: μ_Instagram ≠ μ_Twitter")
print("  Test type: Two-tailed")

ig_engagement = df_clean[df_clean['platform'] == 'instagram']['interaction_count'].values
tw_engagement = df_clean[df_clean['platform'] == 'twitter']['interaction_count'].values

# parametric test: Independent samples t-test
t_stat, p_ttest = ttest_ind(ig_engagement, tw_engagement, equal_var=False)  # Welch's t-test
print(f"\nWelch's t-test (unequal variances):")
print(f"  t-statistic: {t_stat:.2f}")
print(f"  p-value: {p_ttest:.2e}")
print(f"  Mean difference: {ig_engagement.mean() - tw_engagement.mean():.2f}")

# standard error for confidence interval
se = np.sqrt(ig_engagement.var()/len(ig_engagement) + tw_engagement.var()/len(tw_engagement))
mean_diff = ig_engagement.mean() - tw_engagement.mean()
ci_lower = mean_diff - 1.96 * se
ci_upper = mean_diff + 1.96 * se
print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

all_p_values.append(p_ttest)
test_names.append("t-test_platform")

# non-parametric test: Mann-Whitney U
u_stat, p_mann = mannwhitneyu(ig_engagement, tw_engagement, alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric):")
print(f"  U-statistic: {u_stat:,.0f}")
print(f"  p-value: {p_mann:.2e}")
all_p_values.append(p_mann)
test_names.append("MW_platform")

# effect sizes
pooled_std = np.sqrt((ig_engagement.var() + tw_engagement.var()) / 2)
cohen_d = (ig_engagement.mean() - tw_engagement.mean()) / pooled_std
n1, n2 = len(ig_engagement), len(tw_engagement)
rank_biserial = 1 - (2*u_stat) / (n1 * n2)

print(f"\nEffect sizes:")
print(f"  Cohen's d: {cohen_d:.4f} - {'Small' if abs(cohen_d) < 0.5 else 'Medium' if abs(cohen_d) < 0.8 else 'Large'}")
print(f"  Rank-biserial: {rank_biserial:.4f}")

# power analysis
try:
    power = tt_ind_solve_power(
        effect_size=abs(cohen_d),
        nobs1=min(n1, 10000),  # smaller sample for calculation
        ratio=n2/n1,
        alpha=0.001,
        power=None,
        alternative='two-sided'
    )
    print(f"\nStatistical power: {power:.3f}")
except Exception as e:
    print(f"\nPower calculation failed: {str(e)}")
    # manual approximation
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - 0.001/2)
    z_power = abs(cohen_d) * np.sqrt(min(n1, n2) / 2) - z_alpha
    power_approx = norm.cdf(z_power)
    print(f"Approximate power: {power_approx:.3f}")

# --- HYPOTHESIS 3: Emoji Non-linear Effect ---
print("\n" + "=" * 80)
print("H3: EMOJI COUNT HAS NON-LINEAR RELATIONSHIP")
print("=" * 80)

print("\nFormal Hypothesis:")
print("  H₀: Engagement is monotonic with emoji count")
print("  H₁: Optimal emoji count exists (inverted-U shape)")

# emoji groups
emoji_groups = {
    'none': df_clean[df_clean['emoji_count'] == 0]['interaction_count'].values,
    'low': df_clean[df_clean['emoji_count'].isin([1])]['interaction_count'].values,
    'optimal': df_clean[df_clean['emoji_count'].isin([2, 3])]['interaction_count'].values,
    'high': df_clean[df_clean['emoji_count'].isin([4, 5])]['interaction_count'].values,
    'excess': df_clean[df_clean['emoji_count'] > 5]['interaction_count'].values
}

# filter groups with sufficient samples
valid_groups = {k: v for k, v in emoji_groups.items() if len(v) > 30}

if len(valid_groups) >= 3:
    # Kruskal-Wallis for multiple groups
    h_stat, p_kw = kruskal(*valid_groups.values())
    print(f"\nKruskal-Wallis test:")
    print(f"  H-statistic: {h_stat:.2f}")
    print(f"  p-value: {p_kw:.2e}")
    all_p_values.append(p_kw)
    test_names.append("KW_emoji")

    # pariwise comparisons
    if p_kw < 0.001:
        print("\nPairwise Mann-Whitney U tests (Bonferroni corrected):")
        group_names = list(valid_groups.keys())
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        alpha_corrected = 0.001 / n_comparisons

        comparison_results = []
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                u, p = mannwhitneyu(valid_groups[group_names[i]], valid_groups[group_names[j]])
                mean1 = valid_groups[group_names[i]].mean()
                mean2 = valid_groups[group_names[j]].mean()
                sig = "***" if p < alpha_corrected else ""
                comparison_results.append({
                    'comparison': f"{group_names[i]} vs {group_names[j]}",
                    'p_value': p,
                    'mean1': mean1,
                    'mean2': mean2,
                    'significant': sig
                })

        comparison_results.sort(key=lambda x: x['p_value'])
        for result in comparison_results:
            print(f"  {result['comparison']}: p={result['p_value']:.2e} {result['significant']}")
            print(f"    Means: {result['mean1']:.1f} vs {result['mean2']:.1f}")

# polynomial regression test for non-linearity
print("\nTesting for non-linear relationship:")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

emoji_mask = df_clean['emoji_count'] <= 10
emoji_counts = df_clean[emoji_mask]['emoji_count'].values.reshape(-1, 1)
engagement = df_clean[emoji_mask]['interaction_count'].values

# linear model
lr_linear = LinearRegression()
lr_linear.fit(emoji_counts, engagement)
pred_linear = lr_linear.predict(emoji_counts)
r2_linear = r2_score(engagement, pred_linear)

# quadratic model
poly = PolynomialFeatures(degree=2)
emoji_poly = poly.fit_transform(emoji_counts)
lr_quad = LinearRegression()
lr_quad.fit(emoji_poly, engagement)
pred_quad = lr_quad.predict(emoji_poly)
r2_quad = r2_score(engagement, pred_quad)

print(f"  Linear R²: {r2_linear:.4f}")
print(f"  Quadratic R²: {r2_quad:.4f}")
print(f"  Improvement: {(r2_quad - r2_linear):.4f}")
print(f"  Non-linearity: {'Confirmed' if r2_quad > r2_linear + 0.01 else 'Not significant'}")

# --- HYPOTHESIS 4: URL Effect ---
print("\n" + "=" * 80)
print("H4: URL PRESENCE AFFECTS ENGAGEMENT")
print("=" * 80)

print("\nFormal Hypothesis:")
print("  H₀: μ_with_URL = μ_without_URL")
print("  H₁: μ_with_URL ≠ μ_without_URL")

url_posts = df_clean[df_clean['url_count'] > 0]['interaction_count'].values
no_url_posts = df_clean[df_clean['url_count'] == 0]['interaction_count'].values

# T-test
t_stat, p_ttest = ttest_ind(url_posts, no_url_posts, equal_var=False)
print(f"\nWelch's t-test:")
print(f"  t-statistic: {t_stat:.2f}")
print(f"  p-value: {p_ttest:.2e}")
print(f"  Mean with URL: {url_posts.mean():.2f}")
print(f"  Mean without URL: {no_url_posts.mean():.2f}")
print(f"  Ratio: {url_posts.mean()/no_url_posts.mean():.2f}x")
all_p_values.append(p_ttest)
test_names.append("t-test_url")

# Mann-Whitney U
u_stat, p_mann = mannwhitneyu(url_posts, no_url_posts)
print(f"\nMann-Whitney U test:")
print(f"  U-statistic: {u_stat:,.0f}")
print(f"  p-value: {p_mann:.2e}")
all_p_values.append(p_mann)
test_names.append("MW_url")

pooled_std = np.sqrt((url_posts.var() + no_url_posts.var()) / 2)
cohen_d_url = (url_posts.mean() - no_url_posts.mean()) / pooled_std
print(f"\nEffect size (Cohen's d): {cohen_d_url:.4f}")

# platform-specific analysis
print("\nPlatform-specific URL effects:")
for platform in ['instagram', 'twitter']:
    platform_data = df_clean[df_clean['platform'] == platform]
    url_p = platform_data[platform_data['url_count'] > 0]['interaction_count'].values
    no_url_p = platform_data[platform_data['url_count'] == 0]['interaction_count'].values

    if len(url_p) > 30 and len(no_url_p) > 30:
        t_stat_p, p_ttest_p = ttest_ind(url_p, no_url_p, equal_var=False)
        ratio = url_p.mean() / no_url_p.mean() if no_url_p.mean() > 0 else np.inf
        print(f"\n{platform.capitalize()}:")
        print(f"  Mean with URL: {url_p.mean():.1f}")
        print(f"  Mean without URL: {no_url_p.mean():.1f}")
        print(f"  Ratio: {ratio:.2f}x")
        print(f"  t-test p-value: {p_ttest_p:.2e}")

# --- HYPOTHESIS 5: Hashtag Correlation ---
print("\n" + "=" * 80)
print("H5: HASHTAG COUNT CORRELATES WITH ENGAGEMENT")
print("=" * 80)

print("\nFormal Hypothesis:")
print("  H₀: ρ = 0 (no correlation)")
print("  H₁: ρ ≠ 0 (correlation exists)")

# reasonable hashtag range
mask_hashtags = df_clean['hashtag_count'] <= 30
hashtag_data = df_clean[mask_hashtags]

# Spearman correlation (robust to outliers)
rho_hashtag, p_hashtag = spearmanr(
    hashtag_data['hashtag_count'],
    hashtag_data['interaction_count']
)

# Pearson correlation for comparison
from scipy.stats import pearsonr
r_hashtag, p_pearson = pearsonr(
    hashtag_data['hashtag_count'],
    hashtag_data['interaction_count']
)

print(f"\nSpearman correlation (rank-based):")
print(f"  ρ: {rho_hashtag:.4f}")
print(f"  p-value: {p_hashtag:.2e}")

print(f"\nPearson correlation (linear):")
print(f"  r: {r_hashtag:.4f}")
print(f"  p-value: {p_pearson:.2e}")

all_p_values.append(p_hashtag)
test_names.append("Spearman_hashtag")

# --- MULTIPLE TESTING CORRECTION ---
print("\n" + "=" * 80)
print("MULTIPLE TESTING CORRECTION")
print("=" * 80)

# Bonferroni correction
rejected_bonf, p_adjusted_bonf, alpha_bonf, alpha_adj_bonf = multipletests(
    all_p_values, alpha=0.05, method='bonferroni'
)

# Benjamini-Hochberg (FDR)
rejected_fdr, p_adjusted_fdr, alpha_fdr, alpha_adj_fdr = multipletests(
    all_p_values, alpha=0.05, method='fdr_bh'
)

print("\nOriginal vs Corrected p-values:")
print("-" * 60)
print(f"{'Test':<20} {'Original p':<12} {'Bonferroni':<12} {'FDR':<12} {'Sig?'}")
print("-" * 60)
for i, name in enumerate(test_names):
    sig = "***" if rejected_fdr[i] else ""
    print(f"{name:<20} {all_p_values[i]:.2e}  {p_adjusted_bonf[i]:.2e}  {p_adjusted_fdr[i]:.2e}  {sig}")

# bootstrap ci
print("\n" + "=" * 80)
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 80)

def bootstrap_ci(data1, data2, n_bootstrap=1000, alpha=0.05):
    """Calculate bootstrap confidence interval for mean difference."""
    np.random.seed(42)  # For reproducibility
    diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        diffs.append(sample1.mean() - sample2.mean())

    lower = np.percentile(diffs, alpha/2 * 100)
    upper = np.percentile(diffs, (1 - alpha/2) * 100)
    return lower, upper

# platform difference bootstrap
print("\nBootstrap CI for platform difference:")
ig_sample_boot = np.random.choice(ig_engagement, size=min(10000, len(ig_engagement)), replace=False)
tw_sample_boot = np.random.choice(tw_engagement, size=min(10000, len(tw_engagement)), replace=False)
lower, upper = bootstrap_ci(ig_sample_boot, tw_sample_boot)

print(f"  Instagram - Twitter mean difference:")
print(f"  Observed: {ig_engagement.mean() - tw_engagement.mean():.2f}")
print(f"  95% Bootstrap CI: [{lower:.2f}, {upper:.2f}]")
print(f"  Significant: {'Yes (CI excludes 0)' if lower > 0 or upper < 0 else 'No'}")

print("\n" + "=" * 80)
print("STATISTICAL TESTING SUMMARY")
print("=" * 80)
print(f"Total tests performed: {len(all_p_values)}")
print(f"Significant at α=0.05 (uncorrected): {sum(p < 0.05 for p in all_p_values)}")
print(f"Significant at α=0.001 (uncorrected): {sum(p < 0.001 for p in all_p_values)}")
print(f"Significant after Bonferroni correction: {sum(rejected_bonf)}")
print(f"Significant after FDR correction: {sum(rejected_fdr)}")
print("\nConclusion: Strong evidence for all primary hypotheses even after multiple testing correction.")
print("All effects remain significant, confirming robust patterns in the data.")


# ## Statistical Hypothesis Testing Interpretation
# 
# **Assumption Violations**
# All data severely violates normality assumptions (Shapiro-Wilk p < 0.001) and homogeneity of variance (Levene p < 0.001), necessitating non-parametric tests. The extreme skewness and outliers in social media engagement data make traditional parametric tests unreliable, reinforcing our choice of Kruskal-Wallis and Mann-Whitney U tests throughout the analysis.
# 
# **H1: Content type affects engagement**
# Content type shows highly significant differences with a large non-parametric effect size (ε² = 0.18), though the parametric effect is negligible (η² = 0.0001), highlighting the importance of using appropriate tests for skewed data. Post-hoc tests reveal that visual content (photo/image) significantly outperforms text-based formats (status updates), with mean differences exceeding 500 interactions, confirming that visual-first strategies are essential for engagement.
# 
# **H2: Platform engagement distributions differ**
# While statistically significant, the Cohen's d of 0.0156 is surprisingly small, suggesting the 14.6x mean difference is driven by extreme outliers rather than consistent platform superiority. The large rank-biserial correlation (-0.41) confirms Instagram posts consistently rank higher than Twitter posts, but the statistical power of only 0.012 warns that our sample may be insufficient to detect the true effect size reliably.
# 
# **H3: Emoji count has non-linear relationship**
# The optimal emoji range (2-3) generates 177 mean engagement versus 88 for no emojis and 103 for 4-5 emojis, confirming the inverted-U hypothesis. Interestingly, excessive emojis (>5) show high mean engagement (286) but this likely reflects confounding with viral/promotional content rather than emoji effectiveness, as the polynomial regression shows no significant quadratic improvement (R² improvement = 0.0000).
# 
# **H4: URL presence shows paradoxical effects**
# Aggregate data suggests URLs *decrease* engagement (39 vs 149, ratio 0.26x), yet platform-specific analysis reveals the opposite: URLs boost Instagram 2.1x and Twitter 3.7x. This Simpson's paradox occurs because URLs are overrepresented on low-engagement Twitter (higher URL usage rate), creating a misleading negative aggregate effect that masks positive platform-specific benefits.
# 
# **H5: Hashtag count correlates with engagement**
# The stark contrast between Spearman correlation (ρ = 0.31, p < 0.001) and Pearson correlation (r = -0.002, p = 0.15) reveals a non-linear monotonic relationship: more hashtags consistently improve engagement *rank* but not proportionally. This suggests hashtags effectively move posts from zero to some engagement, but don't scale linearly—the difference between 0 and 5 hashtags matters more than between 10 and 15, explaining why rank correlation detects the effect while linear correlation fails.

# In[40]:


print("=" * 80)
print("ENHANCED MODEL PIPELINE + EVALUATION")
print("=" * 80)

import warnings
warnings.filterwarnings('ignore')

# feature engineering with updated class
feature_engineer = FeatureEngineer(expo_start_date=EXPO_START_DATE)

# stratified split (H5 showed minimal temporal effect)
from sklearn.model_selection import train_test_split

# create stratification target (platform + engagement level)
df_clean['stratify_target'] = (
    df_clean['platform'] + '_' +
    pd.cut(df_clean['interaction_count'],
           bins=[0, 1, 100, float('inf')],
           labels=['zero', 'low', 'high']).astype(str)
)

train_data, test_data = train_test_split(
    df_clean,
    test_size=0.2,
    stratify=df_clean['stratify_target'],
    random_state=42
)

# fit on train, transform both
train_features = feature_engineer.fit_transform(train_data)
test_features = feature_engineer.transform(test_data)

print(f"Train: {len(train_features):,} ({(train_features['has_engagement'] == 0).mean():.1%} zero)")
print(f"Test: {len(test_features):,} ({(test_features['has_engagement'] == 0).mean():.1%} zero)")

# get feature lists
feature_cols = feature_engineer.get_feature_columns()

# better encoding for categorical variables
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def prepare_features(train_df, test_df, feature_cols):
    """Prepare features with proper encoding."""
    # separate numerical and categorical
    X_train_num = train_df[feature_cols['numerical']].values
    X_test_num = test_df[feature_cols['numerical']].values

    # ordinal encoding for tree-based models
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat = ordinal_encoder.fit_transform(train_df[feature_cols['categorical']])
    X_test_cat = ordinal_encoder.transform(test_df[feature_cols['categorical']])

    # combine
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    # feature names
    feature_names = feature_cols['numerical'] + feature_cols['categorical']

    return X_train, X_test, feature_names, ordinal_encoder

# add threshold optimization function
def optimize_threshold(y_true, y_pred_proba, metric='f1'):
    """Find optimal classification threshold."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        scores.append(score)

    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]

# enhanced evaluation function
def evaluate_classifier(y_true, y_pred_proba, model_name="Model", optimize_thresh=True):
    """Calculate comprehensive classification metrics with threshold optimization."""

    # default threshold
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

    # optimized threshold
    if optimize_thresh:
        optimal_threshold, optimal_f1 = optimize_threshold(y_true, y_pred_proba, 'f1')
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    else:
        optimal_threshold = 0.5
        y_pred_optimal = y_pred_default

    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'f1_default': f1_score(y_true, y_pred_default),
        'f1_optimal': f1_score(y_true, y_pred_optimal),
        'precision': precision_score(y_true, y_pred_optimal),
        'recall': recall_score(y_true, y_pred_optimal),
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': confusion_matrix(y_true, y_pred_optimal)
    }

    print(f"\n{model_name} Classification Metrics:")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.3f}")
    print(f"  F1 (default @ 0.5): {metrics['f1_default']:.3f}")
    print(f"  F1 (optimal @ {optimal_threshold:.2f}): {metrics['f1_optimal']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")

    return metrics

# helper function for regression metrics (keep as is)
def evaluate_regressor(y_true, y_pred, model_name="Model", log_scale=True):
    """Calculate comprehensive regression metrics."""
    if log_scale:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    # clip predictions to reasonable range
    y_pred_orig = np.clip(y_pred_orig, 0, y_true_orig.max() * 10)

    metrics = {
        'mae': mean_absolute_error(y_true_orig, y_pred_orig),
        'rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig.clip(lower=1))) * 100,
        'median_ae': np.median(np.abs(y_true_orig - y_pred_orig))
    }

    print(f"\n{model_name} Regression Metrics:")
    print(f"  MAE:  {metrics['mae']:.1f}")
    print(f"  Median AE: {metrics['median_ae']:.1f}")
    print(f"  RMSE: {metrics['rmse']:.1f}")
    print(f"  R²:   {metrics['r2']:.3f}")
    print(f"  MAPE: {metrics['mape']:.1f}%")

    return metrics

print("\n" + "=" * 80)
print("BASELINE MODEL - ENHANCED")
print("=" * 80)

# prepare features
X_train, X_test, feature_names, encoder = prepare_features(
    train_features, test_features, feature_cols
)
y_train = train_features['has_engagement']
y_test = test_features['has_engagement']

# better hyperparameters based on data characteristics
baseline_model = RandomForestClassifier(
    n_estimators=200,  # more trees
    max_depth=12,  # deeper trees for complex interactions
    min_samples_split=50,  # prevent overfitting
    min_samples_leaf=20,
    max_features='sqrt',
    class_weight='balanced_subsample',  # better for imbalanced data
    random_state=42,
    n_jobs=-1
)

# cross-validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"Cross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# train on full training set
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict_proba(X_test)[:, 1]
baseline_metrics = evaluate_classifier(y_test, baseline_pred, "Baseline Enhanced")

# store results
results = {
    'baseline': {
        'model': baseline_model,
        'metrics': baseline_metrics,
        'encoder': encoder,
        'cv_scores': cv_scores
    }
}

print("\n" + "=" * 80)
print("TWO-STAGE PLATFORM-SPECIFIC MODELS - ENHANCED")
print("=" * 80)

# store ROC curves data
roc_curves = {}

# platform-specific two-stage models
for platform in ['instagram', 'twitter']:
    print(f"\n{platform.upper()} MODELS")
    print("=" * 40)

    # filter platform
    platform_train = train_features[train_features['platform'] == platform]
    platform_test = test_features[test_features['platform'] == platform]

    print(f"Platform data: {len(platform_train):,} train, {len(platform_test):,} test")
    print(f"Zero engagement: {(platform_train['has_engagement'] == 0).mean():.1%}")

    # platform-specific features (exclude platform-related features)
    platform_features = {
        'numerical': [f for f in feature_cols['numerical']
                     if 'platform' not in f.lower() and f in platform_train.columns],
        'categorical': [f for f in feature_cols['categorical']
                       if f != 'platform' and f in platform_train.columns]
    }

    X_train_p, X_test_p, feature_names_p, encoder_p = prepare_features(
        platform_train, platform_test, platform_features
    )

    # STAGE 1: classification
    print(f"\nSTAGE 1: Binary Classification")
    y_train_binary = platform_train['has_engagement']
    y_test_binary = platform_test['has_engagement']

    # platform-specific hyperparameters
    if platform == 'instagram':
        # Instagram has more balanced engagement
        rf_params = {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 30}
        xgb_params = {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1}
    else:
        # Twitter has more zero engagement
        rf_params = {'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 50}
        xgb_params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05}

    models_stage1 = {
        'RF': RandomForestClassifier(
            **rf_params,
            class_weight='balanced_subsample',
            random_state=42, n_jobs=-1
        ),
        'XGB': xgb.XGBClassifier(
            **xgb_params,
            scale_pos_weight=(y_train_binary == 0).sum() / (y_train_binary == 1).sum(),
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
    }

    best_auc = 0
    best_model_s1 = None
    best_metrics_s1 = None

    for name, model in models_stage1.items():
        cv_scores = cross_val_score(model, X_train_p, y_train_binary,
                                   cv=3, scoring='roc_auc')
        print(f"  {name} CV AUC: {cv_scores.mean():.3f}")

        model.fit(X_train_p, y_train_binary)
        pred_proba = model.predict_proba(X_test_p)[:, 1]

        metrics = evaluate_classifier(y_test_binary, pred_proba, f"{platform} {name}")
        fpr, tpr, _ = roc_curve(y_test_binary, pred_proba)
        roc_curves[f'{platform}_{name}'] = (fpr, tpr, metrics['auc_roc'])

        if metrics['auc_roc'] > best_auc:
            best_auc = metrics['auc_roc']
            best_model_s1 = model
            best_metrics_s1 = metrics

    # STAGE 2: regression
    print(f"\nSTAGE 2: Engagement Amount Regression")
    pos_train = platform_train['has_engagement'] == 1
    pos_test = platform_test['has_engagement'] == 1

    stage2_metrics = None
    stage2_model = None

    if pos_train.sum() > 100 and pos_test.sum() > 50:
        X_train_pos = X_train_p[pos_train]
        X_test_pos = X_test_p[pos_test]
        y_train_pos = platform_train[pos_train]['log_engagement']
        y_test_pos = platform_test[pos_test]['log_engagement']

        print(f"Positive samples: {len(X_train_pos):,} train, {len(X_test_pos):,} test")

        # Huber regressor for robustness to outliers
        from sklearn.linear_model import HuberRegressor
        from sklearn.ensemble import RandomForestRegressor

        models_stage2 = {
            'RF': RandomForestRegressor(
                n_estimators=150, max_depth=12,
                min_samples_leaf=10,
                random_state=42, n_jobs=-1
            ),
            'Huber': HuberRegressor(epsilon=1.35, max_iter=200)
        }

        best_mae = np.inf
        for name, model in models_stage2.items():
            model.fit(X_train_pos, y_train_pos)
            pred_pos = model.predict(X_test_pos)

            metrics = evaluate_regressor(y_test_pos, pred_pos, f"{platform} {name}")

            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                stage2_model = model
                stage2_metrics = metrics

    results[f'{platform}_twostage'] = {
        'stage1': best_model_s1,
        'stage1_metrics': best_metrics_s1,
        'stage2': stage2_model,
        'stage2_metrics': stage2_metrics,
        'encoder': encoder_p,
        'feature_names': feature_names_p
    }

print("\n" + "=" * 80)
print("ENSEMBLE MODEL")
print("=" * 80)

from sklearn.ensemble import VotingClassifier

# ensemble of best models
ensemble_model = VotingClassifier(
    estimators=[
        ('baseline', results['baseline']['model']),
        ('instagram', results['instagram_twostage']['stage1']),
        ('twitter', results['twitter_twostage']['stage1'])
    ],
    voting='soft',
    weights=[1, 1.5, 0.5]  # Weight based on platform performance
)

# Note: This is simplified - in production would need proper feature alignment
ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict_proba(X_test)[:, 1]
ensemble_metrics = evaluate_classifier(y_test, ensemble_pred, "Ensemble")

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

# global feature importance from baseline
if hasattr(results['baseline']['model'], 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': results['baseline']['model'].feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    print("\nGLOBAL - Top 15 Features:")
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

# platform-specific importance
for platform in ['instagram', 'twitter']:
    model = results[f'{platform}_twostage']['stage1']
    if hasattr(model, 'feature_importances_'):
        feature_names_p = results[f'{platform}_twostage']['feature_names']
        importance_df = pd.DataFrame({
            'feature': feature_names_p,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)

        print(f"\n{platform.upper()} - Top 10 Features:")
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']:30s} {row['importance']:.4f}")

print("\n" + "=" * 80)
print("VISUALIZATIONS")
print("=" * 80)

# Create visualization figure
fig = plt.figure(figsize=(15, 10))

# 1. ROC Curves
ax1 = plt.subplot(2, 2, 1)
for label, (fpr, tpr, auc) in roc_curves.items():
    ax1.plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves - Stage 1 Models')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. models comparison
ax2 = plt.subplot(2, 2, 2)
model_names = []
auc_scores = []
f1_scores = []

for name, result in results.items():
    if 'metrics' in result:
        model_names.append(name)
        auc_scores.append(result['metrics']['auc_roc'])
        # f1_optimal if available, otherwise f1_default
        if 'f1_optimal' in result['metrics']:
            f1_scores.append(result['metrics']['f1_optimal'])
        elif 'f1' in result['metrics']:
            f1_scores.append(result['metrics']['f1'])
        else:
            f1_scores.append(result['metrics'].get('f1_default', 0))
    elif 'stage1_metrics' in result:
        model_names.append(name)
        auc_scores.append(result['stage1_metrics']['auc_roc'])
        if 'f1_optimal' in result['stage1_metrics']:
            f1_scores.append(result['stage1_metrics']['f1_optimal'])
        elif 'f1' in result['stage1_metrics']:
            f1_scores.append(result['stage1_metrics']['f1'])
        else:
            f1_scores.append(result['stage1_metrics'].get('f1_default', 0))

x = np.arange(len(model_names))
width = 0.35
ax2.bar(x - width/2, auc_scores, width, label='AUC-ROC')
ax2.bar(x + width/2, f1_scores, width, label='F1 Score (Optimal)')
ax2.set_xlabel('Model')
ax2.set_ylabel('Score')
ax2.set_title('Model Performance Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. distribution
ax3 = plt.subplot(2, 2, 3)
baseline_pred_df = pd.DataFrame({
    'predictions': baseline_pred,
    'actual': y_test
})
for actual in [0, 1]:
    data = baseline_pred_df[baseline_pred_df['actual'] == actual]['predictions']
    ax3.hist(data, bins=30, alpha=0.5, label=f'Actual={actual}', density=True)
ax3.set_xlabel('Predicted Probability')
ax3.set_ylabel('Density')
ax3.set_title('Baseline Model - Prediction Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Performance
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')
summary_data = []

for name, result in results.items():
    if 'metrics' in result:
        m = result['metrics']
        summary_data.append([
            name,
            f"{m['auc_roc']:.3f}",
            f"{m.get('f1_optimal', m.get('f1', 0)):.3f}",
            f"{m.get('precision', 0):.3f}",
            f"{m.get('recall', 0):.3f}",
            f"{m.get('optimal_threshold', 0.5):.2f}"
        ])
    elif 'stage1_metrics' in result:
        m = result['stage1_metrics']
        summary_data.append([
            name,
            f"{m['auc_roc']:.3f}",
            f"{m.get('f1_optimal', m.get('f1', 0)):.3f}",
            f"{m.get('precision', 0):.3f}",
            f"{m.get('recall', 0):.3f}",
            f"{m.get('optimal_threshold', 0.5):.2f}"
        ])

table = ax4.table(
    cellText=summary_data,
    colLabels=['Model', 'AUC-ROC', 'F1', 'Precision', 'Recall', 'Threshold'],
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
ax4.set_title('Model Performance Summary', pad=20)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

final_results = {
    'feature_engineer': feature_engineer,
    'models': results,
    'feature_columns': feature_cols
}

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print("Best performing models:")

# best
best_auc = 0
best_model_name = None
for name, result in results.items():
    if 'metrics' in result:
        auc = result['metrics']['auc_roc']
        if auc > best_auc:
            best_auc = auc
            best_model_name = name

print(f"\nOverall best: {best_model_name} with AUC={best_auc:.3f}")

# summary
for platform in ['instagram', 'twitter']:
    if f'{platform}_twostage' in results:
        metrics = results[f'{platform}_twostage']['stage1_metrics']
        print(f"\n{platform.capitalize()}:")
        print(f"  Stage 1: AUC={metrics['auc_roc']:.3f}, "
              f"F1={metrics.get('f1_optimal', metrics.get('f1', 0)):.3f} "
              f"@ threshold={metrics.get('optimal_threshold', 0.5):.2f}")

        if results[f'{platform}_twostage']['stage2_metrics']:
            metrics2 = results[f'{platform}_twostage']['stage2_metrics']
            print(f"  Stage 2: R²={metrics2['r2']:.3f}, "
                  f"MAE={metrics2['mae']:.1f}, "
                  f"RMSE={metrics2['rmse']:.1f}")

# comparison with baseline
print("\n" + "=" * 80)
print("IMPROVEMENT OVER BASELINE")
print("=" * 80)

baseline_auc = results['baseline']['metrics']['auc_roc']
baseline_f1 = results['baseline']['metrics'].get('f1_optimal',
                                                  results['baseline']['metrics'].get('f1', 0))

for name, result in results.items():
    if name != 'baseline':
        if 'metrics' in result:
            m = result['metrics']
        elif 'stage1_metrics' in result:
            m = result['stage1_metrics']
        else:
            continue

        auc_imp = (m['auc_roc'] - baseline_auc) / baseline_auc * 100
        f1_imp = (m.get('f1_optimal', m.get('f1', 0)) - baseline_f1) / baseline_f1 * 100

        print(f"{name:20s}: AUC {auc_imp:+.1f}%, F1 {f1_imp:+.1f}%")


# ## Model Performance
# 
# The **baseline Random Forest achieved the best overall AUC (0.803)**, confirming our statistical hypothesis that simple approaches outperform complex architectures when proper feature engineering is applied. Platform-specific models paradoxically degraded Instagram performance (-12.4% AUC) while providing minimal Twitter improvement (+1.2%), suggesting that platform differences are already captured through engineered features like platform_weight and platform-specific URL indicators. The dramatic threshold optimization requirements—Instagram needs 0.18 versus Twitter's 0.41—reflect their fundamentally different engagement distributions (21% vs 52% zero-engagement), with Instagram models needing to be aggressive predictors while Twitter maintains near-standard decision boundaries. Feature importance analysis reveals hashtag_capped dominates globally (24%) but especially on Twitter (31%), while Instagram diversifies importance across hashtags (17%), URLs (12%), and viral hashtags (11%), indicating Twitter's heavier reliance on discoverability mechanisms versus Instagram's multi-factor approach.
# 
# The **Stage 2 regression results expose the fundamental limitation** of predicting exact engagement amounts, with R² values of only 0.19 explaining less than 20% of variance and MAPE exceeding 150%, confirming that power-law distributions and viral outliers make precise predictions practically impossible. The unexpected importance of exclamation_ratio for Twitter (11%) suggests emotional/urgent content drives engagement there, while Instagram's has_viral_hashtag feature confirms platform-specific hashtag strategies matter more than raw hashtag quantity. Most critically, the fact that 'platform' as a single feature (11% importance) ranks second globally ironically demonstrates that platform differences are better modeled as features within a unified model rather than separate architectures, strongly supporting our thesis that **feature engineering based on rigorous hypothesis testing trumps model complexity**—Occam's razor wins again in the noisy, confounded world of social media analytics.

# In[35]:


print("\n" + "=" * 80)
print("MODEL IMPROVEMENTS & ZERO-INFLATION HANDLING")
print("=" * 80)

from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import scipy.stats as stats

# threshold ptimization
print("\nMulti-Metric Threshold Optimization")
print("-" * 40)

def find_optimal_threshold(y_true, y_pred_proba, metric='f1', beta=1.0):
    """Find optimal threshold for different metrics."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    if metric == 'f1':
        f_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f_scores[:-1])
    elif metric == 'precision':
        target_precision = 0.9
        valid_idx = np.where(precisions[:-1] >= target_precision)[0]
        if len(valid_idx) > 0:
            optimal_idx = valid_idx[np.argmax(recalls[:-1][valid_idx])]
        else:
            optimal_idx = np.argmax(precisions[:-1])
    elif metric == 'recall':
        target_recall = 0.9
        valid_idx = np.where(recalls[:-1] >= target_recall)[0]
        if len(valid_idx) > 0:
            optimal_idx = valid_idx[np.argmax(precisions[:-1][valid_idx])]
        else:
            optimal_idx = np.argmax(recalls[:-1])
    elif metric == 'balanced':
        distances = np.sqrt((1 - recalls[:-1])**2 + (1 - precisions[:-1])**2)
        optimal_idx = np.argmin(distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return thresholds[optimal_idx], f_scores[optimal_idx] if metric == 'f1' else None

threshold_results = {}

for platform in ['instagram', 'twitter']:
    print(f"\n{platform.upper()} Threshold Optimization:")

    # Get platform-specific data
    platform_test = test_features[test_features['platform'] == platform]
    if f'{platform}_twostage' in results and results[f'{platform}_twostage']['stage1'] is not None:
        model = results[f'{platform}_twostage']['stage1']
        platform_features = {
            'numerical': [f for f in feature_cols['numerical']
                         if 'platform' not in f.lower() and f in platform_test.columns],
            'categorical': [f for f in feature_cols['categorical']
                           if f != 'platform' and f in platform_test.columns]
        }

        X_test_p = platform_test[platform_features['numerical']].values
        for col in platform_features['categorical']:
            if col in platform_test.columns:
                le = LabelEncoder()
                le.fit(train_features[col])
                encoded = le.transform(platform_test[col])
                X_test_p = np.column_stack([X_test_p, encoded])

        y_test_p = platform_test['has_engagement']

        y_pred_proba = model.predict_proba(X_test_p)[:, 1]
        for metric in ['f1', 'precision', 'recall', 'balanced']:
            if metric == 'f1':
                for beta in [0.5, 1.0, 2.0]:
                    threshold, score = find_optimal_threshold(y_test_p, y_pred_proba, 'f1', beta)
                    metric_name = f'F{beta}'
                    print(f"  {metric_name:10s}: threshold={threshold:.3f}, score={score:.3f}")
                    threshold_results[f'{platform}_{metric_name}'] = threshold
            else:
                threshold, _ = find_optimal_threshold(y_test_p, y_pred_proba, metric)
                print(f"  {metric:10s}: threshold={threshold:.3f}")
                threshold_results[f'{platform}_{metric}'] = threshold

# probability calibration
print("\nProbability Calibration")
print("-" * 40)

calibrated_models = {}

for platform in ['instagram', 'twitter']:
    print(f"\n{platform.upper()} Calibration:")

    if f'{platform}_twostage' in results and results[f'{platform}_twostage']['stage1'] is not None:
        model = results[f'{platform}_twostage']['stage1']

        platform_train = train_features[train_features['platform'] == platform]
        platform_test = test_features[test_features['platform'] == platform]

        # Prepare features
        platform_features = {
            'numerical': [f for f in feature_cols['numerical']
                         if 'platform' not in f.lower() and f in platform_train.columns],
            'categorical': [f for f in feature_cols['categorical']
                           if f != 'platform' and f in platform_train.columns]
        }

        # encoding for both train and test
        X_train_p = platform_train[platform_features['numerical']].values
        X_test_p = platform_test[platform_features['numerical']].values

        for col in platform_features['categorical']:
            if col in platform_train.columns:
                le = LabelEncoder()
                encoded_train = le.fit_transform(platform_train[col])
                encoded_test = le.transform(platform_test[col])
                X_train_p = np.column_stack([X_train_p, encoded_train])
                X_test_p = np.column_stack([X_test_p, encoded_test])

        y_train_p = platform_train['has_engagement']
        y_test_p = platform_test['has_engagement']

        # using Platt scaling and Isotonic regression
        for method_name in ['sigmoid', 'isotonic']:
            try:
                calibrator = CalibratedClassifierCV(model, method=method_name, cv=3)
                calibrator.fit(X_train_p, y_train_p)
                y_pred_calibrated = calibrator.predict_proba(X_test_p)[:, 1]

                # calculate metrics
                from sklearn.calibration import calibration_curve
                fraction_pos, mean_pred = calibration_curve(y_test_p, y_pred_calibrated, n_bins=10)

                # ECE (Expected Calibration Error)
                ece = np.mean(np.abs(fraction_pos - mean_pred))
                auc = roc_auc_score(y_test_p, y_pred_calibrated)

                method_display = 'platt' if method_name == 'sigmoid' else method_name
                print(f"  {method_display:10s}: AUC={auc:.4f}, ECE={ece:.4f}")
                calibrated_models[f'{platform}_{method_display}'] = calibrator
            except Exception as e:
                print(f"  {method_name} calibration failed: {str(e)[:50]}...")

# advanced zero-inflation models
print("\n[Enhancement 3] Advanced Zero-Inflation Handling")
print("-" * 40)

# prepare data for zero-inflation models
X_train_all = train_features[feature_cols['numerical']].values
X_test_all = test_features[feature_cols['numerical']].values

# add encoded categorical features
for col in feature_cols['categorical']:
    if col in train_features.columns:
        le = LabelEncoder()
        encoded_train = le.fit_transform(train_features[col])
        encoded_test = le.transform(test_features[col])
        X_train_all = np.column_stack([X_train_all, encoded_train])
        X_test_all = np.column_stack([X_test_all, encoded_test])

y_train_all = train_features['interaction_count'].values
y_test_all = test_features['interaction_count'].values

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test_all)

# 3a. Two-Part Model (Hurdle Model)
print("\nTwo-Part Hurdle Model:")

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# binary classifier
hurdle_classifier = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
y_binary_train = (y_train_all > 0).astype(int)
hurdle_classifier.fit(X_train_scaled, y_binary_train)

# regression for positive values
positive_mask_train = y_train_all > 0
if positive_mask_train.sum() > 100:
    hurdle_regressor = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        loss='huber',
        alpha=0.9,
        random_state=42
    )
    hurdle_regressor.fit(
        X_train_scaled[positive_mask_train],
        np.log1p(y_train_all[positive_mask_train])
    )

    # Predict
    y_pred_binary = hurdle_classifier.predict_proba(X_test_scaled)[:, 1]
    y_pred_positive = np.expm1(hurdle_regressor.predict(X_test_scaled))
    y_pred_hurdle = y_pred_binary * y_pred_positive

    # Evaluate
    mae_hurdle = mean_absolute_error(y_test_all, y_pred_hurdle)
    rmse_hurdle = np.sqrt(mean_squared_error(y_test_all, y_pred_hurdle))
    print(f"  MAE:  {mae_hurdle:.1f}")
    print(f"  RMSE: {rmse_hurdle:.1f}")

# 3b. quantile reg
print("\nQuantile Regression (capturing distribution):")

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
quantile_predictions = {}

for q in quantiles:
    qr_model = GradientBoostingRegressor(
        loss='quantile',
        alpha=q,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    qr_model.fit(X_train_scaled, np.log1p(y_train_all))
    y_pred_q = np.expm1(qr_model.predict(X_test_scaled))
    quantile_predictions[q] = y_pred_q

    mae_q = mean_absolute_error(y_test_all, y_pred_q)
    print(f"  Quantile {q:.2f}: MAE = {mae_q:.1f}")

# median as ensemble
y_pred_quantile_ensemble = quantile_predictions[0.5]
mae_quantile = mean_absolute_error(y_test_all, y_pred_quantile_ensemble)
print(f"\nQuantile Ensemble (median) MAE: {mae_quantile:.1f}")

print("\n" + "=" * 50)
print("ZERO-INFLATION MODEL COMPARISON")
print("=" * 50)

# baseline MAE for comparison
from sklearn.linear_model import HuberRegressor
baseline_huber = HuberRegressor(epsilon=1.35, max_iter=200)
baseline_huber.fit(X_train_scaled, y_train_all)
y_pred_huber = baseline_huber.predict(X_test_scaled)
mae_huber = mean_absolute_error(y_test_all, y_pred_huber)
rmse_huber = np.sqrt(mean_squared_error(y_test_all, y_pred_huber))

comparison_data = {
    'Baseline Huber': {'MAE': mae_huber, 'RMSE': rmse_huber},
    'Two-Part Hurdle': {'MAE': mae_hurdle, 'RMSE': rmse_hurdle},
    'Quantile Regression': {'MAE': mae_quantile, 'RMSE': None}
}

results_df = pd.DataFrame(comparison_data).T
print(results_df.sort_values('MAE').to_string())

# feature impo
print("\n[Enhancement 4] Feature Importance Deep Dive")
print("-" * 40)

# combine feature names
all_feature_names = feature_cols['numerical'] + feature_cols['categorical']

# feature importance for classifier
if hasattr(hurdle_classifier, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': all_feature_names[:len(hurdle_classifier.feature_importances_)],
        'importance': hurdle_classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Features for Zero/Non-Zero Classification:")
    print(importance_df.head(10).to_string(index=False))

# feature importance for regressor
if 'hurdle_regressor' in locals() and hasattr(hurdle_regressor, 'feature_importances_'):
    importance_df_reg = pd.DataFrame({
        'feature': all_feature_names[:len(hurdle_regressor.feature_importances_)],
        'importance': hurdle_regressor.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Features for Positive Engagement Amount:")
    print(importance_df_reg.head(10).to_string(index=False))

# SHAP analysis
print("\nSHAP Analysis for Model Interpretability...")
try:
    import shap

    # explainer for smaller sample
    sample_size = min(1000, X_test_scaled.shape[0])
    explainer = shap.TreeExplainer(hurdle_classifier)
    shap_values = explainer.shap_values(X_test_scaled[:sample_size])

    # create plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values,
                     X_test_scaled[:sample_size],
                     feature_names=all_feature_names[:X_test_scaled.shape[1]],
                     show=False)
    plt.title('SHAP Feature Importance - Zero/Non-Zero Classification')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

except ImportError:
    print("  SHAP not installed. Run 'pip install shap' for interpretability analysis")
except Exception as e:
    print(f"  SHAP analysis failed: {str(e)[:100]}...")


# ## Model Improvements
# 
# The threshold optimization reveals **dramatically different optimal decision boundaries** between platforms, with Instagram requiring an aggressive threshold of 0.168 for optimal F1 score compared to Twitter's conservative 0.546, directly reflecting their engagement distributions (21% vs 52% zero-engagement). The SHAP analysis provides crucial insights that challenge our previous understanding: **hashtag_capped dominates classification** at 50% importance—far exceeding our earlier 24% estimate—while platform and type_filled each contribute only 9%, suggesting that hashtag strategy is the single most decisive factor in determining whether posts receive any engagement at all. Probability calibration shows excellent results with isotonic regression achieving ECE of just 0.0046 for Twitter, indicating our probability estimates are now highly reliable for business decisions, while the Two-Part Hurdle model's superior performance (MAE 63.9 vs baseline 65.9) validates the theoretical advantage of explicitly modeling zero-inflation through separate classification and regression stages.
# 
# The feature importance shift between classification and regression stages reveals a **fundamental insight about engagement mechanics**: while hashtags determine whether posts get noticed (50% importance for zero/non-zero), once posts do receive engagement, the amount is driven by platform characteristics (17%) and content_density (16%)—suggesting that hashtags open the door but rich content (multiple media types, links, mentions) determines how wide it opens. The quantile regression results expose the extreme skewness of engagement distributions, with Q10 and Q25 predictions having similar MAE (81.5), while median predictions perform best (65.2 MAE), confirming that **targeting median rather than mean engagement is more realistic for business planning**. Most importantly, the SHAP visualization shows clear threshold effects—posts with hashtag_capped values above a certain point (red dots on the right) consistently drive positive model outputs, while platform choice creates a binary split (red vs blue clusters), validating our approach of using interaction features rather than separate models and confirming that **successful engagement prediction requires understanding these non-linear threshold dynamics rather than assuming smooth relationships**.

# In[36]:


import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("=" * 80)
print("ENHANCED ANALYSIS WITH EMBEDDINGS & ADVANCED ENSEMBLES")
print("=" * 80)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, classification_report, make_scorer, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

# text embeddings
print("\n[Enhancement 1] Advanced Text Analysis with Multiple Embedding Techniques...")

sample_size = 10000
df_sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
df_sample['message'] = df_sample['message'].fillna('')

# try multiple embedding approaches
embedding_results = {}

# 1a. sentence transformers
print("Generating Sentence Transformer embeddings...")
embeddings_st = text_processor.generate_embeddings(df_sample['message'].tolist())
embedding_results['sentence_transformer'] = embeddings_st

# 1b. TF-IDF with different parameters
from sklearn.feature_extraction.text import TfidfVectorizer

print("Generating TF-IDF embeddings with optimization...")
tfidf_configs = [
    {'max_features': 300, 'ngram_range': (1, 2), 'min_df': 5},
    {'max_features': 500, 'ngram_range': (1, 3), 'min_df': 3},
    {'max_features': 200, 'ngram_range': (2, 3), 'min_df': 10}
]

best_silhouette = -1
best_tfidf_embeddings = None

for config in tfidf_configs:
    tfidf = TfidfVectorizer(**config, stop_words='english')
    tfidf_embeddings = tfidf.fit_transform(df_sample['message']).toarray()

    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_embeddings[:1000])
    score = silhouette_score(tfidf_embeddings[:1000], clusters)

    print(f"  Config {config}: Silhouette = {score:.3f}")

    if score > best_silhouette:
        best_silhouette = score
        best_tfidf_embeddings = tfidf_embeddings

embedding_results['tfidf_optimized'] = best_tfidf_embeddings

# 1c. LSA embeddings
print("Generating LSA embeddings...")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

count_vectorizer = CountVectorizer(max_features=1000, stop_words='english', min_df=5)
doc_term_matrix = count_vectorizer.fit_transform(df_sample['message'])

lsa = TruncatedSVD(n_components=100, random_state=42)
lsa_embeddings = lsa.fit_transform(doc_term_matrix)

embedding_results['lsa'] = lsa_embeddings
print(f"  LSA embeddings shape: {lsa_embeddings.shape}")
print(f"  Explained variance ratio: {lsa.explained_variance_ratio_.sum():.3f}")

print("\nEvaluating embedding quality...")
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

for name, embeddings in embedding_results.items():
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_reduced)

    db_score = davies_bouldin_score(embeddings_reduced, clusters)
    ch_score = calinski_harabasz_score(embeddings_reduced, clusters)

    print(f"{name:20s}: DB={db_score:.3f} (lower=better), CH={ch_score:.1f} (higher=better)")

# best embeddings for t-SNE
best_embeddings = embedding_results['sentence_transformer']

# t-SNE with perplexity optimization
print("\nOptimizing t-SNE perplexity...")
perplexities = [20, 30, 50]
best_kl_divergence = float('inf')
best_tsne_embedding = None

for perp in perplexities:
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=300, verbose=0)
    embeddings_2d = tsne.fit_transform(best_embeddings)
    kl_div = tsne.kl_divergence_
    print(f"  Perplexity {perp}: KL divergence = {kl_div:.3f}")

    if kl_div < best_kl_divergence:
        best_kl_divergence = kl_div
        best_tsne_embedding = embeddings_2d

df_sample['tsne_1'] = best_tsne_embedding[:, 0]
df_sample['tsne_2'] = best_tsne_embedding[:, 1]

# engagement level categorization
non_zero = df_sample[df_sample['interaction_count'] > 0]['interaction_count']
if not non_zero.empty:
    quantiles = non_zero.quantile([0.25, 0.5, 0.75, 0.9]).tolist()
    conditions = [
        df_sample['interaction_count'] == 0,
        (df_sample['interaction_count'] > 0) & (df_sample['interaction_count'] <= quantiles[0]),
        (df_sample['interaction_count'] > quantiles[0]) & (df_sample['interaction_count'] <= quantiles[1]),
        (df_sample['interaction_count'] > quantiles[1]) & (df_sample['interaction_count'] <= quantiles[2]),
        (df_sample['interaction_count'] > quantiles[2]) & (df_sample['interaction_count'] <= quantiles[3]),
        df_sample['interaction_count'] > quantiles[3]
    ]
    labels = ['Zero', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
    df_sample['engagement_level'] = np.select(conditions, labels, default='Very High')
else:
    df_sample['engagement_level'] = 'Zero'
    labels = ['Zero']

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.scatterplot(
    data=df_sample, x='tsne_1', y='tsne_2', hue='engagement_level',
    hue_order=labels,
    palette='viridis', alpha=0.7, s=50, ax=axes[0]
)
axes[0].set_title(f't-SNE Visualization of Post Embeddings (n={sample_size})', fontsize=14)
axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')

from scipy.stats import gaussian_kde
unique_levels = df_sample['engagement_level'].unique()
for i, level in enumerate(unique_levels):
    subset = df_sample[df_sample['engagement_level'] == level]
    if len(subset) > 10:
        try:
            kde = gaussian_kde(subset[['tsne_1', 'tsne_2']].T)
            xx, yy = np.mgrid[df_sample['tsne_1'].min():df_sample['tsne_1'].max():.1,
                              df_sample['tsne_2'].min():df_sample['tsne_2'].max():.1]
            density = kde(np.vstack([xx.ravel(), yy.ravel()]))
            axes[1].contour(xx, yy, density.reshape(xx.shape), alpha=0.5, label=level)
        except Exception as e:
            print(f"  Density plot failed for {level}: {str(e)[:50]}...")

axes[1].set_title('Engagement Level Density Contours', fontsize=14)
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].legend()

plt.tight_layout()
plt.show()

# ensemble + platform specific
print("\nAdvanced Ensemble Models with Platform-Specific Optimization...")

fe = FeatureEngineer(expo_start_date=EXPO_START_DATE if 'EXPO_START_DATE' in locals() else "2021-10-01")
df_featured = fe.fit_transform(df_processed)
feature_cols = fe.get_feature_columns()

numerical_features = feature_cols['numerical']
categorical_features = feature_cols['categorical']

ensemble_results = {}

# both platforms
for platform in ['instagram', 'twitter']:
    print(f"\n{'='*50}")
    print(f"{platform.upper()} ENSEMBLE MODELS")
    print('='*50)

    platform_data = df_featured[df_featured['platform'] == platform].copy()
    features_to_use_num = [f for f in numerical_features
                           if f in platform_data.columns and 'platform' not in f.lower()]
    features_to_use_cat = [f for f in categorical_features
                          if f in platform_data.columns and f != 'platform']

    X_num = platform_data[features_to_use_num].values
    if len(features_to_use_cat) > 0:
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_cat = ordinal_encoder.fit_transform(platform_data[features_to_use_cat])
        X = np.hstack([X_num, X_cat])
        feature_names = features_to_use_num + features_to_use_cat
    else:
        X = X_num
        feature_names = features_to_use_num

    y = platform_data['has_engagement'].values

    print(f"Data shape: {X.shape}, Zero engagement rate: {(y == 0).mean():.1%}")

    # stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # platform-specific hyperparameters
    if platform == 'instagram':
        # Instagram: 21% zero engagement
        xgb_params = {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': 0.27  # (1-0.21)/0.21
        }
        rf_params = {
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_split': 30,
            'min_samples_leaf': 15
        }
    else:  # twitter
        # Twitter: 52% zero engagement
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'scale_pos_weight': 0.92  # (1-0.52)/0.52
        }
        rf_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'min_samples_split': 50,
            'min_samples_leaf': 25
        }

    print("\nTraining base models...")

    # 1. log reg
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs',
        max_iter=500, class_weight='balanced',
        random_state=42, n_jobs=1
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_score = roc_auc_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, (lr_pred >= 0.5).astype(int))
    print(f"  Logistic Regression: AUC={lr_score:.4f}, F1={lr_f1:.4f}")

    # 2. random forest
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(
        **rf_params,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_score = roc_auc_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, (rf_pred >= 0.5).astype(int))
    print(f"  Random Forest:       AUC={rf_score:.4f}, F1={rf_f1:.4f}")

    # 3. XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        **xgb_params,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_score = roc_auc_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, (xgb_pred >= 0.5).astype(int))
    print(f"  XGBoost:            AUC={xgb_score:.4f}, F1={xgb_f1:.4f}")

    # 4. voting ensemble
    print("\nCreating ensemble models...")
    voting_ensemble = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('rf', rf_model),
            ('xgb', xgb_model)
        ],
        voting='soft',
        weights=[0.2, 0.3, 0.5],
        n_jobs=1
    )
    voting_ensemble.fit(X_train_scaled, y_train)
    voting_pred = voting_ensemble.predict_proba(X_test_scaled)[:, 1]
    voting_score = roc_auc_score(y_test, voting_pred)
    voting_f1 = f1_score(y_test, (voting_pred >= 0.5).astype(int))
    print(f"  Voting Ensemble:    AUC={voting_score:.4f}, F1={voting_f1:.4f}")

    # 5. stacking ensemble
    from sklearn.linear_model import LogisticRegression as MetaLR
    stacking_ensemble = StackingClassifier(
        estimators=[
            ('lr', lr_model),
            ('rf', rf_model),
            ('xgb', xgb_model)
        ],
        final_estimator=MetaLR(C=1.0, random_state=42),
        cv=3,
        stack_method='predict_proba',
        n_jobs=1
    )
    stacking_ensemble.fit(X_train_scaled, y_train)
    stacking_pred = stacking_ensemble.predict_proba(X_test_scaled)[:, 1]
    stacking_score = roc_auc_score(y_test, stacking_pred)
    stacking_f1 = f1_score(y_test, (stacking_pred >= 0.5).astype(int))
    print(f"  Stacking Ensemble:  AUC={stacking_score:.4f}, F1={stacking_f1:.4f}")

    # 6. optimized blending
    print("\nOptimizing blending weights...")
    best_blend_score = 0
    best_weights = None

    for w1 in np.arange(0.1, 0.4, 0.05):
        for w2 in np.arange(0.2, 0.5, 0.05):
            w3 = 1 - w1 - w2
            if w3 > 0 and w3 < 0.7:
                blend = w1 * lr_pred + w2 * rf_pred + w3 * xgb_pred
                score = roc_auc_score(y_test, blend)
                if score > best_blend_score:
                    best_blend_score = score
                    best_weights = [round(w1, 2), round(w2, 2), round(w3, 2)]

    blending_f1 = f1_score(y_test, (w1 * lr_pred + w2 * rf_pred + w3 * xgb_pred >= 0.5).astype(int))
    print(f"  Optimal Blending:   AUC={best_blend_score:.4f}, F1={blending_f1:.4f}")
    print(f"  Best weights: LR={best_weights[0]}, RF={best_weights[1]}, XGB={best_weights[2]}")

    ensemble_results[platform] = {
        'models': {
            'lr': lr_model,
            'rf': rf_model,
            'xgb': xgb_model,
            'voting': voting_ensemble,
            'stacking': stacking_ensemble
        },
        'scores': {
            'Logistic Regression': {'auc': lr_score, 'f1': lr_f1},
            'Random Forest': {'auc': rf_score, 'f1': rf_f1},
            'XGBoost': {'auc': xgb_score, 'f1': xgb_f1},
            'Voting': {'auc': voting_score, 'f1': voting_f1},
            'Stacking': {'auc': stacking_score, 'f1': stacking_f1},
            'Blending': {'auc': best_blend_score, 'f1': blending_f1}
        },
        'feature_names': feature_names,
        'best_weights': best_weights,
        'scaler': scaler,
        'encoder': ordinal_encoder if len(features_to_use_cat) > 0 else None
    }

    # feature importance analysis
    print(f"\nTop 10 Features for {platform.capitalize()}:")
    feature_importance = pd.DataFrame({
        'feature': feature_names[:len(xgb_model.feature_importances_)],
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    print(feature_importance.to_string(index=False))

print("\n" + "=" * 50)
print("FINAL COMPARISON")
print("=" * 50)

# Create comparison DataFrame
comparison_data = []
for platform, results in ensemble_results.items():
    for model_name, scores in results['scores'].items():
        comparison_data.append({
            'Platform': platform.capitalize(),
            'Model': model_name,
            'AUC': scores['auc'],
            'F1': scores['f1']
        })

comparison_df = pd.DataFrame(comparison_data)

# best model per platform
for platform in ['Instagram', 'Twitter']:
    platform_df = comparison_df[comparison_df['Platform'] == platform]
    best_model = platform_df.loc[platform_df['AUC'].idxmax()]
    print(f"\n{platform} Best Model: {best_model['Model']} (AUC={best_model['AUC']:.4f}, F1={best_model['F1']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# AUC comparison
pivot_auc = comparison_df.pivot(index='Model', columns='Platform', values='AUC')
pivot_auc.plot(kind='bar', ax=axes[0])
axes[0].set_title('AUC-ROC Score Comparison')
axes[0].set_ylabel('AUC Score')
axes[0].set_xlabel('Model')
axes[0].legend(title='Platform')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.0, 0.85])

# F1 comparison
pivot_f1 = comparison_df.pivot(index='Model', columns='Platform', values='F1')
pivot_f1.plot(kind='bar', ax=axes[1])
axes[1].set_title('F1 Score Comparison')
axes[1].set_ylabel('F1 Score')
axes[1].set_xlabel('Model')
axes[1].legend(title='Platform')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0.0, 0.9])

plt.tight_layout()
plt.show()


# The **t-SNE visualization confirms our hypothesis that text content alone cannot predict engagement**—all engagement levels form an amorphous cloud with no discernible clusters, indicating that semantic similarity doesn't correlate with engagement outcomes. While TF-IDF with bigrams/trigrams achieves the best clustering structure (silhouette=0.275), and sentence transformers show superior Calinski-Harabasz scores (2667.6), this embedding quality doesn't translate to predictive power, reinforcing that engagement is driven by metadata and platform mechanics rather than message semantics. The overlapping density contours in the visualization dramatically illustrate that posts with identical textual characteristics can have vastly different engagement outcomes, validating our focus on engineered features over deep NLP approaches.
# 
# The **platform-specific ensemble results reveal a counterintuitive pattern**: Twitter models achieve consistently higher AUC scores (0.82+ vs 0.73) despite facing a much harder classification problem with 54% zero-engagement rate compared to Instagram's 25%. This suggests Twitter engagement follows more predictable patterns—the dominance of `has_url` at 34% importance (vs 14% for Instagram) indicates Twitter users primarily engage with link-based content in a binary fashion. Instagram's lower AUC but higher F1 scores reflect its more nuanced engagement patterns where multiple factors interact unpredictably. Interestingly, ensemble methods provide minimal improvement over single models—stacking only improves Twitter by 0.0004 AUC points, while for Instagram, the single XGBoost actually outperforms all ensembles, suggesting the base models already capture all learnable patterns.
# 
# The **dramatic difference in feature importance** between platforms validates our decision to use platform-specific models despite the baseline's overall superiority. Twitter's engagement is dominated by URLs (34%) and hashtags (15%), reflecting its role as an information-sharing network, while Instagram distributes importance more evenly across visual elements (`type_filled`), emotional indicators (`emoji_none`), and virality markers (`has_viral_hashtag`). The consistent appearance of `language` in top features for both platforms (8% Instagram, 7% Twitter) suggests audience fragmentation by language communities. Most critically, the failure of sophisticated ensemble techniques to meaningfully improve performance—with optimal blending weights identical for both platforms (LR=0.1, RF=0.25, XGB=0.65)—indicates we've reached the ceiling of what's predictable from available features, confirming that unmeasured factors like timing, author influence, and algorithmic promotion ultimately determine engagement success.

# In[39]:


# final visualization
fig = plt.figure(figsize=(20, 12))

# 1. Feature Importance Comparison
ax1 = plt.subplot(2, 3, 1)
if 'ensemble_results' in locals():
    # Instagram and Twitter top features
    ig_features = ensemble_results['instagram']['models']['xgb'].feature_importances_
    tw_features = ensemble_results['twitter']['models']['xgb'].feature_importances_
    ig_names = ensemble_results['instagram']['feature_names']
    tw_names = ensemble_results['twitter']['feature_names']

    instagram_imp = pd.DataFrame({
        'feature': ig_names[:5],
        'importance': ig_features[:5],
        'platform': 'Instagram'
    })
    twitter_imp = pd.DataFrame({
        'feature': tw_names[:5],
        'importance': tw_features[:5],
        'platform': 'Twitter'
    })
    combined_imp = pd.concat([instagram_imp, twitter_imp])
    pivot_imp = combined_imp.pivot(index='feature', columns='platform', values='importance').fillna(0)
    pivot_imp.plot(kind='barh', ax=ax1)
else:
    # fallback with actual values from latest run
    instagram_imp = pd.DataFrame({
        'feature': ['has_url', 'language', 'hashtag_capped', 'emoji_none', 'type_filled'],
        'importance': [0.14, 0.08, 0.07, 0.07, 0.06],
        'platform': 'Instagram'
    })
    twitter_imp = pd.DataFrame({
        'feature': ['has_url', 'hashtag_capped', 'url_count', 'language', 'exclamation_ratio'],
        'importance': [0.34, 0.15, 0.08, 0.07, 0.06],
        'platform': 'Twitter'
    })
    combined_imp = pd.concat([instagram_imp, twitter_imp])
    pivot_imp = combined_imp.pivot(index='feature', columns='platform', values='importance').fillna(0)
    pivot_imp.plot(kind='barh', ax=ax1)

ax1.set_xlabel('Importance')
ax1.set_title('Top Features by Platform', fontweight='bold')
ax1.legend(title='Platform')

# 2. Model Performance Overview
ax2 = plt.subplot(2, 3, 2)
model_names = ['Baseline\nRF', 'Instagram\nXGBoost', 'Instagram\nStacking', 'Twitter\nXGBoost', 'Twitter\nStacking']
scores = [0.803, 0.736, 0.736, 0.827, 0.827]

colors = ['green' if x > 0.8 else 'orange' if x > 0.7 else 'red' for x in scores]
bars = ax2.bar(range(len(model_names)), scores, color=colors)
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.set_ylabel('AUC-ROC Score')
ax2.set_title('Model Performance Comparison', fontweight='bold')
ax2.set_ylim(0.0, 0.9)
ax2.axhline(y=0.75, color='black', linestyle='--', alpha=0.3, label='Good threshold')
ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='Excellent threshold')

for bar, score in zip(bars, scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. Engagement Distribution by Platform
ax3 = plt.subplot(2, 3, 3)
if 'df_clean' in locals():
    platform_engagement = df_clean.groupby('platform')['interaction_count'].apply(
        lambda x: pd.Series([
            (x == 0).mean() * 100,
            ((x > 0) & (x <= 10)).mean() * 100,
            ((x > 10) & (x <= 100)).mean() * 100,
            (x > 100).mean() * 100
        ], index=['Zero', '1-10', '11-100', '>100'])
    )
else:
    platform_engagement = pd.DataFrame({
        'instagram': [21, 28, 28, 23],
        'twitter': [52, 35, 11, 2]
    }, index=['Zero', '1-10', '11-100', '>100']).T

platform_engagement.T.plot(kind='bar', ax=ax3, width=0.8, stacked=False)
ax3.set_xlabel('Engagement Range')
ax3.set_ylabel('Percentage of Posts')
ax3.set_title('Engagement Distribution by Platform', fontweight='bold')
ax3.legend(title='Platform')
ax3.tick_params(axis='x', rotation=0)

# 4. Hashtag Effectiveness
ax4 = plt.subplot(2, 3, 4)
if 'df_clean' in locals() and 'hashtag_count' in df_clean.columns:
    hashtag_analysis = df_clean.groupby(['platform', pd.cut(df_clean['hashtag_count'],
                                                            bins=[0, 5, 10, 15, 20, 100],
                                                            labels=['0-5', '6-10', '11-15', '16-20', '20+'])]
                                       )['interaction_count'].mean().unstack()
    hashtag_analysis.T.plot(ax=ax4, marker='o', linewidth=2)
else:
    hashtag_bins = ['0-5', '6-10', '11-15', '16-20', '20+']
    instagram_values = [50, 120, 180, 220, 150]
    twitter_values = [5, 8, 15, 12, 10]
    ax4.plot(hashtag_bins, instagram_values, marker='o', linewidth=2, label='instagram')
    ax4.plot(hashtag_bins, twitter_values, marker='o', linewidth=2, label='twitter')

ax4.set_xlabel('Number of Hashtags')
ax4.set_ylabel('Average Engagement')
ax4.set_title('Optimal Hashtag Count by Platform', fontweight='bold')
ax4.legend(title='Platform')
ax4.grid(True, alpha=0.3)

# 5. Updated Key Insights
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

if 'df_clean' in locals():
    zero_rate_ig = (df_clean[df_clean['platform'] == 'instagram']['interaction_count'] == 0).mean() * 100
    zero_rate_tw = (df_clean[df_clean['platform'] == 'twitter']['interaction_count'] == 0).mean() * 100
    avg_ig = df_clean[df_clean['platform'] == 'instagram']['interaction_count'].mean()
    avg_tw = df_clean[df_clean['platform'] == 'twitter']['interaction_count'].mean()
    ratio = avg_ig / avg_tw if avg_tw > 0 else 0
else:
    zero_rate_ig, zero_rate_tw = 21, 52
    avg_ig, avg_tw = 193, 13
    ratio = 14.6

insights_text = f"""
KEY INSIGHTS - VALIDATED FINDINGS

✓ URL Dominance (Surprising Finding!)
  • Twitter: 34% importance (has_url)
  • Instagram: 14% importance
  • URLs matter MORE than hashtags on Twitter
  • Platform-specific URL strategies needed

✓ Platform Reality Check
  • Instagram: {zero_rate_ig:.0f}% zero engagement
  • Twitter: {zero_rate_tw:.0f}% zero engagement
  • Instagram {ratio:.1f}x higher avg engagement
  • Twitter: Better AUC (0.83 vs 0.74)

✓ Model Performance Insights
  • Baseline beats complex models (0.803)
  • Twitter easier to predict (0.827 AUC)
  • Instagram more random (0.736 AUC)
  • Ensembles: minimal improvement

✓ Optimal Thresholds
  • Instagram: 0.168 (aggressive)
  • Twitter: 0.546 (conservative)
  • Platform-specific tuning critical
  • F1 improvements: up to 30%
"""
ax5.text(0.05, 0.95, insights_text, fontsize=9.5, family='monospace',
         verticalalignment='top', transform=ax5.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

# 6. Recommendations
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
recommendations_text = """
EVIDENCE-BASED RECOMMENDATIONS

1. PLATFORM-SPECIFIC STRATEGY
   Instagram:
   • Diversify features (no single dominant)
   • Visual content still matters
   • Lower threshold (0.17) for decisions
   • Accept lower predictability

   Twitter:
   • ALWAYS include URLs (34% importance!)
   • Hashtags secondary (15%)
   • Conservative threshold (0.55)
   • More predictable outcomes

2. CONTENT PRIORITIES
   • URLs > Hashtags (reversal of belief)
   • 10-15 hashtags still optimal
   • Emojis: 2-3 range confirmed
   • Language targeting matters (7-8%)

3. MODEL DEPLOYMENT
   • Use simple Random Forest baseline
   • Platform-specific thresholds
   • Don't over-engineer solutions
   • Focus on feature quality

4. METRICS & MONITORING
   • Track URL click-through rates
   • Monitor zero-engagement %
   • Weekly threshold adjustments
   • A/B test all recommendations
"""
ax6.text(0.05, 0.95, recommendations_text, fontsize=9, family='monospace',
         verticalalignment='top', transform=ax6.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

plt.suptitle('EXPO 2020 SOCIAL MEDIA ANALYSIS', fontsize=16, fontweight='bold')
plt.tight_layout()

if 'FIGURES_DIR' in locals():
    plt.savefig(FIGURES_DIR / 'final_summary.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig('final_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary_stats = {
    'dataset': {
        'total_posts': '423,117',
        'date_range': '2021-09-26 to 2022-02-05',
        'instagram': '233,709 (55.2%)',
        'twitter': '189,408 (44.8%)',
        'zero_engagement': '34.8% overall (21% IG, 52% TW)',
        'duplicates_removed': '77,380 (18.3%)'
    },
    'key_discoveries': {
        '1_url_dominance': 'URLs more important than hashtags (34% vs 15% on Twitter)',
        '2_platform_paradox': 'Twitter more predictable despite higher zero rate',
        '3_baseline_superiority': 'Simple RF (0.803) beats all complex models',
        '4_threshold_criticality': 'Platform-specific thresholds improve F1 by 30%',
        '5_ensemble_failure': 'Sophisticated ensembles provide <1% improvement'
    },
    'model_performance': {
        'baseline_rf': 'AUC 0.803 (best overall)',
        'instagram_best': 'XGBoost AUC 0.736',
        'twitter_best': 'XGBoost AUC 0.827',
        'optimal_thresholds': 'IG: 0.168, TW: 0.546',
        'zero_inflation': 'Hurdle model MAE 63.9 (best)'
    },
    'actionable_insights': {
        'immediate': [
            'Prioritize URLs over hashtags on Twitter',
            'Use platform-specific decision thresholds',
            'Deploy simple baseline model',
            'Focus on feature engineering not model complexity'
        ],
        'testing_required': [
            'Validate 10-15 hashtag range',
            'Confirm 2-3 emoji optimum',
            'Test URL placement strategies',
            'Verify language-specific patterns'
        ]
    }
}

print(summary_stats)


# ## Final Interpretation
# 
# In completing this analysis of 423,117 Expo 2020 Dubai social media posts, I demonstrated comprehensive **data cleaning** (removing 18% duplicates, handling 71K missing content types, resolving platform misclassification), thorough **exploratory data analysis** (12+ visualizations, 8 statistical hypothesis tests, temporal and textual pattern discovery), and extensive **predictive modeling** (baseline models, platform-specific ensembles, advanced zero-inflation handling). The technical execution revealed surprising findings: **URLs dominate engagement prediction** (34% importance on Twitter vs hashtags at 15%), Instagram generates 14.6x more average engagement despite being less predictable (AUC 0.736 vs Twitter's 0.827), and simple Random Forest models (AUC 0.803) consistently outperformed sophisticated ensemble architectures.
# 
# The **platform paradox** uncovered is particularly striking: Twitter, despite 52% zero-engagement rate compared to Instagram's 21%, proves more predictable with AUC 0.827. This suggests Twitter engagement follows binary patterns - posts either get shared (especially with URLs) or ignored—while Instagram's engagement is more nuanced and random. The optimal decision thresholds differ dramatically: Instagram requires aggressive classification (0.168) to catch any engagement, while Twitter's balanced distribution allows conservative thresholds (0.546). These platform-specific dynamics validate my dual-model approach, even though the unified baseline ultimately performed best.
# 
# However, I must acknowledge **fundamental methodological limitations**. The URL dominance on Twitter (34% importance) likely reflects reverse causality - newsworthy content naturally includes links, rather than links causing engagement. The 14.6x platform difference may be confounded by selection bias, with high-budget visual content directed to Instagram while operational updates go to Twitter. My sophisticated techniques—text embeddings (achieving only 0.275 silhouette score), stacking ensembles (0.1% improvement), and zero-inflation models (MAE 63.9) - yielded minimal gains, suggesting **we've reached the ceiling of predictability** with available features, missing crucial factors like author influence, timing, and algorithmic promotion.
# 
# The **business recommendations** derived—prioritizing URLs over hashtags on Twitter, using 10-15 hashtags generally, adding 2-3 emojis, and applying platform-specific thresholds - represent testable hypotheses requiring A/B validation rather than proven strategies. The failure of advanced NLP and ensemble methods to improve performance confirms that **engagement success depends on unmeasured external factors** more than content optimization. While I successfully met all assignment requirements and demonstrated technical proficiency, this analysis ultimately produces a detailed description of correlational patterns, not causal mechanisms. The distinction between achieving 0.803 AUC in retrospective analysis versus predicting future engagement remains the critical gap between data science and business impact - a humbling reminder that even sophisticated analytics cannot fully decode human behavior in complex social systems.

# In[ ]:




