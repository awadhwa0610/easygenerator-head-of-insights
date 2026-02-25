#!/usr/bin/env python3
"""
Easygenerator Head of Insights Case Study — Part 1: Exploratory Data Analysis
Comprehensive analysis of authors.csv and events data.
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import json

print("=" * 70)
print("EASYGENERATOR — HEAD OF INSIGHTS CASE STUDY")
print("PART 1: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ─── 1. LOAD AUTHORS DATA ───────────────────────────────────────────────
print("\n\n📊 SECTION 1: AUTHORS (USER-LEVEL) DATA PROFILING")
print("─" * 60)

authors = pd.read_csv('authors.csv')
print(f"Total authors/users: {len(authors):,}")
print(f"Columns: {list(authors.columns)}")
print(f"\nPlan distribution:")
plan_dist = authors['PLAN'].value_counts()
for plan, count in plan_dist.items():
    pct = count / len(authors) * 100
    print(f"  {plan:15s}: {count:>7,} ({pct:.1f}%)")

# Paying vs Free
paying = authors[authors['PLAN'].isin(['Enterprise', 'Team', 'Pro'])].shape[0]
free_trial = authors[authors['PLAN'].isin(['Free', 'Trial', 'Academy'])].shape[0]
print(f"\n📌 Paying users: {paying:,} ({paying/len(authors)*100:.1f}%)")
print(f"📌 Free/Trial users: {free_trial:,} ({free_trial/len(authors)*100:.1f}%)")

# Country distribution
print(f"\nTop 15 Countries:")
country_dist = authors['COUNTRY'].fillna('Unknown').value_counts().head(15)
for country, count in country_dist.items():
    pct = count / len(authors) * 100
    label = country if country else "Unknown/Empty"
    print(f"  {label:25s}: {count:>6,} ({pct:.1f}%)")

# Organization analysis
org_counts = authors['FKID_ORGANIZATION'].fillna('NO_ORG').value_counts()
users_with_org = authors['FKID_ORGANIZATION'].notna().sum()
users_no_org = authors['FKID_ORGANIZATION'].isna().sum()
unique_orgs = authors['FKID_ORGANIZATION'].nunique()
print(f"\nOrganization Analysis:")
print(f"  Users with organization: {users_with_org:,} ({users_with_org/len(authors)*100:.1f}%)")
print(f"  Users without org (individual): {users_no_org:,} ({users_no_org/len(authors)*100:.1f}%)")
print(f"  Unique organizations: {unique_orgs:,}")

# Registration analysis
authors['CREATED_ON'] = pd.to_datetime(authors['CREATED_ON'], errors='coerce')
print(f"\nUser Creation Timeline:")
print(f"  Earliest: {authors['CREATED_ON'].min()}")
print(f"  Latest: {authors['CREATED_ON'].max()}")

# Cohort analysis by year
authors['created_year'] = authors['CREATED_ON'].dt.year
year_dist = authors['created_year'].value_counts().sort_index()
print(f"\nUsers Created by Year:")
for year, count in year_dist.items():
    if pd.notna(year):
        print(f"  {int(year)}: {count:>6,}")

# Plan by year of creation
print(f"\nPlan Distribution by Creation Year (recent years):")
recent = authors[authors['created_year'] >= 2023]
plan_by_year = pd.crosstab(recent['created_year'], recent['PLAN'], margins=True)
print(plan_by_year.to_string())

# ─── 2. LOAD EVENTS DATA (SAMPLING STRATEGY) ────────────────────────
print("\n\n📊 SECTION 2: EVENTS DATA — FULL ANALYSIS")
print("─" * 60)

# Load ALL events files
events_dir = 'mightymerge.io__6n3xp3sm'
event_files = sorted(glob.glob(os.path.join(events_dir, 'events_*.csv')))
print(f"Loading {len(event_files)} event files...")

dfs = []
for i, f in enumerate(event_files):
    df = pd.read_csv(f, low_memory=False)
    if i > 0:
        # Skip header rows from subsequent files
        df = df[df['TIMESTAMP'] != 'TIMESTAMP']
    dfs.append(df)
    if (i + 1) % 50 == 0:
        print(f"  Loaded {i+1}/{len(event_files)} files...")

events = pd.concat(dfs, ignore_index=True)
events = events[events['TIMESTAMP'] != 'TIMESTAMP']  # Remove any remaining header rows
print(f"Total events loaded: {len(events):,}")

events['TIMESTAMP'] = pd.to_datetime(events['TIMESTAMP'], errors='coerce')
events['date'] = events['TIMESTAMP'].dt.date
events['hour'] = events['TIMESTAMP'].dt.hour
events['day_of_week'] = events['TIMESTAMP'].dt.dayofweek  # 0=Mon
events['week'] = events['TIMESTAMP'].dt.isocalendar().week.astype(int)

print(f"Date range: {events['TIMESTAMP'].min()} to {events['TIMESTAMP'].max()}")
print(f"Unique users in events: {events['FKID_USER_REPLACED'].nunique():,}")
print(f"Unique courses in events: {events['FKID_COURSE'].nunique():,}")
print(f"Unique event types: {events['EVENT'].nunique():,}")
print(f"Unique categories: {events['CATEGORY'].nunique():,}")

# ─── 3. EVENT TAXONOMY ──────────────────────────────────────────────────
print("\n\n📊 SECTION 3: EVENT TAXONOMY")
print("─" * 60)

print("\nTop 40 Events by Volume:")
event_counts = events['EVENT'].value_counts()
for event, count in event_counts.head(40).items():
    pct = count / len(events) * 100
    print(f"  {event:55s}: {count:>8,} ({pct:.1f}%)")

print(f"\nCategory Distribution:")
cat_counts = events['CATEGORY'].fillna('Uncategorized').value_counts()
for cat, count in cat_counts.items():
    pct = count / len(events) * 100
    print(f"  {cat:40s}: {count:>8,} ({pct:.1f}%)")

# ─── 4. ENGAGEMENT ANALYSIS ─────────────────────────────────────────────
print("\n\n📊 SECTION 4: ENGAGEMENT ANALYSIS")
print("─" * 60)

# Merge events with authors
events_with_plan = events.merge(
    authors[['PKID_USER_REPLACED', 'PLAN', 'COUNTRY', 'FKID_ORGANIZATION', 'CREATED_ON']],
    left_on='FKID_USER_REPLACED',
    right_on='PKID_USER_REPLACED',
    how='left'
)

# Events per user
events_per_user = events.groupby('FKID_USER_REPLACED').size()
print(f"\nEvents per User Distribution:")
print(f"  Mean: {events_per_user.mean():.1f}")
print(f"  Median: {events_per_user.median():.1f}")
print(f"  P25: {events_per_user.quantile(0.25):.0f}")
print(f"  P75: {events_per_user.quantile(0.75):.0f}")
print(f"  P95: {events_per_user.quantile(0.95):.0f}")
print(f"  P99: {events_per_user.quantile(0.99):.0f}")
print(f"  Max: {events_per_user.max():,}")

# Engagement by Plan
print(f"\nEngagement by Plan:")
plan_engagement = events_with_plan.groupby('PLAN').agg(
    total_events=('EVENT', 'count'),
    unique_users=('FKID_USER_REPLACED', 'nunique'),
    unique_courses=('FKID_COURSE', 'nunique')
).reset_index()
plan_engagement['events_per_user'] = plan_engagement['total_events'] / plan_engagement['unique_users']
plan_engagement['courses_per_user'] = plan_engagement['unique_courses'] / plan_engagement['unique_users']
print(plan_engagement.to_string(index=False))

# ─── 5. DAILY ACTIVITY PATTERNS ─────────────────────────────────────────
print("\n\n📊 SECTION 5: DAILY ACTIVITY PATTERNS")
print("─" * 60)

daily_events = events.groupby('date').size()
print(f"\nDaily Events:")
print(f"  Mean: {daily_events.mean():,.0f}")
print(f"  Min: {daily_events.min():,} (date: {daily_events.idxmin()})")
print(f"  Max: {daily_events.max():,} (date: {daily_events.idxmax()})")

# Day of week
dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
dow_events = events.groupby('day_of_week').size()
print(f"\nEvents by Day of Week:")
for dow, count in dow_events.items():
    pct = count / len(events) * 100
    print(f"  {dow_names.get(dow, 'Unknown'):12s}: {count:>8,} ({pct:.1f}%)")

# Hour of day
print(f"\nEvents by Hour (UTC):")
hourly = events.groupby('hour').size()
for h, count in hourly.items():
    pct = count / len(events) * 100
    bar = '█' * int(pct * 3)
    print(f"  {h:02d}:00  {count:>7,} ({pct:.1f}%) {bar}")

# ─── 6. USER SEGMENTATION ───────────────────────────────────────────────
print("\n\n📊 SECTION 6: USER SEGMENTATION & ACTIVATION")
print("─" * 60)

# Active days per user in May
active_days = events.groupby('FKID_USER_REPLACED')['date'].nunique()
print(f"\nActive Days per User (May 2025):")
print(f"  Mean: {active_days.mean():.1f}")
print(f"  Median: {active_days.median():.0f}")

# User segmentation
def segment_user(days):
    if days >= 20: return 'Power User (20+ days)'
    elif days >= 10: return 'Regular (10-19 days)'
    elif days >= 4: return 'Casual (4-9 days)'
    else: return 'Light (1-3 days)'

user_segments = active_days.apply(segment_user).value_counts()
print(f"\nUser Segments by Activity Level:")
for seg, count in user_segments.items():
    pct = count / len(active_days) * 100
    print(f"  {seg:30s}: {count:>5,} ({pct:.1f}%)")

# Users in events but NOT in authors (potential data quality check)
event_users = set(events['FKID_USER_REPLACED'].unique())
author_users = set(authors['PKID_USER_REPLACED'].unique())
only_in_events = event_users - author_users
only_in_authors = author_users - event_users
print(f"\nData Quality:")
print(f"  Users in events but NOT in authors: {len(only_in_events):,}")
print(f"  Authors with NO events in May: {len(only_in_authors):,}")
print(f"  Overlap: {len(event_users & author_users):,}")

# ─── 7. FEATURE ADOPTION ANALYSIS ───────────────────────────────────────
print("\n\n📊 SECTION 7: FEATURE ADOPTION ANALYSIS")
print("─" * 60)

# EasyAI features
ai_events = events[events['CATEGORY'].fillna('').str.contains('EasyAI|Ask AI|Image generation', case=False) | 
                    events['EVENT'].fillna('').str.contains('Generate TTS|Text to Speech|Quick Actions.*EasyAI|AI', case=False)]
ai_users = ai_events['FKID_USER_REPLACED'].nunique()
total_active = events['FKID_USER_REPLACED'].nunique()
print(f"\n🤖 EasyAI Adoption:")
print(f"  AI-related events: {len(ai_events):,}")
print(f"  Users using AI features: {ai_users:,} ({ai_users/total_active*100:.1f}% of active users)")

# AI by plan
ai_with_plan = ai_events.merge(
    authors[['PKID_USER_REPLACED', 'PLAN']],
    left_on='FKID_USER_REPLACED', right_on='PKID_USER_REPLACED', how='left'
)
print(f"  AI usage by plan:")
ai_plan = ai_with_plan.groupby('PLAN').agg(
    ai_events=('EVENT', 'count'),
    ai_users=('FKID_USER_REPLACED', 'nunique')
).reset_index()
for _, row in ai_plan.iterrows():
    print(f"    {row['PLAN']:15s}: {row['ai_events']:>6,} events, {row['ai_users']:>5,} users")

# SCORM / Publishing
scorm_events = events[events['EVENT'].fillna('').str.contains('SCORM|scorm', case=False)]
print(f"\n📦 SCORM Publishing:")
print(f"  SCORM-related events: {len(scorm_events):,}")
print(f"  Users using SCORM: {scorm_events['FKID_USER_REPLACED'].nunique():,}")
scorm_breakdown = scorm_events['EVENT'].value_counts()
for event, count in scorm_breakdown.items():
    print(f"    {event}: {count:,}")

# Collaboration features
collab_events = events[events['EVENT'].fillna('').str.contains('co-author|comment|Collaborator|review', case=False)]
print(f"\n🤝 Collaboration Features:")
print(f"  Collaboration events: {len(collab_events):,}")
print(f"  Users using collaboration: {collab_events['FKID_USER_REPLACED'].nunique():,}")

# Publishing
publish_events = events[events['CATEGORY'].fillna('') == 'Publish']
print(f"\n🚀 Publishing Activity:")
print(f"  Publish events: {len(publish_events):,}")
print(f"  Users who published: {publish_events['FKID_USER_REPLACED'].nunique():,}")
pub_breakdown = publish_events['EVENT'].value_counts().head(10)
for event, count in pub_breakdown.items():
    print(f"    {event}: {count:,}")

# Course creation
create_events = events[events['EVENT'].fillna('').str.contains('course created|Create course', case=False)]
print(f"\n📝 Course Creation:")
print(f"  Course creation events: {len(create_events):,}")
print(f"  Users who created courses: {create_events['FKID_USER_REPLACED'].nunique():,}")

# ─── 8. CONTENT CREATION DEPTH ──────────────────────────────────────────
print("\n\n📊 SECTION 8: CONTENT CREATION DEPTH")
print("─" * 60)

# Content block types
print(f"\nContent Block Types Used:")
cbt = events['CONTENT_BLOCK_TYPE'].dropna().value_counts().head(20)
for block, count in cbt.items():
    print(f"  {block:40s}: {count:>7,}")

# Courses per user
courses_per_user = events.groupby('FKID_USER_REPLACED')['FKID_COURSE'].nunique()
print(f"\nCourses per User:")
print(f"  Mean: {courses_per_user.mean():.1f}")
print(f"  Median: {courses_per_user.median():.0f}")
print(f"  P75: {courses_per_user.quantile(0.75):.0f}")
print(f"  P95: {courses_per_user.quantile(0.95):.0f}")
print(f"  Max: {courses_per_user.max()}")

# ─── 9. RETENTION & STICKINESS ──────────────────────────────────────────
print("\n\n📊 SECTION 9: RETENTION & STICKINESS")
print("─" * 60)

# Weekly retention
weekly_users = events.groupby('week')['FKID_USER_REPLACED'].nunique()
print(f"\nWeekly Active Users:")
for week, count in weekly_users.items():
    print(f"  Week {week}: {count:,}")

# DAU / WAU / MAU
dau = events.groupby('date')['FKID_USER_REPLACED'].nunique()
mau = events['FKID_USER_REPLACED'].nunique()
print(f"\nStickiness Metrics (May 2025):")
print(f"  Average DAU: {dau.mean():,.0f}")
print(f"  MAU: {mau:,}")
print(f"  DAU/MAU Ratio: {dau.mean()/mau:.2%}")

# Week-over-week retention
weeks = sorted(events['week'].unique())
if len(weeks) >= 2:
    print(f"\nWeek-over-week Retention:")
    for i in range(1, len(weeks)):
        prev_users = set(events[events['week'] == weeks[i-1]]['FKID_USER_REPLACED'].unique())
        curr_users = set(events[events['week'] == weeks[i]]['FKID_USER_REPLACED'].unique())
        retained = prev_users & curr_users
        if len(prev_users) > 0:
            ret_rate = len(retained) / len(prev_users) * 100
            print(f"  Week {weeks[i-1]} → {weeks[i]}: {len(retained):,}/{len(prev_users):,} = {ret_rate:.1f}%")

# ─── 10. GEOGRAPHY & PLAN DEEP DIVE ─────────────────────────────────────
print("\n\n📊 SECTION 10: GEOGRAPHY & PLAN DEEP DIVE")
print("─" * 60)

# Activity intensity by country (events per user)
country_engagement = events_with_plan.groupby('COUNTRY').agg(
    total_events=('EVENT', 'count'),
    unique_users=('FKID_USER_REPLACED', 'nunique')
).reset_index()
country_engagement['events_per_user'] = country_engagement['total_events'] / country_engagement['unique_users']
country_engagement = country_engagement[country_engagement['unique_users'] >= 50].sort_values('events_per_user', ascending=False)
print(f"\nTop 15 Countries by Events-per-User (min 50 users):")
for _, row in country_engagement.head(15).iterrows():
    c = row['COUNTRY'] if pd.notna(row['COUNTRY']) else 'Unknown'
    print(f"  {c:25s}: {row['events_per_user']:>7.1f} events/user ({int(row['unique_users']):,} users)")

# Translation / language events
translation_events = events[events['EVENT'].fillna('').str.contains('translat|language|XLIFF', case=False)]
print(f"\n🌐 Translation/Language Events:")
print(f"  Total: {len(translation_events):,}")
print(f"  Users: {translation_events['FKID_USER_REPLACED'].nunique():,}")
if len(translation_events) > 0:
    trans_breakdown = translation_events['EVENT'].value_counts()
    for event, count in trans_breakdown.items():
        print(f"    {event}: {count:,}")

# ─── 11. SIGN-IN / SESSION ANALYSIS ─────────────────────────────────────
print("\n\n📊 SECTION 11: SIGN-IN & SESSION PATTERNS")
print("─" * 60)

signin_events = events[events['EVENT'] == 'Sign in']
print(f"\nSign-in Events: {len(signin_events):,}")
print(f"Users who signed in: {signin_events['FKID_USER_REPLACED'].nunique():,}")

# Sign-ins per user
signins_per_user = signin_events.groupby('FKID_USER_REPLACED').size()
print(f"\nSign-ins per User:")
print(f"  Mean: {signins_per_user.mean():.1f}")
print(f"  Median: {signins_per_user.median():.0f}")
print(f"  Max: {signins_per_user.max()}")

# Method column analysis
print(f"\nEvent Methods:")
method_counts = events['METHOD'].dropna().value_counts().head(15)
for m, c in method_counts.items():
    print(f"  {m:40s}: {c:>7,}")

# ─── 12. ORGANIZATION SIZE ANALYSIS ─────────────────────────────────────
print("\n\n📊 SECTION 12: ORGANIZATION SIZE ANALYSIS")
print("─" * 60)

# Org size distribution
org_sizes = authors.groupby('FKID_ORGANIZATION').size().reset_index(name='org_size')
print(f"\nOrganization Size Distribution:")
print(f"  Mean: {org_sizes['org_size'].mean():.1f}")
print(f"  Median: {org_sizes['org_size'].median():.0f}")
print(f"  P75: {org_sizes['org_size'].quantile(0.75):.0f}")
print(f"  P95: {org_sizes['org_size'].quantile(0.95):.0f}")
print(f"  Max: {org_sizes['org_size'].max()}")

# Large orgs
large_orgs = org_sizes[org_sizes['org_size'] >= 50]
print(f"  Orgs with 50+ users: {len(large_orgs):,}")
very_large = org_sizes[org_sizes['org_size'] >= 200]
print(f"  Orgs with 200+ users: {len(very_large):,}")

# ─── 13. USER JOURNEY / FUNNEL ──────────────────────────────────────────
print("\n\n📊 SECTION 13: USER JOURNEY FUNNEL")
print("─" * 60)

funnel_steps = {
    'Sign in': events[events['EVENT'] == 'Sign in']['FKID_USER_REPLACED'].nunique(),
    'Create course': events[events['EVENT'].fillna('').str.contains('course created|Create course', case=False)]['FKID_USER_REPLACED'].nunique(),
    'Edit content': events[events['EVENT'] == 'Edit content']['FKID_USER_REPLACED'].nunique(),
    'Preview course': events[events['EVENT'] == 'Preview course']['FKID_USER_REPLACED'].nunique(),
    'Publish/Share': events[events['CATEGORY'].fillna('') == 'Publish']['FKID_USER_REPLACED'].nunique(),
    'Download SCORM': events[events['EVENT'].fillna('').str.contains('Download SCORM', case=False)]['FKID_USER_REPLACED'].nunique(),
}

total_users = events['FKID_USER_REPLACED'].nunique()
print(f"\nUser Journey Funnel (Unique Users in May 2025):")
for step, count in funnel_steps.items():
    pct = count / total_users * 100
    bar = '█' * int(pct / 2)
    print(f"  {step:25s}: {count:>5,} ({pct:.1f}%) {bar}")

# ─── 14. KEY INSIGHTS SUMMARY ───────────────────────────────────────────
print("\n\n" + "=" * 70)
print("📌 KEY INSIGHTS SUMMARY")
print("=" * 70)

print(f"""
1. SCALE & REACH
   • {len(authors):,} total authors, {paying:,} paying ({paying/len(authors)*100:.1f}%)
   • {mau:,} Monthly Active Users (MAU) in May 2025
   • {events['FKID_COURSE'].nunique():,} unique courses touched
   • DAU/MAU ratio: {dau.mean()/mau:.2%} — indicates {'strong' if dau.mean()/mau > 0.2 else 'moderate' if dau.mean()/mau > 0.1 else 'low'} engagement

2. PLAN INSIGHTS
   • Enterprise users are {plan_engagement[plan_engagement['PLAN']=='Enterprise'].get('events_per_user', pd.Series([0])).values[0]:.0f} events/user on average
   • Free plan has {plan_dist.get('Free', 0):,} users — massive potential for conversion
   
3. AI FEATURES
   • {ai_users:,} users ({ai_users/total_active*100:.1f}%) adopted AI features
   • Key AI features: TTS generation, EasyAI widget, Image generation

4. PUBLISHING HEALTH
   • {funnel_steps['Publish/Share']:,} users published/shared content
   • {funnel_steps['Download SCORM']:,} users downloaded SCORM packages
   • Publishing is a key conversion indicator from casual to power user

5. COLLABORATION
   • {collab_events['FKID_USER_REPLACED'].nunique():,} users engaged with collaboration features
   • Co-author comments and reviewer workflows are actively used

6. TRANSLATION/MLC RELEVANCE
   • {len(translation_events):,} translation-related events found
   • This baseline data is critical for measuring MLC feature impact
""")

print("\n✅ Analysis complete!")
