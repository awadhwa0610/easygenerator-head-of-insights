#!/usr/bin/env python3
"""
Easygenerator Case Study — Chart Generation
Generates all visualization PNGs for the slide deck.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, glob

# ─── STYLE CONFIG ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f172a',
    'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155',
    'axes.labelcolor': '#e2e8f0',
    'text.color': '#e2e8f0',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'grid.color': '#334155',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'figure.dpi': 200,
})

ACCENT = '#3b82f6'
ACCENT2 = '#8b5cf6'
ACCENT3 = '#06b6d4'
ACCENT4 = '#f59e0b'
ACCENT5 = '#ef4444'
GREEN = '#22c55e'
GRADIENT = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#22c55e']

OUT_DIR = 'charts'
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")

# ─── LOAD DATA ───────────────────────────────────────────────────────────
authors = pd.read_csv('authors.csv')
authors['CREATED_ON'] = pd.to_datetime(authors['CREATED_ON'], errors='coerce')

events_dir = 'mightymerge.io__6n3xp3sm'
event_files = sorted(glob.glob(os.path.join(events_dir, 'events_*.csv')))
dfs = []
for i, f in enumerate(event_files):
    df = pd.read_csv(f, low_memory=False)
    if i > 0:
        df = df[df['TIMESTAMP'] != 'TIMESTAMP']
    dfs.append(df)
    if (i + 1) % 50 == 0:
        print(f"  Loaded {i+1}/{len(event_files)} files...")

events = pd.concat(dfs, ignore_index=True)
events = events[events['TIMESTAMP'] != 'TIMESTAMP']
events['TIMESTAMP'] = pd.to_datetime(events['TIMESTAMP'], errors='coerce')
events['date'] = events['TIMESTAMP'].dt.date
events['hour'] = events['TIMESTAMP'].dt.hour
events['day_of_week'] = events['TIMESTAMP'].dt.dayofweek
events['week'] = events['TIMESTAMP'].dt.isocalendar().week.astype(int)

# Merge
events_merged = events.merge(
    authors[['PKID_USER_REPLACED', 'PLAN', 'COUNTRY', 'FKID_ORGANIZATION']],
    left_on='FKID_USER_REPLACED', right_on='PKID_USER_REPLACED', how='left'
)

print(f"Data loaded: {len(events):,} events, {len(authors):,} authors")
print("Generating charts...\n")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1: Platform Health KPI Card (horizontal bar style)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
mau = events['FKID_USER_REPLACED'].nunique()
dau = events.groupby('date')['FKID_USER_REPLACED'].nunique()
avg_dau = dau.mean()
total_events = len(events)
courses = events['FKID_COURSE'].nunique()
paying = authors[authors['PLAN'].isin(['Enterprise', 'Team', 'Pro'])].shape[0]

metrics = ['Total Authors', 'Paying Users', 'MAU (May)', 'Avg DAU', 'Courses Touched']
values = [len(authors), paying, mau, int(avg_dau), courses]
colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4, GREEN]

bars = ax.barh(metrics[::-1], [v/1000 for v in values[::-1]], color=colors[::-1], height=0.55, edgecolor='none')
for bar, val in zip(bars, values[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:,}',
            va='center', fontsize=14, fontweight='bold', color='white')
ax.set_xlabel('(thousands)')
ax.set_title('Platform Scale — May 2025', pad=15)
ax.set_xlim(0, max(values)/1000 * 1.25)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_platform_health.png', bbox_inches='tight')
plt.close()
print("✅ 01_platform_health.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2: User Segmentation (Donut)
# ═══════════════════════════════════════════════════════════════════════════
active_days = events.groupby('FKID_USER_REPLACED')['date'].nunique()
def segment(d):
    if d >= 20: return 'Power (20+ days)'
    elif d >= 10: return 'Regular (10-19)'
    elif d >= 4: return 'Casual (4-9)'
    else: return 'Light (1-3)'
segs = active_days.apply(segment).value_counts()
order = ['Light (1-3)', 'Casual (4-9)', 'Regular (10-19)', 'Power (20+ days)']
segs = segs.reindex(order)

fig, ax = plt.subplots(figsize=(8, 8))
colors_donut = [ACCENT5, ACCENT4, ACCENT3, GREEN]
wedges, texts, autotexts = ax.pie(segs.values, labels=segs.index, autopct='%1.1f%%',
    colors=colors_donut, startangle=90, pctdistance=0.8,
    wedgeprops=dict(width=0.4, edgecolor='#0f172a', linewidth=2))
for t in autotexts:
    t.set_fontsize(13)
    t.set_fontweight('bold')
for t in texts:
    t.set_fontsize(12)
ax.set_title('User Activity Segmentation\n(Active Days in May 2025)', fontsize=16, fontweight='bold', pad=20)

# Center text
ax.text(0, 0, f'{mau:,}\nMAU', ha='center', va='center', fontsize=20, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_user_segmentation.png', bbox_inches='tight')
plt.close()
print("✅ 02_user_segmentation.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 3: Plan-Level Engagement (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════
plan_eng = events_merged.groupby('PLAN').agg(
    total_events=('EVENT', 'count'),
    unique_users=('FKID_USER_REPLACED', 'nunique'),
    unique_courses=('FKID_COURSE', 'nunique')
).reset_index()
plan_eng['events_per_user'] = plan_eng['total_events'] / plan_eng['unique_users']
plan_eng['courses_per_user'] = plan_eng['unique_courses'] / plan_eng['unique_users']
plan_eng = plan_eng[plan_eng['PLAN'].isin(['Free', 'Trial', 'Team', 'Enterprise', 'Pro'])]
plan_order = ['Free', 'Trial', 'Pro', 'Team', 'Enterprise']
plan_eng['PLAN'] = pd.Categorical(plan_eng['PLAN'], categories=plan_order, ordered=True)
plan_eng = plan_eng.sort_values('PLAN')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Events per user
bars1 = ax1.bar(plan_eng['PLAN'], plan_eng['events_per_user'],
    color=[ACCENT5, ACCENT4, ACCENT2, ACCENT3, ACCENT], edgecolor='none')
for bar, val in zip(bars1, plan_eng['events_per_user']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:.0f}', ha='center', fontsize=12, fontweight='bold', color='white')
ax1.set_title('Events per User', pad=10)
ax1.set_ylabel('Avg Events / User')
ax1.grid(axis='y', alpha=0.3)

# Courses per user
bars2 = ax2.bar(plan_eng['PLAN'], plan_eng['courses_per_user'],
    color=[ACCENT5, ACCENT4, ACCENT2, ACCENT3, ACCENT], edgecolor='none')
for bar, val in zip(bars2, plan_eng['courses_per_user']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}', ha='center', fontsize=12, fontweight='bold', color='white')
ax2.set_title('Courses per User', pad=10)
ax2.set_ylabel('Avg Courses / User')
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('Engagement by Plan Tier', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_plan_engagement.png', bbox_inches='tight')
plt.close()
print("✅ 03_plan_engagement.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 4: User Journey Funnel
# ═══════════════════════════════════════════════════════════════════════════
total_active = events['FKID_USER_REPLACED'].nunique()
funnel = {
    'Sign In': events[events['EVENT'] == 'Sign in']['FKID_USER_REPLACED'].nunique(),
    'Edit Content': events[events['EVENT'] == 'Edit content']['FKID_USER_REPLACED'].nunique(),
    'Preview': events[events['EVENT'] == 'Preview course']['FKID_USER_REPLACED'].nunique(),
    'Create Course': events[events['EVENT'].fillna('').str.contains('course created|Create course', case=False)]['FKID_USER_REPLACED'].nunique(),
    'Publish/Share': events[events['CATEGORY'].fillna('') == 'Publish']['FKID_USER_REPLACED'].nunique(),
    'SCORM Download': events[events['EVENT'].fillna('').str.contains('Download SCORM', case=False)]['FKID_USER_REPLACED'].nunique(),
}

fig, ax = plt.subplots(figsize=(10, 6))
steps = list(funnel.keys())
vals = list(funnel.values())
pcts = [v/total_active*100 for v in vals]

# Funnel as horizontal bars with gradient
colors_funnel = [ACCENT, '#4f8ff7', ACCENT3, '#34d399', ACCENT4, ACCENT5]
bars = ax.barh(steps[::-1], vals[::-1], color=colors_funnel[::-1], height=0.6, edgecolor='none')
for bar, val, pct in zip(bars, vals[::-1], pcts[::-1]):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f'{val:,} ({pct:.0f}%)', va='center', fontsize=12, fontweight='bold', color='white')

ax.set_title('User Journey Funnel — May 2025', pad=15)
ax.set_xlabel('Unique Users')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(vals) * 1.35)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_user_funnel.png', bbox_inches='tight')
plt.close()
print("✅ 04_user_funnel.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 5: Feature Adoption Rates
# ═══════════════════════════════════════════════════════════════════════════
ai_users = events[events['CATEGORY'].fillna('').str.contains('EasyAI|Ask AI|Image generation', case=False) |
                  events['EVENT'].fillna('').str.contains('Generate TTS|Text to Speech|Quick Actions.*EasyAI|AI', case=False)]['FKID_USER_REPLACED'].nunique()
collab_users = events[events['EVENT'].fillna('').str.contains('co-author|comment|Collaborator|review', case=False)]['FKID_USER_REPLACED'].nunique()
scorm_users = events[events['EVENT'].fillna('').str.contains('SCORM|scorm', case=False)]['FKID_USER_REPLACED'].nunique()
publish_users = events[events['CATEGORY'].fillna('') == 'Publish']['FKID_USER_REPLACED'].nunique()
trans_users = events[events['EVENT'].fillna('').str.contains('translat|language|XLIFF', case=False)]['FKID_USER_REPLACED'].nunique()

features = ['Collaboration', 'Publishing', 'EasyAI', 'SCORM', 'Translation']
f_vals = [collab_users, publish_users, ai_users, scorm_users, trans_users]
f_pcts = [v/total_active*100 for v in f_vals]

fig, ax = plt.subplots(figsize=(10, 5.5))
colors_feat = [ACCENT3, GREEN, ACCENT2, ACCENT4, ACCENT]
bars = ax.barh(features[::-1], f_pcts[::-1], color=colors_feat[::-1], height=0.55, edgecolor='none')
for bar, pct, val in zip(bars, f_pcts[::-1], f_vals[::-1]):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}% ({val:,} users)', va='center', fontsize=12, fontweight='bold', color='white')
ax.set_title('Feature Adoption Rates — % of MAU', pad=15)
ax.set_xlabel('% of Active Users')
ax.set_xlim(0, 75)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_feature_adoption.png', bbox_inches='tight')
plt.close()
print("✅ 05_feature_adoption.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 6: Daily Events Time Series
# ═══════════════════════════════════════════════════════════════════════════
daily = events.groupby('date').agg(
    events=('EVENT', 'count'),
    users=('FKID_USER_REPLACED', 'nunique')
).reset_index()
daily['date'] = pd.to_datetime(daily['date'])

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.fill_between(daily['date'], daily['events'], alpha=0.3, color=ACCENT)
ax1.plot(daily['date'], daily['events'], color=ACCENT, linewidth=2, label='Events')
ax1.set_ylabel('Daily Events', color=ACCENT)
ax1.set_xlabel('')
ax1.tick_params(axis='y', labelcolor=ACCENT)

ax2 = ax1.twinx()
ax2.plot(daily['date'], daily['users'], color=ACCENT4, linewidth=2, linestyle='--', label='DAU')
ax2.set_ylabel('Daily Active Users', color=ACCENT4)
ax2.tick_params(axis='y', labelcolor=ACCENT4)

ax1.set_title('Daily Activity — May 2025', pad=15)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
           facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
ax1.grid(axis='both', alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_daily_activity.png', bbox_inches='tight')
plt.close()
print("✅ 06_daily_activity.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 7: Day of Week + Hourly Heatmap
# ═══════════════════════════════════════════════════════════════════════════
dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.8]})

# Day of week bar
dow = events.groupby('day_of_week').size()
dow_colors = [ACCENT if d < 5 else ACCENT5 for d in dow.index]
bars = ax1.bar([dow_names[d] for d in dow.index], dow.values / 1000, color=dow_colors, edgecolor='none')
for bar, val in zip(bars, dow.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val/1000:.0f}K', ha='center', fontsize=10, fontweight='bold', color='white')
ax1.set_title('Events by Day of Week', pad=10)
ax1.set_ylabel('Events (thousands)')
ax1.grid(axis='y', alpha=0.3)

# Hourly bar
hourly = events.groupby('hour').size()
hour_colors = [GREEN if 7 <= h <= 16 else ACCENT if 6 <= h <= 17 else '#475569' for h in hourly.index]
ax2.bar([f'{h:02d}' for h in hourly.index], hourly.values / 1000, color=hour_colors, edgecolor='none', width=0.8)
ax2.set_title('Events by Hour (UTC)', pad=10)
ax2.set_ylabel('Events (thousands)')
ax2.set_xlabel('Hour (UTC)')
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/07_temporal_patterns.png', bbox_inches='tight')
plt.close()
print("✅ 07_temporal_patterns.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 8: Weekly Retention
# ═══════════════════════════════════════════════════════════════════════════
weeks = sorted(events['week'].unique())
ret_data = []
for i in range(1, len(weeks)):
    prev = set(events[events['week'] == weeks[i-1]]['FKID_USER_REPLACED'].unique())
    curr = set(events[events['week'] == weeks[i]]['FKID_USER_REPLACED'].unique())
    retained = len(prev & curr)
    ret_rate = retained / len(prev) * 100 if len(prev) > 0 else 0
    ret_data.append({'transition': f'W{weeks[i-1]}→W{weeks[i]}', 'rate': ret_rate, 'retained': retained, 'prev': len(prev)})
ret_df = pd.DataFrame(ret_data)

fig, ax = plt.subplots(figsize=(10, 5))
colors_ret = [GREEN if r > 55 else ACCENT4 if r > 50 else ACCENT5 for r in ret_df['rate']]
bars = ax.bar(ret_df['transition'], ret_df['rate'], color=colors_ret, edgecolor='none', width=0.5)
for bar, row in zip(bars, ret_df.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{row.rate:.1f}%\n({row.retained:,}/{row.prev:,})',
            ha='center', fontsize=11, fontweight='bold', color='white')
ax.set_title('Week-over-Week Retention', pad=15)
ax.set_ylabel('Retention Rate (%)')
ax.set_ylim(0, 75)
ax.axhline(y=50, color=ACCENT5, linestyle='--', alpha=0.5, label='50% threshold')
ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/08_weekly_retention.png', bbox_inches='tight')
plt.close()
print("✅ 08_weekly_retention.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 9: Geographic Engagement
# ═══════════════════════════════════════════════════════════════════════════
geo = events_merged.groupby('COUNTRY').agg(
    events=('EVENT', 'count'), users=('FKID_USER_REPLACED', 'nunique')
).reset_index()
geo['epu'] = geo['events'] / geo['users']
geo = geo[geo['users'] >= 50].sort_values('epu', ascending=True).tail(12)

fig, ax = plt.subplots(figsize=(10, 6))
colors_geo = plt.cm.viridis(np.linspace(0.3, 0.9, len(geo)))
bars = ax.barh(geo['COUNTRY'], geo['epu'], color=colors_geo, height=0.6, edgecolor='none')
for bar, row in zip(bars, geo.itertuples()):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{row.epu:.0f} evt/user ({row.users:,})',
            va='center', fontsize=11, fontweight='bold', color='white')
ax.set_title('Most Engaged Countries\n(Events per User, min 50 users)', pad=15)
ax.set_xlabel('Events per User')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, geo['epu'].max() * 1.35)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/09_geographic_engagement.png', bbox_inches='tight')
plt.close()
print("✅ 09_geographic_engagement.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 10: Content Block Types (Top 10)
# ═══════════════════════════════════════════════════════════════════════════
cbt = events['CONTENT_BLOCK_TYPE'].dropna().value_counts().head(10)
short_names = {
    'textEditorWithoutHeading': 'Text Block',
    "'Image' block": 'Image Block',
    'textEditorOneColumn': 'Text (1-col)',
    'oneImage': 'Single Image',
    'hotspot': 'Hotspot',
    'flipCards': 'Flip Cards',
    'imageInTheLeft': 'Img Left Layout',
    'scenarios': 'Scenarios',
    'singleVideo': 'Video',
    'imageInTheRight': 'Img Right Layout',
}
cbt.index = [short_names.get(x, x) for x in cbt.index]

fig, ax = plt.subplots(figsize=(10, 5.5))
colors_cbt = plt.cm.cool(np.linspace(0.2, 0.8, len(cbt)))
bars = ax.barh(cbt.index[::-1], cbt.values[::-1] / 1000, color=colors_cbt[::-1], height=0.6, edgecolor='none')
for bar, val in zip(bars, cbt.values[::-1]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=11, fontweight='bold', color='white')
ax.set_title('Most Used Content Block Types', pad=15)
ax.set_xlabel('Usage Count (thousands)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/10_content_blocks.png', bbox_inches='tight')
plt.close()
print("✅ 10_content_blocks.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 11: Course Creation Methods
# ═══════════════════════════════════════════════════════════════════════════
methods = events['METHOD'].dropna().value_counts().head(7)
method_names = {
    'Dynamic': 'Dynamic',
    'Describe & Generate': 'AI: Describe & Generate',
    'Course Builder': 'Course Builder',
    'Manual': 'Manual',
    'fromScratch': 'From Scratch',
    'Generate from Content Block': 'AI: From Block',
    'fromPresentation': 'From PPT',
}
methods.index = [method_names.get(x, x) for x in methods.index]

fig, ax = plt.subplots(figsize=(9, 5.5))
mc = [ACCENT2 if 'AI' in m else ACCENT for m in methods.index]
bars = ax.barh(methods.index[::-1], methods.values[::-1], color=mc[::-1], height=0.55, edgecolor='none')
for bar, val in zip(bars, methods.values[::-1]):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=12, fontweight='bold', color='white')
ax.set_title('Course Creation Methods', pad=15)
ax.set_xlabel('Count')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, methods.values[0] * 1.25)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ACCENT2, label='AI-Assisted'), Patch(facecolor=ACCENT, label='Traditional')]
ax.legend(handles=legend_elements, loc='lower right', facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/11_creation_methods.png', bbox_inches='tight')
plt.close()
print("✅ 11_creation_methods.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 12: Translation Baseline (for MLC)
# ═══════════════════════════════════════════════════════════════════════════
trans_events = events[events['EVENT'].fillna('').str.contains('translat|language|XLIFF', case=False)]
trans_breakdown = trans_events['EVENT'].value_counts().head(8)
short_trans = {
    'Translate course (configure tab - Automated translation)': 'Auto-translate (config)',
    'Add languages (Automated translation)': 'Add Languages',
    'Translate course (Automated translation process)': 'Auto-translate (process)',
    'Translate (Automated translation - Course list)': 'Auto-translate (list)',
    'Go to the course list (Automated translation)': 'Go to Course List',
    'select language [AGS]': 'Select Language',
    'Open translated course (Automated translations)': 'Open Translated',
    'Go back to the current course (Auto translation - from the confirmation screen)': 'Back to Course',
}
trans_breakdown.index = [short_trans.get(x, x[:35]) for x in trans_breakdown.index]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(trans_breakdown.index[::-1], trans_breakdown.values[::-1],
    color=plt.cm.Blues(np.linspace(0.4, 0.9, len(trans_breakdown))), height=0.6, edgecolor='none')
for bar, val in zip(bars, trans_breakdown.values[::-1]):
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=11, fontweight='bold', color='white')
ax.set_title(f'Translation Feature Baseline\n{trans_events["FKID_USER_REPLACED"].nunique()} users · {len(trans_events):,} events', pad=15)
ax.set_xlabel('Events')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, trans_breakdown.values[0] * 1.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/12_translation_baseline.png', bbox_inches='tight')
plt.close()
print("✅ 12_translation_baseline.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 13: Plan Distribution (Authors)
# ═══════════════════════════════════════════════════════════════════════════
plan_dist = authors['PLAN'].value_counts()
plan_dist = plan_dist[plan_dist.index.isin(['Free', 'Enterprise', 'Team', 'Trial', 'Pro'])]
plan_order2 = ['Enterprise', 'Free', 'Team', 'Trial', 'Pro']
plan_dist = plan_dist.reindex(plan_order2)

fig, ax = plt.subplots(figsize=(8, 8))
colors_plan = [ACCENT, ACCENT5, ACCENT3, ACCENT4, ACCENT2]
wedges, texts, autotexts = ax.pie(plan_dist.values, labels=plan_dist.index, autopct='%1.1f%%',
    colors=colors_plan, startangle=90, pctdistance=0.8,
    wedgeprops=dict(width=0.4, edgecolor='#0f172a', linewidth=2))
for t in autotexts:
    t.set_fontsize(13)
    t.set_fontweight('bold')
for t in texts:
    t.set_fontsize(12)
ax.text(0, 0, f'{len(authors):,}\nTotal', ha='center', va='center', fontsize=18, fontweight='bold', color='white')
ax.set_title('Author Distribution by Plan', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/13_plan_distribution.png', bbox_inches='tight')
plt.close()
print("✅ 13_plan_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 14: DAU/MAU Stickiness Gauge
# ═══════════════════════════════════════════════════════════════════════════
dau_mau = avg_dau / mau * 100

fig, ax = plt.subplots(figsize=(8, 5))
# Simple bar comparison
categories = ['Easygenerator', 'B2B SaaS\nBenchmark Low', 'B2B SaaS\nBenchmark High']
vals_gauge = [dau_mau, 13, 20]
colors_gauge = [ACCENT4, '#475569', '#475569']
bars = ax.bar(categories, vals_gauge, color=colors_gauge, width=0.5, edgecolor='none')
for bar, val in zip(bars, vals_gauge):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontsize=14, fontweight='bold', color='white')
ax.set_title('DAU/MAU Stickiness Ratio', pad=15)
ax.set_ylabel('DAU/MAU %')
ax.set_ylim(0, 25)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/14_stickiness.png', bbox_inches='tight')
plt.close()
print("✅ 14_stickiness.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 15: Active vs Dormant Users
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
dormant = len(authors) - mau
ax.bar(['Active (MAU)', 'Dormant'], [mau, dormant],
    color=[GREEN, ACCENT5], width=0.4, edgecolor='none')
ax.text(0, mau + 500, f'{mau:,}\n({mau/len(authors)*100:.1f}%)',
        ha='center', fontsize=14, fontweight='bold', color=GREEN)
ax.text(1, dormant + 500, f'{dormant:,}\n({dormant/len(authors)*100:.1f}%)',
        ha='center', fontsize=14, fontweight='bold', color=ACCENT5)
ax.set_title('Active vs Dormant Authors — May 2025', pad=15)
ax.set_ylabel('Users')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/15_active_dormant.png', bbox_inches='tight')
plt.close()
print("✅ 15_active_dormant.png")


print(f"\n✅ All charts saved to {OUT_DIR}/")
print("📁 Files:", sorted(os.listdir(OUT_DIR)))
