"""
eda/analysis.py
===============
9-step Exploratory Data Analysis for the Credit Card Fraud Detection project.
Each step returns a dict with: title, description, plot (Plotly JSON), stats.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils

PLOTLY_TEMPLATE = 'plotly_dark'


def _fig_to_json(fig) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


class EDAAnalyzer:
    """Performs 9-step EDA on a fraud detection DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._enrich()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _enrich(self):
        """Pre-compute convenient derived columns used by multiple steps."""
        df = self.df
        if 'trans_date_trans_time' in df.columns:
            dt = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
            df['_hour'] = dt.dt.hour
            df['_dow'] = dt.dt.day_name()
            df['_month'] = dt.dt.month_name()
            df['_date'] = dt.dt.date
        else:
            for c in ['_hour', '_dow', '_month', '_date']:
                df[c] = None

        if 'dob' in df.columns:
            dob = pd.to_datetime(df['dob'], errors='coerce')
            df['_age'] = ((pd.Timestamp('today') - dob).dt.days / 365.25).fillna(35).astype(int)
        else:
            df['_age'] = df.get('age', pd.Series(35, index=df.index))

        lat_cols = ['lat', 'long', 'merch_lat', 'merch_long']
        if all(c in df.columns for c in lat_cols):
            R = 6371.0
            lat1 = np.radians(df['lat'])
            lat2 = np.radians(df['merch_lat'])
            dlat = np.radians(df['merch_lat'] - df['lat'])
            dlon = np.radians(df['merch_long'] - df['long'])
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            df['_distance'] = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        else:
            df['_distance'] = 0.0

        self.has_target = 'is_fraud' in df.columns

    def _fraud_col(self) -> pd.Series:
        return self.df['is_fraud'] if self.has_target else pd.Series(0, index=self.df.index)

    # ==================================================================
    # Step 1 — Dataset Overview
    # ==================================================================
    def step1_overview(self) -> dict:
        df = self.df
        dtypes_count = df.dtypes.value_counts().rename(index=str).to_dict()
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        # Missing-values bar chart
        missing_df = pd.DataFrame({'Column': missing.index,
                                   'Missing': missing.values,
                                   'Pct': missing_pct.values})
        missing_df = missing_df[missing_df['Missing'] > 0]

        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Column', y='Pct',
                         title='Missing Values (%)',
                         template=PLOTLY_TEMPLATE,
                         color='Pct',
                         color_continuous_scale='Reds',
                         labels={'Pct': 'Missing %'})
        else:
            fig = go.Figure()
            fig.add_annotation(text='No missing values found!',
                               xref='paper', yref='paper', x=0.5, y=0.5,
                               showarrow=False, font=dict(size=20, color='#2ecc71'))
            fig.update_layout(template=PLOTLY_TEMPLATE,
                              title='Missing Values — Dataset is Complete')

        stats = {
            'Total Rows': f"{len(df):,}",
            'Total Columns': len(df.columns),
            'Numeric Columns': int(df.select_dtypes(include=np.number).shape[1]),
            'Categorical Columns': int(df.select_dtypes(include='object').shape[1]),
            'Total Missing Values': int(missing.sum()),
            'Memory Usage': f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB",
            'Duplicate Rows': int(df.duplicated().sum()),
            'Data Types': dtypes_count,
        }

        return {
            'title': 'Step 1: Dataset Overview',
            'description': (
                'A high-level summary of the dataset: shape, data types, missing values, '
                'memory footprint and duplicates.'
            ),
            'plot': _fig_to_json(fig),
            'stats': stats,
        }

    # ==================================================================
    # Step 2 — Target Distribution
    # ==================================================================
    def step2_target_distribution(self) -> dict:
        if not self.has_target:
            fig = go.Figure()
            fig.add_annotation(text='No target column (is_fraud) in this dataset.',
                               xref='paper', yref='paper', x=0.5, y=0.5,
                               showarrow=False, font=dict(size=16, color='#e74c3c'))
            fig.update_layout(template=PLOTLY_TEMPLATE, title='Target Distribution — N/A')
            return {'title': 'Step 2: Target Distribution', 'description': 'Target column not found.',
                    'plot': _fig_to_json(fig), 'stats': {}}

        fraud = self._fraud_col()
        counts = fraud.value_counts()
        labels = ['Legitimate', 'Fraud']
        values = [counts.get(0, 0), counts.get(1, 0)]
        colors = ['#2ecc71', '#e74c3c']

        fig = make_subplots(rows=1, cols=2,
                            specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                            subplot_titles=('', 'Class Counts'),
                            horizontal_spacing=0.12)

        fig.add_trace(go.Pie(labels=labels, values=values,
                             marker_colors=colors, hole=0.4,
                             textinfo='percent+label'), row=1, col=1)

        fig.add_trace(go.Bar(x=labels, y=values,
                             marker_color=colors,
                             text=[f'{v:,}' for v in values],
                             textposition='outside'), row=1, col=2)

        # "Class Proportion" label placed BELOW the pie chart
        fig.add_annotation(
            text='Class Proportion',
            x=0.18, y=-0.08,
            xref='paper', yref='paper',
            showarrow=False,
            font=dict(size=16, color='#e6edf3'),
            xanchor='center',
        )

        fig.update_layout(template=PLOTLY_TEMPLATE,
                          showlegend=False, height=440,
                          margin=dict(t=40, b=70, l=40, r=20))

        total = len(fraud)
        fraud_n = int(values[1])
        legit_n = int(values[0])
        ratio = fraud_n / legit_n if legit_n > 0 else 0

        stats = {
            'Total Transactions': f"{total:,}",
            'Legitimate': f"{legit_n:,} ({legit_n/total*100:.2f}%)",
            'Fraudulent': f"{fraud_n:,} ({fraud_n/total*100:.2f}%)",
            'Imbalance Ratio': f"1 fraud per {int(1/ratio) if ratio > 0 else 'N/A'} legitimate",
            'Fraud Rate': f"{fraud_n/total*100:.4f}%",
        }

        return {
            'title': 'Step 2: Target Distribution (Class Imbalance)',
            'description': (
                'Credit card fraud datasets are typically highly imbalanced — fraudulent '
                'transactions represent a tiny fraction of all transactions. Understanding '
                'this imbalance is critical before selecting evaluation metrics and models.'
            ),
            'plot': _fig_to_json(fig),
            'stats': stats,
        }

    # ==================================================================
    # Step 3 — Transaction Amount Analysis
    # ==================================================================
    def step3_transaction_amount(self) -> dict:
        df = self.df
        if 'amt' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text='Amount column (amt) not found.',
                               xref='paper', yref='paper', x=0.5, y=0.5,
                               showarrow=False, font=dict(size=16))
            fig.update_layout(template=PLOTLY_TEMPLATE)
            return {'title': 'Step 3: Transaction Amount', 'description': '',
                    'plot': _fig_to_json(fig), 'stats': {}}

        amt = df['amt']
        fraud_col = self._fraud_col()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Amount Distribution (log scale)',
                'Box Plot by Class',
                'Amount Histogram — Fraud',
                'Amount Histogram — Legitimate',
            )
        )

        # Log histogram
        fig.add_trace(go.Histogram(x=np.log1p(amt), nbinsx=80,
                                   marker_color='#3498db', name='All'), row=1, col=1)

        if self.has_target:
            fraud_amt = amt[fraud_col == 1]
            legit_amt = amt[fraud_col == 0]
            fig.add_trace(go.Box(y=fraud_amt, name='Fraud',
                                 marker_color='#e74c3c'), row=1, col=2)
            fig.add_trace(go.Box(y=legit_amt, name='Legitimate',
                                 marker_color='#2ecc71'), row=1, col=2)
            fig.add_trace(go.Histogram(x=fraud_amt, nbinsx=50,
                                       marker_color='#e74c3c', name='Fraud'), row=2, col=1)
            fig.add_trace(go.Histogram(x=legit_amt, nbinsx=50,
                                       marker_color='#2ecc71', name='Legit'), row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE,
                          height=580, showlegend=False,
                          margin=dict(t=40, b=40, l=40, r=20))

        stats = {
            'Min Amount': f"${amt.min():.2f}",
            'Max Amount': f"${amt.max():.2f}",
            'Mean Amount': f"${amt.mean():.2f}",
            'Median Amount': f"${amt.median():.2f}",
            'Std Deviation': f"${amt.std():.2f}",
        }
        if self.has_target:
            stats['Avg Fraud Amount'] = f"${amt[fraud_col == 1].mean():.2f}"
            stats['Avg Legit Amount'] = f"${amt[fraud_col == 0].mean():.2f}"

        return {
            'title': 'Step 3: Transaction Amount Analysis',
            'description': (
                'Fraudulent transactions often differ in amount from legitimate ones. '
                'We examine the distribution on a log scale, compare box plots by class, '
                'and highlight extreme values.'
            ),
            'plot': _fig_to_json(fig),
            'stats': stats,
        }

    # ==================================================================
    # Step 4 — Temporal Analysis
    # ==================================================================
    def step4_temporal_analysis(self) -> dict:
        df = self.df
        fraud_col = self._fraud_col()

        if df['_hour'].isnull().all():
            fig = go.Figure()
            fig.add_annotation(text='Temporal column not found.',
                               xref='paper', yref='paper', x=0.5, y=0.5,
                               showarrow=False, font=dict(size=16))
            fig.update_layout(template=PLOTLY_TEMPLATE)
            return {'title': 'Step 4: Temporal Analysis', 'description': '',
                    'plot': _fig_to_json(fig), 'stats': {}}

        hour_counts = df.groupby('_hour').size()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = df.groupby('_dow').size().reindex(dow_order, fill_value=0)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Transactions by Hour of Day',
                'Transactions by Day of Week',
                'Fraud Rate by Hour',
                'Transactions by Month',
            )
        )

        fig.add_trace(go.Bar(x=hour_counts.index, y=hour_counts.values,
                             marker_color='#3498db', name='Hour'), row=1, col=1)
        fig.add_trace(go.Bar(x=dow_counts.index, y=dow_counts.values,
                             marker_color='#9b59b6', name='DoW'), row=1, col=2)

        if self.has_target:
            hour_fraud_rate = df.groupby('_hour')['is_fraud'].mean() * 100
            fig.add_trace(go.Scatter(x=hour_fraud_rate.index, y=hour_fraud_rate.values,
                                     mode='lines+markers', marker_color='#e74c3c',
                                     name='Fraud Rate %'), row=2, col=1)

        month_counts = df.groupby('_month').size()
        fig.add_trace(go.Bar(x=month_counts.index, y=month_counts.values,
                             marker_color='#f39c12', name='Month'), row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE,
                          height=580, showlegend=False,
                          margin=dict(t=40, b=40, l=40, r=20))

        peak_hour = int(hour_counts.idxmax()) if len(hour_counts) else 'N/A'
        peak_dow = str(dow_counts.idxmax()) if len(dow_counts) else 'N/A'

        stats = {
            'Peak Hour (most transactions)': f"{peak_hour}:00",
            'Peak Day of Week': peak_dow,
            'Date Range': f"{df['_date'].min()} → {df['_date'].max()}" if df['_date'].notnull().any() else 'N/A',
        }
        if self.has_target:
            most_fraud_hour = int(df.groupby('_hour')['is_fraud'].mean().idxmax())
            stats['Highest Fraud Rate Hour'] = f"{most_fraud_hour}:00"

        return {
            'title': 'Step 4: Temporal Analysis',
            'description': (
                'Time-of-day, day-of-week and monthly patterns reveal when fraud is '
                'most likely to occur. Late-night hours often show elevated fraud rates.'
            ),
            'plot': _fig_to_json(fig),
            'stats': stats,
        }

    # ==================================================================
    # Step 5 — Category Analysis
    # ==================================================================
    def step5_category_analysis(self) -> dict:
        df = self.df
        if 'category' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text='Category column not found.',
                               xref='paper', yref='paper', x=0.5, y=0.5,
                               showarrow=False, font=dict(size=16))
            fig.update_layout(template=PLOTLY_TEMPLATE)
            return {'title': 'Step 5: Category Analysis', 'description': '',
                    'plot': _fig_to_json(fig), 'stats': {}}

        cat_counts = df['category'].value_counts().head(15)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Transaction Count by Category',
                                            'Fraud Rate by Category (%)'))

        fig.add_trace(go.Bar(x=cat_counts.values, y=cat_counts.index,
                             orientation='h', marker_color='#3498db',
                             name='Count'), row=1, col=1)

        if self.has_target:
            fraud_by_cat = df.groupby('category')['is_fraud'].mean() * 100
            fraud_by_cat = fraud_by_cat.sort_values(ascending=False).head(15)
            colors = ['#e74c3c' if v > fraud_by_cat.mean() else '#f39c12'
                      for v in fraud_by_cat.values]
            fig.add_trace(go.Bar(x=fraud_by_cat.values, y=fraud_by_cat.index,
                                 orientation='h', marker_color=colors,
                                 name='Fraud %'), row=1, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE,
                          height=480, showlegend=False,
                          margin=dict(t=40, b=40, l=40, r=20))

        stats = {
            'Total Categories': int(df['category'].nunique()),
            'Most Common Category': str(cat_counts.index[0]),
            'Transactions in Top Category': f"{int(cat_counts.iloc[0]):,}",
        }
        if self.has_target:
            top_fraud_cat = df.groupby('category')['is_fraud'].mean().idxmax()
            stats['Highest Fraud Rate Category'] = str(top_fraud_cat)
            stats['Fraud Rate in Top Category'] = f"{df.groupby('category')['is_fraud'].mean().max()*100:.2f}%"

        return {
            'title': 'Step 5: Merchant Category Analysis',
            'description': (
                'Different merchant categories carry vastly different fraud risks. '
                'Shopping, travel, and online categories typically show higher fraud rates.'
            ),
            'plot': _fig_to_json(fig),
            'stats': stats,
        }

    # ==================================================================
    # Step 6 — Geographic Analysis
    # ==================================================================
    def step6_geographic_analysis(self) -> dict:
        df = self.df
        if 'state' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text='State column not found.',
                               xref='paper', yref='paper', x=0.5, y=0.5,
                               showarrow=False, font=dict(size=16))
            fig.update_layout(template=PLOTLY_TEMPLATE)
            return {'title': 'Step 6: Geographic Analysis', 'description': '',
                    'plot': _fig_to_json(fig), 'stats': {}}

        state_counts = df['state'].value_counts().head(20)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Top 20 States by Transaction Count',
                                            'Fraud Rate by State (%)'))

        fig.add_trace(go.Bar(x=state_counts.index, y=state_counts.values,
                             marker_color='#3498db', name='Count'), row=1, col=1)

        if self.has_target:
            state_fraud = df.groupby('state')['is_fraud'].mean() * 100
            state_fraud = state_fraud.sort_values(ascending=False).head(20)
            fig.add_trace(go.Bar(x=state_fraud.index, y=state_fraud.values,
                                 marker_color='#e74c3c', name='Fraud %'), row=1, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE,
                          height=480, showlegend=False,
                          margin=dict(t=40, b=40, l=40, r=20))

        stats = {
            'Total States': int(df['state'].nunique()),
            'Most Active State': str(state_counts.index[0]),
        }
        if self.has_target:
            top_fraud_state = df.groupby('state')['is_fraud'].mean().idxmax()
            stats['Highest Fraud State'] = str(top_fraud_state)

        return {
            'title': 'Step 6: Geographic Analysis',
            'description': (
                'Geographic patterns can reveal regional fraud hotspots. '
                'Transaction volume and fraud rates vary significantly by state.'
            ),
            'plot': _fig_to_json(fig),
            'stats': stats,
        }

    # ==================================================================
    # Step 7 — Customer Analysis
    # ==================================================================
    def step7_customer_analysis(self) -> dict:
        df = self.df

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'domain'}],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=('Age Distribution', 'Gender Breakdown',
                            'Top Job Categories', 'Fraud Rate by Gender')
        )

        # Age distribution
        fig.add_trace(go.Histogram(x=df['_age'], nbinsx=30,
                                   marker_color='#3498db', name='Age'), row=1, col=1)

        # Gender
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            fig.add_trace(go.Pie(labels=gender_counts.index.tolist(),
                                 values=gender_counts.values.tolist(),
                                 marker_colors=['#9b59b6', '#e67e22'],
                                 hole=0.4, name='Gender'), row=1, col=2)

        # Job categories
        if 'job' in df.columns:
            job_counts = df['job'].value_counts().head(10)
            fig.add_trace(go.Bar(x=job_counts.values, y=job_counts.index,
                                 orientation='h', marker_color='#27ae60',
                                 name='Job'), row=2, col=1)

        # Fraud by gender
        if 'gender' in df.columns and self.has_target:
            fraud_by_gender = df.groupby('gender')['is_fraud'].mean() * 100
            fig.add_trace(go.Bar(x=fraud_by_gender.index.tolist(),
                                 y=fraud_by_gender.values.tolist(),
                                 marker_color=['#e74c3c', '#f39c12'],
                                 name='Fraud %'), row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE,
                          height=580, showlegend=False,
                          margin=dict(t=40, b=40, l=40, r=20))

        age = df['_age']
        stats = {
            'Age Range': f"{int(age.min())} – {int(age.max())} years",
            'Average Age': f"{age.mean():.1f} years",
            'Median Age': f"{age.median():.1f} years",
        }
        if 'gender' in df.columns:
            stats['Gender Split'] = {
                str(key): f"{int(value):,}"
                for key, value in df['gender'].value_counts().to_dict().items()
            }

        return {
            'title': 'Step 7: Customer Demographics',
            'description': (
                'Understanding the demographic profile of cardholders helps identify '
                'at-risk groups and improve targeted fraud prevention strategies.'
            ),
            'plot': _fig_to_json(fig),
            'stats': stats,
        }

    # ==================================================================
    # Step 8 — Correlation Analysis
    # ==================================================================
    def step8_correlation_analysis(self) -> dict:
        df = self.df
        num_df = df.select_dtypes(include=np.number).copy()

        # Drop internally added underscore columns
        num_df = num_df[[c for c in num_df.columns if not c.startswith('_')]]

        # Limit to 20 most correlated with is_fraud (if present)
        if 'is_fraud' in num_df.columns and num_df.shape[1] > 2:
            corr_with_target = num_df.corr()['is_fraud'].abs().sort_values(ascending=False)
            top_cols = corr_with_target.head(20).index.tolist()
            num_df = num_df[top_cols]

        corr_matrix = num_df.corr()

        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont_size=9,
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          height=560,
                          margin=dict(t=40, b=40, l=40, r=20))

        stats = {
            'Numeric Features Analyzed': len(corr_matrix.columns),
        }
        if 'is_fraud' in corr_matrix.columns:
            top_corr = corr_matrix['is_fraud'].drop('is_fraud').abs().sort_values(ascending=False)
            stats['Top Correlated Feature'] = str(top_corr.index[0])
            stats['Top Correlation Value'] = f"{top_corr.iloc[0]:.4f}"

        return {
            'title': 'Step 8: Correlation Analysis',
            'description': (
                'The correlation heatmap reveals linear relationships between numeric features '
                'and the fraud target. High absolute correlation with is_fraud indicates '
                'predictive power.'
            ),
            'plot': _fig_to_json(fig),
            'stats': stats,
        }

    # ==================================================================
    # Step 9 — Fraud Patterns
    # ==================================================================
    def step9_fraud_patterns(self) -> dict:
        df = self.df
        fraud_col = self._fraud_col()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distance Distribution (Fraud vs Legit)',
                'Amount vs Distance (Fraud Highlighted)',
                'Fraud by Hour Heatmap',
                'Key Fraud Indicators',
            )
        )

        # Distance histogram
        if self.has_target:
            fraud_dist = df.loc[fraud_col == 1, '_distance']
            legit_dist = df.loc[fraud_col == 0, '_distance']
            fig.add_trace(go.Histogram(x=fraud_dist, nbinsx=50,
                                       marker_color='#e74c3c', opacity=0.7,
                                       name='Fraud'), row=1, col=1)
            fig.add_trace(go.Histogram(x=legit_dist, nbinsx=50,
                                       marker_color='#2ecc71', opacity=0.7,
                                       name='Legit'), row=1, col=1)
        else:
            fig.add_trace(go.Histogram(x=df['_distance'], nbinsx=50,
                                       marker_color='#3498db', name='Distance'), row=1, col=1)

        # Scatter: amt vs distance
        if 'amt' in df.columns:
            sample = df.sample(min(2000, len(df)), random_state=42)
            colors = ['#e74c3c' if f == 1 else '#2ecc71'
                      for f in (fraud_col.loc[sample.index] if self.has_target else [0]*len(sample))]
            fig.add_trace(go.Scatter(
                x=sample['_distance'],
                y=sample['amt'],
                mode='markers',
                marker=dict(color=colors, size=4, opacity=0.6),
                name='Transactions',
            ), row=1, col=2)

        # Fraud heatmap by hour and day
        if self.has_target and df['_hour'].notnull().any():
            pivot = df.groupby(['_dow', '_hour'])['is_fraud'].mean() * 100
            pivot = pivot.unstack(fill_value=0)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot = pivot.reindex([d for d in day_order if d in pivot.index])
            fig.add_trace(go.Heatmap(
                z=pivot.values,
                x=list(range(24)),
                y=pivot.index.tolist(),
                colorscale='Reds',
                name='Fraud Rate',
            ), row=2, col=1)

        # Key indicators bar
        indicators = {}
        if self.has_target:
            if 'amt' in df.columns:
                indicators['Avg Fraud Amt'] = float(df.loc[fraud_col == 1, 'amt'].mean())
                indicators['Avg Legit Amt'] = float(df.loc[fraud_col == 0, 'amt'].mean())
            indicators['Avg Fraud Dist'] = float(df.loc[fraud_col == 1, '_distance'].mean())
            indicators['Avg Legit Dist'] = float(df.loc[fraud_col == 0, '_distance'].mean())

        if indicators:
            fig.add_trace(go.Bar(
                x=list(indicators.keys()),
                y=list(indicators.values()),
                marker_color=['#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71'],
                name='Indicators',
            ), row=2, col=2)

        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title='Fraud Patterns & Key Indicators',
                          height=650, showlegend=True,
                          barmode='overlay')

        stats = {
            'Avg Transaction Distance': f"{df['_distance'].mean():.2f} km",
        }
        if self.has_target:
            stats['Avg Fraud Distance'] = f"{df.loc[fraud_col == 1, '_distance'].mean():.2f} km"
            stats['Avg Legit Distance'] = f"{df.loc[fraud_col == 0, '_distance'].mean():.2f} km"
            if 'amt' in df.columns:
                stats['Avg Fraud Amount'] = f"${df.loc[fraud_col == 1, 'amt'].mean():.2f}"
                stats['Avg Legit Amount'] = f"${df.loc[fraud_col == 0, 'amt'].mean():.2f}"

        return {
            'title': 'Step 9: Fraud Patterns & Key Indicators',
            'description': (
                'Combining distance, amount, time and merchant patterns uncovers the '
                'most discriminating signals for fraud detection. These insights directly '
                'inform feature engineering and model selection.'
            ),
            'plot': _fig_to_json(fig),
            'stats': stats,
        }

    # ==================================================================
    # Run all steps
    # ==================================================================
    def run_all(self) -> list[dict]:
        steps = [
            self.step1_overview,
            self.step2_target_distribution,
            self.step3_transaction_amount,
            self.step4_temporal_analysis,
            self.step5_category_analysis,
            self.step6_geographic_analysis,
            self.step7_customer_analysis,
            self.step8_correlation_analysis,
            self.step9_fraud_patterns,
        ]
        results = []
        for i, step_fn in enumerate(steps, start=1):
            try:
                result = step_fn()
            except Exception as exc:
                fig = go.Figure()
                fig.add_annotation(text=f'Error: {exc}',
                                   xref='paper', yref='paper', x=0.5, y=0.5,
                                   showarrow=False, font=dict(size=14, color='#e74c3c'))
                fig.update_layout(template=PLOTLY_TEMPLATE)
                result = {
                    'title': f'Step {i}: Error',
                    'description': str(exc),
                    'plot': _fig_to_json(fig),
                    'stats': {},
                }
            result['step'] = i
            results.append(result)
        return results
