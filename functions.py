import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression

def check_df(df, rows=5):
    print("##################### SHAPE #####################")
    print(df.shape)
    print("##################### TYPES #####################")
    print(df.dtypes)
    print("##################### HEAD #####################")
    print(df.head(rows))
    print("##################### TAIL #####################")
    print(df.tail(rows))
    print("##################### N/A #####################")
    print(df.isnull().sum())

def plot_column_distribution(df, column, title):
    column_count = df[column].value_counts().reset_index()
    column_count.columns = [column, 'count']
    fig = px.histogram(column_count, x=column, y='count', marginal="box", hover_data=column_count.columns, title=title)
    fig.show()
    print(f"Statystyki dla kolumny '{column}':")
    print(df[column].describe())

def plot_boxplot(df, x, y, title, labels):
    fig = px.box(df, x=x, y=y, title=title, labels=labels)
    fig.show()

def plot_line(df, x, y, title, labels):
    fig = px.line(df, x=x, y=y, markers=True, title=title, labels=labels)
    fig.show()

def plot_line_subplots(df, time_col, value_col, variables, title):
    fig = make_subplots(rows=len(variables), cols=1, subplot_titles=[f'{var}' for var in variables], shared_xaxes=True)
    for i, var in enumerate(variables, start=1):
        fig.add_trace(go.Scatter(x=df[time_col], y=df[var], mode='lines+markers', name=var.capitalize()), row=i, col=1)
        fig.add_trace(go.Scatter(x=df[time_col], y=df[value_col], mode='lines+markers', name=value_col.capitalize()), row=i, col=1)
    fig.update_layout(title=title, height=300 * len(variables),)
    fig.show()

def plot_boxplot_summary(df, title):
    numeric_columns = df.select_dtypes(include='number').columns[1:]
    df_numeric = df[numeric_columns]
    df_long = df_numeric.melt(var_name='Variable', value_name='Value')
    fig = px.box(df_long, x='Variable', y='Value', title=title)
    fig.show()
    print("Statystyki dla zmiennych numerycznych:")
    print(df_numeric.describe())

def outliers_summary(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_count = len(outliers)
    outlier_stats = outliers[column].describe()
    print(f"Łączna liczba wartości odstających: {outlier_count}\n")
    print(f"Statystyki dla wartości odstających:")
    print(outlier_stats)

def replace_outliers_and_nans(df, column):
    df_copy = df.copy()
    df_copy.set_index('date', inplace=True)
    Q1 = df_copy[column].quantile(0.25)
    Q3 = df_copy[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_copy[column] = df_copy[column].where((df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound), np.nan)
    df_copy[column] = df_copy[column].interpolate(method='time')
    df_copy[column] = df_copy[column].fillna(df_copy[column].median())
    return df_copy.reset_index()

def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year
    df['day_of_week'] = df[date_column].dt.dayofweek + 1
    df["is_wknd"] = df[date_column].dt.weekday // 4
    return df

def plot_autocorrelation(series, title, alpha=0.05, show_pacf=False):
    corr_array = pacf(series.dropna(), alpha=alpha) if show_pacf else acf(series.dropna(), alpha=alpha)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]
    fig = go.Figure()
    [fig.add_scatter(x=(x, x), y=(0, corr_array[0][x]), mode='lines', line_color='#3f3f3f') 
     for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                   marker_size=12)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',
                    fillcolor='rgba(32, 146, 230, 0.3)', fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, len(corr_array[0])])  
    fig.update_yaxes(zerolinecolor='#000000')  
    title = title
    fig.update_layout(title=title)
    fig.show()

def plot_seasonal_decompose(df, column,  title, period=None):
    df = df.copy()
    df.set_index('date', inplace=True)
    result = seasonal_decompose(df[column], period=period)
    fig = make_subplots(rows=4, cols=1, subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residuals'), row=4, col=1)
    fig.update_layout(title=title, showlegend=False, height=800, xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    return fig

def plot_correlation_clustermap(df, title):
    correlations = df.corr()
    fig = px.imshow(correlations, text_auto=True, title=title, color_continuous_scale='Blues', zmin=-1, zmax=1, aspect='auto')
    fig.show()

def plot_nonlinear_relationship(df, target, title):
    features = df.select_dtypes(include=[np.number]).drop(columns=[target]).columns
    results = {feature: {'original': stats.spearmanr(df[feature], df[target])[0], 'mutual_info': mutual_info_regression(df[[feature]], df[target])[0], 'log': stats.spearmanr(np.log1p(df[feature] - df[feature].min() + 1), df[target])[0], 'square': stats.spearmanr(df[feature]**2, df[target])[0], 'sqrt': stats.spearmanr(np.sqrt(df[feature] - df[feature].min()), df[target])[0], 'inverse': stats.spearmanr(1 / (df[feature] - df[feature].min() + 1), df[target])[0]} for feature in features}
    results_df = pd.DataFrame(results).T
    fig = go.Figure(data=go.Heatmap(z=results_df.values, x=results_df.columns, y=results_df.index, colorscale='Blues', text=np.round(results_df.values, 3), texttemplate='%{text}'))
    fig.update_layout(title=title, xaxis_title='Typ transformacji', yaxis_title='Zmienna')
    return fig

def group_normality_and_comparison(df, target, group_col, alpha=0.05, alternative='two-sided'):
    group_data = {group: df[df[group_col] == group][target] for group in df[group_col].unique()}
    shapiro_results = {group: stats.shapiro(data).pvalue for group, data in group_data.items()}
    for group, p_value in shapiro_results.items():
        print(f"Test Shapiro-Wilka dla grupy {group}: p = {p_value:.4f}")
    if len(group_data) == 2:
        test_used, stat, p_value = (
            ("Test t-Studenta", *stats.ttest_ind(*group_data.values(), alternative=alternative)) if all(p > 0.05 for p in shapiro_results.values())
            else ("Test Manna-Whitneya", *stats.mannwhitneyu(*group_data.values(), alternative=alternative)))
    else:
        test_used, stat, p_value = (
            ("ANOVA (analiza wariancji)", *stats.f_oneway(*group_data.values())) if all(p > 0.05 for p in shapiro_results.values())
            else ("Test Kruskala-Wallisa", *stats.kruskal(*group_data.values())))
    print(f"{test_used}: statystyka = {stat:.4f}, p-wartość = {p_value:.4f}")
    print("Różnice między grupami są statystycznie istotne." if p_value < alpha else "Brak istotnych różnic między grupami.")


def test_weather_variable(df, weather_variable, bins, bin_labels, alpha=0.05):
    df_copy = df.copy()

    def categorize_weather(value):
        for i, b in enumerate(bins[:-1]):
            if b <= value < bins[i + 1]:
                return bin_labels[i]
        return bin_labels[-1] 
    
    df_copy[f'{weather_variable}_bin'] = df_copy[weather_variable].apply(categorize_weather)
        
    fig_scatter = px.scatter(df_copy, x=weather_variable, y='value', color=f'{weather_variable}_bin', title=f"Liczba kroków w zależności od {weather_variable}",
        labels={weather_variable: f'{weather_variable}', 'value': 'Liczba kroków'}, opacity=0.6)
    fig_scatter.show()

    fig_box = px.box(
        df_copy, x=f'{weather_variable}_bin', y='value', color=f'{weather_variable}_bin', title=f"Porównanie liczby kroków w różnych zakresach {weather_variable}",
        labels={f'{weather_variable}_bin': f'Zakres {weather_variable}', 'value': 'Liczba kroków'})
    fig_box.show()

    grouped = df_copy.groupby(f'{weather_variable}_bin')['value']

    for group_name, group_data in grouped:
        shapiro_test = stats.shapiro(group_data)
        print(f"Test Shapiro-Wilka dla grupy '{group_name}': statystyka={shapiro_test.statistic:.4f}, p-wartość={shapiro_test.pvalue:.5f}")

    all_normal = all(stats.shapiro(group_data).pvalue > alpha for _, group_data in grouped)
    
    if all_normal:
        f_stat, p_value = stats.f_oneway(*[group_data for _, group_data in grouped])
        print(f"\nANOVA - Statystyka F: {f_stat:.2f}, wartość p: {p_value:.5f}")
        
        if p_value < alpha:
            print(f"Odrzucamy hipotezę zerową: liczba kroków różni się między grupami {weather_variable}.")
        else:
            print(f"Brak podstaw do odrzucenia hipotezy zerowej: liczba kroków nie różni się istotnie między grupami {weather_variable}.")
    else:
        h_stat, p_value = stats.kruskal(*[group_data for _, group_data in grouped])
        print(f"\nTest Kruskala-Wallisa - Statystyka H: {h_stat:.2f}, wartość p: {p_value:.5f}")
        
        if p_value < alpha:
            print(f"Odrzucamy hipotezę zerową: liczba kroków różni się między grupami {weather_variable}.")
        else:
            print(f"Brak podstaw do odrzucenia hipotezy zerowej: liczba kroków nie różni się istotnie między grupami {weather_variable}.")