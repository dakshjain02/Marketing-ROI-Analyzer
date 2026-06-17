import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

def calculate_metrics(df):

    df["Profit"] = df["Revenue"] - df["Spend"]

    df["ROI"] = (
        (df["Revenue"] - df["Spend"])
        / df["Spend"]
    ) * 100

    df["CPC"] = (
        df["Spend"]
        / df["Clicks"]
    )

    df["Conversion Rate"] = (
        df["Conversions"]
        / df["Clicks"]
    ) * 100

    # Round values
    df["ROI"] = df["ROI"].round(2)
    df["CPC"] = df["CPC"].round(2)
    df["Conversion Rate"] = df["Conversion Rate"].round(2)

    return df

def get_kpis(df):

    return {
    "total_revenue": f"{df['Revenue'].sum():,.0f}",
    "total_spend": f"{df['Spend'].sum():,.0f}",
    "total_profit": f"{df['Profit'].sum():,.0f}",
    "avg_roi": round(df["ROI"].mean(), 2)
}

def revenue_chart(df):

    campaign_revenue = df.groupby("Campaign")["Revenue"].sum().reset_index()

    fig = px.bar(
    campaign_revenue,
    x="Campaign",
    y="Revenue",
    title="Revenue by Campaign"
)

    fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#1e293b",
    plot_bgcolor="#1e293b",
    font_color="white"
)
    fig.update_layout(
    hoverlabel=dict(
        bgcolor="#111827",
        bordercolor="#22c55e",
        font_color="#22c55e",
        font_size=15
    )
)
    fig.update_traces(
    marker_color="#38bdf8"
)
    return fig.to_html(
    full_html=False,
    config={"displayModeBar": False}
)

def generate_insights(df):

    best = df.loc[df["ROI"].idxmax()]

    return {
        "best_campaign": best["Campaign"],
        "best_roi": round(best["ROI"], 2),
        "profit": round(df["Profit"].sum(), 2)
    }

def roi_chart(df):

    campaign_roi = df.groupby("Campaign")["ROI"].mean().reset_index()

    fig = px.bar(
    campaign_roi,
    x="Campaign",
    y="ROI",
    title="Average ROI by Campaign"
)

    fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#1e293b",
    plot_bgcolor="#1e293b",
    font_color="white"
)
    fig.update_layout(
    hoverlabel=dict(
        bgcolor="#111827",
        bordercolor="#c5af22",
        font_color="#c5ad22",
        font_size=15
    )
)
    
    fig.update_traces(
    marker_color="#22c55e"
)
    return fig.to_html(
    full_html=False,
    config={"displayModeBar": False}
)

    return fig.to_html(full_html=False)

def pie_chart(df):

    channel_revenue = df.groupby("Channel")["Revenue"].sum().reset_index()

    fig = px.pie(
        channel_revenue,
        names="Channel",
        values="Revenue",
        title="Revenue Distribution by Channel"
    )

    fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#1e293b",
    plot_bgcolor="#1e293b",
    font_color="white"
)
    fig.update_layout(
    hoverlabel=dict(
        bgcolor="#111827",
        bordercolor="#c5b822",
        font_color="#c52248",
        font_size=15
    )
)
    
    fig.update_traces(
    textinfo="percent+label"
)
    return fig.to_html(
    full_html=False,
    config={"displayModeBar": False}
)

    return fig.to_html(full_html=False)

def clean_data(df):

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Fill missing values
    df.fillna(0, inplace=True)

    return df

def data_quality_report(df):

    report = {

        "total_rows": len(df),

        "missing_values": int(df.isnull().sum().sum()),

        "duplicate_rows": int(df.duplicated().sum())

    }

    return report

def detect_outliers(df):

    Q1 = df["Spend"].quantile(0.25)

    Q3 = df["Spend"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df["Spend"] < lower)
        |
        (df["Spend"] > upper)
    ]

    return len(outliers)

def train_roi_model(df):

    features = ["Spend", "Clicks", "Conversions"]

    X = df[features]

    y = df["ROI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {

        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    best_model = None
    best_accuracy = -999
    model_scores = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        score = model.score(X_test, y_test) * 100

        model_scores[name] = round(score, 2)

        if score > best_accuracy:

            best_accuracy = round(score, 2)

            best_model = model

            best_model_name = name

    predictions = best_model.predict(X_test)

    predicted_roi = round(
        predictions.mean(),
        2
    )

    return (
        predicted_roi,
        best_accuracy,
        best_model_name,
        model_scores
    )

def feature_importance_chart(df):

    features = ["Spend", "Clicks", "Conversions"]

    X = df[features]

    y = df["ROI"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    fig = px.bar(
        imp_df,
        x="Feature",
        y="Importance",
        title="Feature Importance for ROI Prediction"
    )

    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font_color="white"
    )

    return fig.to_html(
    full_html=False,
    config={"displayModeBar": False}
    )

def preprocess_data(df):

    # Missing Values
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Duplicates
    df = df.drop_duplicates()

    # Outliers
    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[
            (df[col] >= lower) &
            (df[col] <= upper)
        ]

    return df

def data_quality_before(df):

    return {
        "rows": len(df),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "outliers": detect_outliers(df)
    }

def data_quality_after(df):

    return {
        "rows": len(df),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "outliers": detect_outliers(df)
    }

def generate_recommendations(df):

    recommendations = []

    best_campaign = df.loc[df["ROI"].idxmax(), "Campaign"]
    recommendations.append(
        f"Increase investment in {best_campaign} campaign as it has the highest ROI."
    )

    avg_conversion = df["Conversion Rate"].mean()

    low_conversion = df[
        df["Conversion Rate"] < avg_conversion
    ]

    if not low_conversion.empty:
        recommendations.append(
            "Some campaigns have below-average conversion rates and need optimization."
        )

    top_channel = (
        df.groupby("Channel")["Revenue"]
        .sum()
        .idxmax()
    )

    recommendations.append(
        f"{top_channel} is currently the highest revenue generating channel."
    )

    if df["ROI"].mean() > 150:
        recommendations.append(
            "Overall campaign performance is strong with excellent ROI."
        )

    return recommendations