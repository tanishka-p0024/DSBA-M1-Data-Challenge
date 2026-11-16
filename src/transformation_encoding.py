import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor():

    # Ordinal feature - I'll preserve anad consider post>under>secoundary
    ordinal_cols = ["education"]

    education_order = [[
        "secondary",
        "undergraduate",
        "postgraduate"
    ]]

    categorical_cols = [
        "home_city",
        "venue_city",
        "concert_id",
        "activity_status"
    ]


    numerical_cols = [
        "age",
        "past_purchase_count",
        "typical_ticket_spend",
        "usual_booking_lead_days",
        "event_page_visits_7d",
        "total_sessions",
        "artist_saved_rate",
        "avg_completion_rate",
        "concert_genre_total_saved",
        "avg_genre_plays",
        "max_genre_plays",
        "artist_play_share",
        "concert_genre_play_share",
        "days_since_sale_opened",
        "days_until_concert",
        "distance",
        "total_cost"
    ]


    preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),

        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_cols),

        ("ordinal", OrdinalEncoder(
            categories=education_order
        ), ordinal_cols)
    ],
    remainder="passthrough"
    )

    return preprocessor
