import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer



def imputer(X_train,X_val,X_test):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    # For Typical ticket spend & usual booking lead
    cols = ['typical_ticket_spend','usual_booking_lead_days']
    imp = SimpleImputer(strategy='constant',fill_value=0)
    X_train[cols] = imp.fit_transform(X_train[cols])
    X_val[cols] = imp.transform(X_val[cols])
    X_test[cols] = imp.transform(X_test[cols])

    # For event page visits median is 0 ,mode was 0 so i'll impute it with most_freq
    imp = SimpleImputer(strategy='most_frequent')
    X_train[['event_page_visits_7d']] = imp.fit_transform(X_train[['event_page_visits_7d']])
    X_val[['event_page_visits_7d']] = imp.transform(X_val[['event_page_visits_7d']])
    X_test[['event_page_visits_7d']] = imp.transform(X_test[['event_page_visits_7d']])
    # For total sessions missing i'll impute w zero but also make a feature inside 
    # Indicating that the value was missing

    X_train["total_sessions_missing"] = (
        X_train["total_sessions"].isna().astype(int)
    )
    X_val["total_sessions_missing"] = (
            X_val["total_sessions"].isna().astype(int)
        )
    X_test["total_sessions_missing"]=(
        X_test["total_sessions"].isna().astype(int)
    )
    cols =["total_sessions"]

    imp = SimpleImputer(
        strategy="constant",
        fill_value=0
    )

    X_train[cols] = imp.fit_transform(X_train[cols])
    X_val[cols] = imp.transform(X_val[cols])
    X_test[cols] = imp.transform(X_test[cols])
    # Activity status

    cols =['activity_status']
    imp = SimpleImputer(strategy='constant',fill_value='no_account')
    X_train[cols] = imp.fit_transform(X_train[cols])
    X_val[cols] = imp.transform(X_val[cols])
    X_test[cols] = imp.transform(X_test[cols])

    # For artist saved rate and avg completion rate
    cols = ["artist_saved_rate", "avg_completion_rate"]

    # Creating one shared missingness indicator since whenever one is missing others missing as well
    X_train["music_rate_missing"] = (
        X_train[cols].isna().any(axis=1).astype(int)
    )

    X_val["music_rate_missing"] = (
        X_val[cols].isna().any(axis=1).astype(int)
    )
    X_test["music_rate_missing"] = (
            X_test[cols].isna().any(axis=1).astype(int)
        )

    # Median imputation: fit on training, transform validation
    imp = SimpleImputer(strategy="median")
    X_train[cols] = imp.fit_transform(X_train[cols])
    X_val[cols] = imp.transform(X_val[cols])
    X_test[cols] = imp.transform(X_test[cols])

    # Now for "concert_genre_total_saved","avg_genre_plays","max_genre_plays"
    # Imputing zero and introducing one more missing indicator for all 3
    # Create one shared missingness indicator
    cols = [
        "concert_genre_total_saved",
        "avg_genre_plays",
        "max_genre_plays"
    ]

    X_train["genre_features_missing"] = (
        X_train[cols].isna().any(axis=1).astype(int)
    )

    X_val["genre_features_missing"] = (
        X_val[cols].isna().any(axis=1).astype(int)
    )

    X_test["genre_features_missing"] = (
            X_test[cols].isna().any(axis=1).astype(int)
        )

    # Zero imputation
    imp = SimpleImputer(
        strategy="constant",
        fill_value=0
    )

    X_train[cols] = imp.fit_transform(X_train[cols])
    X_val[cols] = imp.transform(X_val[cols])
    X_test[cols] = imp.transform(X_test[cols])

    # for shares
    cols = [
    "artist_play_share",
    "concert_genre_play_share"
    ]

    # Shared missingness indicator
    X_train["music_share_missing"] = (
        X_train[cols].isna().any(axis=1).astype(int)
    )

    X_val["music_share_missing"] = (
        X_val[cols].isna().any(axis=1).astype(int)
    )
    X_test["music_share_missing"] = (
            X_test[cols].isna().any(axis=1).astype(int)
        )

    # Zero imputation
    imp = SimpleImputer(strategy="constant", fill_value=0)

    X_train[cols] = imp.fit_transform(X_train[cols])
    X_val[cols] = imp.transform(X_val[cols])
    X_test[cols] = imp.transform(X_test[cols])

    # For fav genre match
    
    cols = ["fav_genre_match"]

    imp = SimpleImputer(strategy="constant", fill_value=0)

    X_train[cols] = imp.fit_transform(X_train[cols])
    X_val[cols] = imp.transform(X_val[cols])
    X_test[cols] = imp.transform(X_test[cols])


    return X_train,X_val,X_test


