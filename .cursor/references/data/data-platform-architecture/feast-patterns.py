# feast_patterns.py -- Feature definitions, serving, and point-in-time joins
from datetime import timedelta, datetime

from feast import Entity, FeatureStore, Field, BatchFeatureView
from feast.data_source import PushMode
from feast.types import Float32, Int64, String
from feast.infra.offline_stores.bigquery_source import BigQuerySource
import pandas as pd


# --- Feature Store Config (feature_store.yaml) ---
# project: my_ml_project
# registry: gs://my-bucket/feast/registry.pb
# provider: gcp
# online_store:
#   type: redis
#   connection_string: redis://10.0.0.5:6379
# offline_store:
#   type: bigquery
# entity_key_serialization_version: 2


# --- Entities ---
user = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="Unique user identifier",
)

product = Entity(
    name="product_id",
    join_keys=["product_id"],
)


# --- Data Sources ---
user_stats_source = BigQuerySource(
    name="user_stats",
    table="ml_features.user_daily_stats",
    timestamp_field="event_date",
    created_timestamp_column="created_at",
)


# --- Feature Views ---
user_features = BatchFeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="order_count_30d", dtype=Int64),
        Field(name="avg_order_value_30d", dtype=Float32),
        Field(name="days_since_last_order", dtype=Int64),
        Field(name="lifetime_value", dtype=Float32),
        Field(name="preferred_category", dtype=String),
    ],
    source=user_stats_source,
    online=True,
    tags={"team": "recommendations", "version": "v2"},
)


# --- Point-in-Time Retrieval ---
store = FeatureStore(repo_path="feature_repo/")

entity_df = pd.DataFrame(
    {
        "user_id": [42, 99, 42, 17],
        "event_timestamp": pd.to_datetime(
            [
                "2024-03-15 10:00:00",
                "2024-03-15 14:00:00",
                "2024-03-10 08:00:00",  # same user, earlier time = different features
                "2024-03-12 12:00:00",
            ]
        ),
        "label": [1, 0, 1, 0],
    }
)

# Feast handles point-in-time join automatically
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_features:order_count_30d",
        "user_features:avg_order_value_30d",
        "user_features:days_since_last_order",
    ],
).to_df()


# --- Online Serving ---
# Materialize: feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
# Or: store.materialize_incremental(end_date=datetime.utcnow())

features = store.get_online_features(
    features=[
        "user_features:order_count_30d",
        "user_features:avg_order_value_30d",
        "user_features:lifetime_value",
    ],
    entity_rows=[{"user_id": 42}],
).to_dict()
# Returns: {"user_id": [42], "order_count_30d": [5], ...}


# --- Push-Based Streaming Features ---
store.push(
    push_source_name="user_realtime_stats",
    df=pd.DataFrame(
        {
            "user_id": [42],
            "session_duration_sec": [340],
            "pages_viewed": [12],
            "event_timestamp": [datetime.utcnow()],
        }
    ),
    to=PushMode.ONLINE,  # or ONLINE_AND_OFFLINE
)


# --- Detect Training-Serving Skew ---
# online = store.get_online_features(
#     features=feature_list, entity_rows=entities,
# ).to_df()
# offline = store.get_historical_features(
#     entity_df=entity_df_now, features=feature_list,
# ).to_df()
# Assert values match within tolerance
