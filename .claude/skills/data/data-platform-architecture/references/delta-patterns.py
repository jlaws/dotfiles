# delta_operations.py -- Delta Lake PySpark operations
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

spark = (SparkSession.builder.appName("delta-lakehouse")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate())

# --- Write initial table ---
df = spark.read.parquet("s3://raw/events/")
(df.write.format("delta")
    .partitionBy("event_date")
    .mode("overwrite")
    .save("s3://lakehouse/events"))


# --- MERGE INTO (upsert) ---
target = DeltaTable.forPath(spark, "s3://lakehouse/customers")
source = spark.read.parquet("s3://staging/customers_update/")
(target.alias("t")
    .merge(source.alias("s"), "t.customer_id = s.customer_id")
    .whenMatchedUpdate(set={
        "name": "s.name",
        "email": "s.email",
        "updated_at": "s.updated_at",
    })
    .whenNotMatchedInsert(values={
        "customer_id": "s.customer_id",
        "name": "s.name",
        "email": "s.email",
        "updated_at": "s.updated_at",
    })
    .whenNotMatchedBySourceDelete()
    .execute())


# --- OPTIMIZE, VACUUM, Change Data Feed ---
dt = DeltaTable.forPath(spark, "s3://lakehouse/events")
dt.optimize().executeCompaction()
dt.optimize().executeZOrderBy("user_id", "event_date")
dt.vacuum(retentionHours=168)  # 7 days

# Enable change data feed
spark.sql("""ALTER TABLE delta.`s3://lakehouse/events`
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

# Read change feed
changes = (spark.read.format("delta")
    .option("readChangeFeed", "true")
    .option("startingVersion", 10)
    .load("s3://lakehouse/events"))
# _change_type: insert, update_preimage, update_postimage, delete
