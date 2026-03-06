# great_expectations_suite.py -- Suite building and quality pipeline
from great_expectations.core import ExpectationSuite
from great_expectations.core.expectation_configuration import (
    ExpectationConfiguration,
)


def build_orders_suite() -> ExpectationSuite:
    suite = ExpectationSuite(expectation_suite_name="orders_suite")

    # Schema
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_set",
            kwargs={
                "column_set": [
                    "order_id",
                    "customer_id",
                    "amount",
                    "status",
                    "created_at",
                ],
                "exact_match": False,
            },
        )
    )
    # Primary key
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "order_id"},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "order_id"},
        )
    )
    # Categorical
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "status",
                "value_set": [
                    "pending",
                    "processing",
                    "shipped",
                    "delivered",
                    "cancelled",
                ],
            },
        )
    )
    # Numeric ranges
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "amount",
                "min_value": 0,
                "max_value": 100000,
                "strict_min": True,
            },
        )
    )
    # Row count sanity
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 1000, "max_value": 10000000},
        )
    )
    return suite


# --- Checkpoint configuration (YAML equivalent) ---
# great_expectations/checkpoints/orders_checkpoint.yml
# name: orders_checkpoint
# config_version: 1.0
# validations:
#   - batch_request:
#       datasource_name: warehouse
#       data_asset_name: orders
#     expectation_suite_name: orders_suite
# action_list:
#   - name: store_validation_result
#     action: { class_name: StoreValidationResultAction }
#   - name: update_data_docs
#     action: { class_name: UpdateDataDocsAction }
#   - name: send_slack_notification
#     action:
#       class_name: SlackNotificationAction
#       slack_webhook: ${SLACK_WEBHOOK}
#       notify_on: failure


# --- Automated Quality Pipeline ---
from dataclasses import dataclass
from typing import List, Dict, Any

import great_expectations as gx


@dataclass
class QualityResult:
    table: str
    passed: bool
    total_expectations: int
    failed_expectations: int
    details: List[Dict[str, Any]]


class DataQualityPipeline:
    def __init__(self, context: gx.DataContext):
        self.context = context

    def validate_table(self, table: str, suite: str) -> QualityResult:
        result = self.context.run_checkpoint(
            **{
                "name": f"{table}_validation",
                "config_version": 1.0,
                "class_name": "Checkpoint",
                "validations": [
                    {
                        "batch_request": {
                            "datasource_name": "warehouse",
                            "data_asset_name": table,
                        },
                        "expectation_suite_name": suite,
                    }
                ],
            }
        )
        validation_result = list(result.run_results.values())[0]
        results = validation_result.results
        failed = [r for r in results if not r.success]
        return QualityResult(
            table=table,
            passed=result.success,
            total_expectations=len(results),
            failed_expectations=len(failed),
            details=[
                {
                    "expectation": r.expectation_config.expectation_type,
                    "success": r.success,
                    "observed_value": r.result.get("observed_value"),
                }
                for r in results
            ],
        )

    def run_all(
        self,
        tables: Dict[str, str],
    ) -> Dict[str, QualityResult]:
        return {table: self.validate_table(table, suite) for table, suite in tables.items()}
