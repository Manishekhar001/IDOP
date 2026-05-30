"""
Central schema registry for the IDOP e-commerce database.

All table names and column definitions are defined once here and imported
by column_mapper, vanna_service, and other components that need schema info.
Add new tables here to make them available across the platform.
"""

# Supported mutation tables and their column definitions
# Extend this dict when adding new tables to the mutation pipeline.
TABLE_SCHEMAS = {
    "customers": [
        "id",
        "name",
        "email",
        "segment",
        "country",
        "created_at",
        "updated_at",
    ],
    "products": [
        "id",
        "name",
        "category",
        "price",
        "stock_quantity",
        "description",
        "created_at",
        "updated_at",
    ],
    "orders": [
        "id",
        "customer_id",
        "order_date",
        "total_amount",
        "status",
        "shipping_address",
        "created_at",
        "updated_at",
    ],
}

# Set of all supported table names for quick lookup
SUPPORTED_MUTATION_TABLES: set[str] = set(TABLE_SCHEMAS.keys())
