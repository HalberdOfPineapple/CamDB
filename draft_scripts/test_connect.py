from camtune.database import PostgresqlConnector


# Define the query
query = """
SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM
    lineitem
WHERE
    l_shipdate <= DATE '1998-12-01' - INTERVAL '1 day'
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;
"""

conn_params = {
    'host':'localhost', 
    'port':'5432',
    'user':'viktor',
    'passwd':'741286', 
    'name':'tpch',
}

db_connector = PostgresqlConnector(**conn_params)
results = db_connector.fetch_results(sql=query)
db_connector.close_db()

print(results)