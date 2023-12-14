import psycopg2
import time
from dbconnector import DBConnector, PostgresqlConnector

# Define connection parameters
params = {
    'database': 'tpch',
    'user': 'viktor',
    'password': '741286',  # Replace with the actual password
    'host': 'localhost',  # Replace with the actual host name or IP address
    'port': '5432'  # Replace with the actual port if not the default
}

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

def main():
    db_connector = PostgresqlConnector(
        host='localhost', port='5432',
        user='viktor', passwd='741286', name='tpch',
    )

    results = db_connector.fetch_results(sql=query)
    print(results)

    db_connector.close_db()

if __name__ == '__main__':
    main()

# # Connect to the database
# conn = psycopg2.connect(**params)
# cur = conn.cursor()

# # Time to run the query for throughput measurement
# throughput_time = 20  # seconds
# end_time = time.time() + throughput_time
# query_count = 0

# # Throughput measurement loop
# while time.time() < end_time:
#     cur.execute(query)
#     # Optionally fetch results if you need to process them
#     # results = cur.fetchall()
#     query_count += 1

# # Close the cursor and the connection
# cur.close()
# conn.close()

# # Calculate throughput as queries per second
# throughput = query_count / throughput_time

# # Print the throughput
# print(f"Throughput: {throughput} queries per second")
