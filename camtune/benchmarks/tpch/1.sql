CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := 'tpch_1.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := '
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
			l_shipdate <= DATE ''1998-12-01'' - INTERVAL ''1 day''
		GROUP BY
			l_returnflag,
			l_linestatus
		ORDER BY
			l_returnflag,
			l_linestatus;
    ';

	-- Start the timer
    start_time := clock_timestamp();

	-- Execute the query
    EXECUTE query;

    -- End the timer
    end_time := clock_timestamp();

    -- Call the function with the query name and the query
    RETURN QUERY SELECT query_name, EXTRACT(EPOCH FROM (end_time - start_time));
END;
$$ LANGUAGE plpgsql;

SELECT * FROM get_query_results();