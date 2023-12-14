
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := 'tpch_6.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := '-- using 1433771997 as a seed to the RNG


select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date ''1994-01-01''
	and l_shipdate < date ''1994-01-01'' + interval ''1'' year
	and l_discount between 0.08 - 0.01 and 0.08 + 0.01
	and l_quantity < 24;

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
