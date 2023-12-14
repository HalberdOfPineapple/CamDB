
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := 'tpch_12.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := '-- using 1433771997 as a seed to the RNG


select
	l_shipmode,
	sum(case
		when o_orderpriority = ''1-URGENT''
			or o_orderpriority = ''2-HIGH''
			then 1
		else 0
	end) as high_line_count,
	sum(case
		when o_orderpriority <> ''1-URGENT''
			and o_orderpriority <> ''2-HIGH''
			then 1
		else 0
	end) as low_line_count
from
	orders,
	lineitem
where
	o_orderkey = l_orderkey
	and l_shipmode in (''FOB'', ''SHIP'')
	and l_commitdate < l_receiptdate
	and l_shipdate < l_commitdate
	and l_receiptdate >= date ''1994-01-01''
	and l_receiptdate < date ''1994-01-01'' + interval ''1'' year
group by
	l_shipmode
order by
	l_shipmode;

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
