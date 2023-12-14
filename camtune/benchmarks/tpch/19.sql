
CREATE OR REPLACE FUNCTION get_query_results()
RETURNS TABLE(executed_query_name text, execution_time numeric) AS $$
DECLARE
    query text;
    query_name text := 'tpch_19.sql';

	start_time timestamp;
    end_time timestamp;
BEGIN
    -- Define your long query here
    query := '-- using 1433771997 as a seed to the RNG


select
	sum(l_extendedprice* (1 - l_discount)) as revenue
from
	lineitem,
	part
where
	(
		p_partkey = l_partkey
		and p_brand = ''Brand#22''
		and p_container in (''SM CASE'', ''SM BOX'', ''SM PACK'', ''SM PKG'')
		and l_quantity >= 8 and l_quantity <= 8 + 10
		and p_size between 1 and 5
		and l_shipmode in (''AIR'', ''AIR REG'')
		and l_shipinstruct = ''DELIVER IN PERSON''
	)
	or
	(
		p_partkey = l_partkey
		and p_brand = ''Brand#23''
		and p_container in (''MED BAG'', ''MED BOX'', ''MED PKG'', ''MED PACK'')
		and l_quantity >= 10 and l_quantity <= 10 + 10
		and p_size between 1 and 10
		and l_shipmode in (''AIR'', ''AIR REG'')
		and l_shipinstruct = ''DELIVER IN PERSON''
	)
	or
	(
		p_partkey = l_partkey
		and p_brand = ''Brand#12''
		and p_container in (''LG CASE'', ''LG BOX'', ''LG PACK'', ''LG PKG'')
		and l_quantity >= 24 and l_quantity <= 24 + 10
		and p_size between 1 and 15
		and l_shipmode in (''AIR'', ''AIR REG'')
		and l_shipinstruct = ''DELIVER IN PERSON''
	);

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
