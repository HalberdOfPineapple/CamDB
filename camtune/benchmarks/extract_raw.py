import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tpch_dir = os.path.join(BASE_DIR, 'tpch')
tpch_raw_dir = os.path.join(BASE_DIR, 'tpch_raw')
tpch_exp_dir = os.path.join(BASE_DIR, 'tpch_explain')


def extract_sql(query_for_time: str):
    start_token = "query := '"
    end_token = "';"

    start_index = query_for_time.find(start_token) + len(start_token)
    end_index = query_for_time.find(end_token, start_index)
    extracted_query = query_for_time[start_index:end_index]

    # Correcting for escaped single quotes within the SQL query
    extracted_query_corrected = extracted_query.replace("''", "'")

    return extracted_query_corrected

def main():
    # get the list of .sql files in directory 'tpch'
    query_files = [f for f in os.listdir(tpch_dir) if f.endswith('.sql')]

    # extract the SQL queries from the .sql files
    for query_file in query_files:
        with open(os.path.join(tpch_dir, query_file), 'r') as f:
            query_for_time = f.read()
            extracted_query = extract_sql(query_for_time)
        # Write the extracted SQL queries to a new file with the same name while in a new directory 'tpch_raw'
        # with open(os.path.join(tpch_raw_dir, query_file), 'w') as f:
        #     f.write(extracted_query)
        
        with open(os.path.join(tpch_exp_dir, query_file), 'w') as f:
            f.write('EXPLAIN ' + extracted_query)


if __name__ == '__main__':
    main()