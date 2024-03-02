import subprocess

SYSBENCH_SCRIPTS_DIR = "/usr/local/share/sysbench/"
SYSBENCH_WORKLOADS = {
    'read_only': {
        'script': 'oltp_read_only.lua',
        'tables': 150,
        'table_size': 800000,
        'time': 30
    }, 
    'write_only': {
        'script': 'oltp_write_only.lua',
        'tables': 150,
        'table_size': 800000,
        'time': 30
    },
    'read_write': {
        'script': 'oltp_read_write.lua',
        'tables': 150,
        'table_size': 800000,
        'time': 30
    },
    'read_only_test': {
        'script': 'oltp_read_only.lua',
        'tables': 10,
        'table_size': 2000,
        'time': 15,
    },
    'write_only_test': {
        'script': 'oltp_write_only.lua',
        'tables': 10,
        'table_size': 2000,
        'time': 15,
    },
    'read_write_test': {
        'script': 'oltp_read_write.lua',
        'tables': 10,
        'table_size': 2000,
        'time': 5,
    },
}


def parse_sysbench_output(result: subprocess.CompletedProcess):
    stats_output = result.stdout.split("Threads started!")[1]
    stats_dict = {}
    for line in stats_output.split('\n'):
        if ':' in line:

            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            if key == 'events/s_(eps)':
                key = 'throughput'

            value = value.strip()
            # Handle numeric values
            try:
                if '.' in value:
                    stats_dict[key] = float(value)
                else:
                    stats_dict[key] = int(value)
            except ValueError:
                # Handle non-numeric values (e.g., strings)
                stats_dict[key] = value
    return stats_dict