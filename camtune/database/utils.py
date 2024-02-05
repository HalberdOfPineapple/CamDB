import json
import subprocess

def run_as_user(command, user, password):
    # Prepend the 'sudo' and '-u' options to run the command as the specified user
    sudo_command = ['sudo', '-u', user, '-S'] + command.split()
    sudo_command = f"echo {password} | {' '.join(sudo_command)}"
        
    result = subprocess.run(sudo_command, shell=True,
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result


def initialize_knobs(knobs_config, num) -> dict:
    if num == -1:
        with open(knobs_config, 'r') as f:
            knob_details = json.load(f)
    else:
        with open(knobs_config, 'r') as f:
            knob_tmp = json.load(f)
            i = 0

            knob_details = {}
            knob_names = list(knob_tmp.keys())
            while i < num:
                key = knob_names[i]
                knob_details[key] = knob_tmp[key]
                i = i + 1

    return knob_details
