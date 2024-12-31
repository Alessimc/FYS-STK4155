import subprocess

# Run the first script
#subprocess.run(["python", "gradient_boost_gridsearch.py"])

# Run the second script
subprocess.run(["python", "xgboost_gridsearch.py"])


import subprocess

def run_script_with_logging(script_name, log_file_name):
    with open(log_file_name, "w") as log_file:
        # Run the script and capture its output
        process = subprocess.Popen(
            ["python", script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Stream the output to both terminal and file
        for line in process.stdout:
            print(line, end="")  # Print to terminal
            log_file.write(line)  # Write to log file
        for line in process.stderr:
            print(line, end="")  # Print errors to terminal
            log_file.write(line)  # Write errors to log file
        process.wait()

# Run the scripts with logging
run_script_with_logging("gradient_boost_gridsearch.py", "gradient_boost_gridsearch_output.log")
run_script_with_logging("xgboost_gridsearch.py", "xgboost_gridsearch_output.log")

