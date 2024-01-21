import subprocess

# Run 'pip freeze' command to get a list of installed packages
result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)

# Check if the command was successful
if result.returncode == 0:
    # Write the list of packages to requirements.txt
    with open('requirements.txt', 'w') as requirements_file:
        requirements_file.write(result.stdout)
        print("requirements.txt generated successfully.")
else:
    print("Error generating requirements.txt:", result.stderr)
