# Set paths
path_repo="C:\GitProjects\EmployeeAttrition"
path_log="C:\GitProjects\EmployeeAttrition\logs\run_$(date +%Y%m%d).log"

# Show status
echo "Running Script..." >> "$path_log" 2>&1

# Run script
python "$path_repo\employeeattrition.py" >> "$path_log" 2>&1

# Show status
echo "Run Finished!" >> "$path_log" 2>&1
