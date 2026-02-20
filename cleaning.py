import json

##############
#
# Script to remove processing results labels 
#
##############


INPUT_JSON = "data.json"
OUTPUT_JSON = "data.json"

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

for entry in data:
    pr = entry.get("processing_result")
    del entry["processing_result"]

with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=2)
