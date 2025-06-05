from datetime import datetime
import sys

format = "%H:%M:%S.%f"
def clean(string):
    return string.replace("UTC ", "")[:-3]

start_time = datetime.strptime(clean(sys.argv[1]), format)
end_time = datetime.strptime(clean(sys.argv[2]), format)

diff = end_time - start_time
print(diff)
print(diff.total_seconds() / 24)
