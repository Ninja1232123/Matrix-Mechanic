#!/usr/bin/env python3
"""
Legacy nightmare - 52 fixable errors in one file.

This is what inheriting a codebase feels like.
Watch it transform from broken to working in seconds.

ONLY fixable errors: KeyError, ZeroDivisionError, IndexError
NO FileNotFoundError - those stop execution before we can fix everything
"""

# Bug cluster 1: Dictionary access (KeyErrors) - 10 bugs
config = {}
db_host = config['database']  # KeyError 1
api_key = config['api_key']  # KeyError 2
port = config['port']  # KeyError 3
timeout = config['timeout']  # KeyError 4
max_conn = config['max']  # KeyError 5

settings = {}
log_level = settings['log_level']  # KeyError 6
log_file = settings['log_file']  # KeyError 7
log_format = settings['log_format']  # KeyError 8
log_rotation = settings['log_rotation']  # KeyError 9
log_size = settings['log_size']  # KeyError 10

# Bug cluster 2: Division by zero - 10 bugs
total = 100
count = 0
avg1 = total / count  # ZeroDivisionError 1
avg2 = 50 / count  # ZeroDivisionError 2
avg3 = 75 / count  # ZeroDivisionError 3
avg4 = 25 / count  # ZeroDivisionError 4
avg5 = 10 / count  # ZeroDivisionError 5

x = 10
y = 0
result1 = x / y  # ZeroDivisionError 6
result2 = x / y  # ZeroDivisionError 7
result3 = x / y  # ZeroDivisionError 8
result4 = x / y  # ZeroDivisionError 9
result5 = x / y  # ZeroDivisionError 10

# Bug cluster 3: Index errors - 10 bugs
items = []
val1 = items[0]  # IndexError 1
val2 = items[1]  # IndexError 2
val3 = items[2]  # IndexError 3
val4 = items[3]  # IndexError 4
val5 = items[4]  # IndexError 5

data = []
item1 = data[0]  # IndexError 6
item2 = data[1]  # IndexError 7
item3 = data[2]  # IndexError 8
item4 = data[3]  # IndexError 9
item5 = data[4]  # IndexError 10

# Bug cluster 4: More KeyErrors - 10 bugs
users = {}
user1 = users['alice']  # KeyError 11
user2 = users['bob']  # KeyError 12
user3 = users['charlie']  # KeyError 13
user4 = users['david']  # KeyError 14
user5 = users['eve']  # KeyError 15

cache = {}
cached1 = cache['key1']  # KeyError 16
cached2 = cache['key2']  # KeyError 17
cached3 = cache['key3']  # KeyError 18
cached4 = cache['key4']  # KeyError 19
cached5 = cache['key5']  # KeyError 20

# Bug cluster 5: More ZeroDivisionErrors - 10 bugs
numerator = 100
denominator = 0
calc1 = numerator / denominator  # ZeroDivisionError 11
calc2 = numerator / denominator  # ZeroDivisionError 12
calc3 = numerator / denominator  # ZeroDivisionError 13
calc4 = numerator / denominator  # ZeroDivisionError 14
calc5 = numerator / denominator  # ZeroDivisionError 15

a = 50
b = 0
div1 = a / b  # ZeroDivisionError 16
div2 = a / b  # ZeroDivisionError 17
div3 = a / b  # ZeroDivisionError 18
div4 = a / b  # ZeroDivisionError 19
div5 = a / b  # ZeroDivisionError 20

# Bug cluster 6: More IndexErrors - 2 bugs
empty_list = []
elem1 = empty_list[0]  # IndexError 11
elem2 = empty_list[1]  # IndexError 12

print("If you can read this, all 52 errors were fixed!")
