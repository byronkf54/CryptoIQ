import botometer
import pandas as pd
import csv
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
i_range = 187
users = set()
# ROW 9849 to ...
# 670258
# 187-223 = 125000 users
df = pd.read_csv(f"../datasets/Tweets/Sorted/05-12-22 to 03-02-23.csv")
print(df.head())
print(df.tail())

print(f"No. ROWS: {len(df.index)}")
total_rows = len(df.index)
# print("non-null usernames")
total_users = df['user_name'].count()
users.update(df['user_name'].tolist())

print(total_rows)
print(total_users)
print(len(users))