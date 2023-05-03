import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


df = pd.read_csv("./datasets/Tweets/Sorted/05-12-22 to 03-02-23.csv", engine="python",
                 usecols=["user_name", "user_location", "user_description", "user_created", "user_followers",
                          "user_friends", "user_favourites", "user_verified", "date", "text", "hashtags", "source",
                          "is_retweet"], sep=',')

# get available dates

df[['date', 'Time']] = df['date'].str.split(' ', 1, expand=True)


# select only the 'Date' and 'user_name' column
df = df[['date', 'user_name']]
print(pd.to_datetime(df['date']).nunique())

# print the result
print(df.head())

df_sampled = df.groupby('date').apply(lambda x: x.sample(n=20)).reset_index(drop=True)

print(df_sampled)
