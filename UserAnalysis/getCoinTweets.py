import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


df = pd.read_csv("./datasets/Tweets/Sorted/05-12-22 to 03-02-23.csv", engine="python",
                 usecols=["user_name", "user_location", "user_description", "user_created", "user_followers",
                          "user_friends", "user_favourites", "user_verified", "date", "text", "hashtags", "source",
                          "is_retweet"], sep=',')
print(len(df))
exit()
print(df.head())
print(df.info())
df_no_missing = df.dropna(subset=['text'])
BTC = df_no_missing[df_no_missing['text'].str.contains("Bitcoin|BTC")]
ETH = df_no_missing[df_no_missing['text'].str.contains("Ethereum|ETH")]

print("### BTC ###")
print(BTC.info())
print(BTC.head())
print(len(BTC))  # 77439

print("### ETH ###")
print(ETH.info())
print(ETH.head())
print(len(ETH))  # 28434
