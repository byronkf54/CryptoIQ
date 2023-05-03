import botometer
import keys

rapidapi_key = "d96b66378bmsh3f6d8ba3786b1f2p141985jsn12962c690489"

twitter_app_auth = {
    'consumer_key': keys.twitter_API_Key,
    'consumer_secret': keys.twitter_API_Key_Secret,
    'access_token': keys.twitter_access_token,
    'access_token_secret': keys.twitter_access_token_secret,
}

bom = botometer.Botometer(rapidapi_key=rapidapi_key, **twitter_app_auth)

# Check a single account by screen name
# result = bom.check_account('@clayadavis')
# print(result)

# Check a single account by id
result = bom.check_account(1548959833)
print(result)

# Check a sequence of accounts
accounts = ['@clayadavis', '@onurvarol', '@jabawack']
# for screen_name, result in bom.check_accounts_in(accounts):
    # Do stuff with `screen_name` and `result`
