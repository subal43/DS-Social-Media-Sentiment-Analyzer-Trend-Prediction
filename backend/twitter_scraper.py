import snscrape.modules.twitter as sntwitter


def scrape_tweets(username, count=10):
    tweets = []
    for i , tweet in enumerate(
        sntwitter.TwitterSearchScraper(f"from:{username}").get_items()
    ):
        if i >= count : 
            break
        tweets.append(tweet.content)
    return tweets

print(scrape_tweets("subal64780", 5))
    

        
