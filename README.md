# Cookie Monster Crawl

Looking for (cookie) recipes! 

Scope

Start with a few curated recipe websites. Then expand to discovering recipes from a seed URL and following links. 

Defining a recipe page: 
* To start, contains Recipe schema 

Relevant data
* Title
* Ingredients
* Instructions
* Source
* Website name 

Nice to have data: prep time, cook time, servings, nutrition label, pictures. 

Asynchronous behavior
* HTTP requests are concurrent
* I/O that is not blocking
* Domains are rate limited
* Retries (potentially with exponential backoff)

Constraints
* Respect robots.txt
* Can be identified by User-Agent