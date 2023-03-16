import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Define the search term and the start and end dates for the search
search_term = "svb"
start_date = datetime(2023, 3, 13)
end_date = datetime(2023, 3, 14)

# Open a file for writing the output
with open("output.txt", "w") as file:
    # Loop through each day between the start and end dates
    current_date = start_date
    while current_date <= end_date:
        # Construct the search URL for the current date and search term
        search_url = f"https://www.google.com/search?q={search_term}&tbs=cdr:1,cd_min:{current_date.strftime('%m/%d/%Y')},cd_max:{current_date.strftime('%m/%d/%Y')}"

        # Use requests and BeautifulSoup to get the search results page HTML
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Count the number of search results on the page
        num_results = len(soup.find_all("div", class="MjjYud"))

        # Write the date and number of search results to the file
        file.write(
            f"{current_date.strftime('%m/%d/%Y')}: {num_results} articles found\n")

        # Increment the current date by one day
        current_date += timedelta(days=1)
