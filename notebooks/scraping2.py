from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import csv
import time

# Configure Chrome options
options = Options()
options.add_argument('--headless')  # Run headless if you don't want a GUI
options.add_argument('--no-sandbox')  # Required for some environments
options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

# Start the WebDriver
driver = webdriver.Chrome(options=options)

# List of App IDs to scrape
app_ids = ff["AppID"].tolist()  # Ensure 'q' is defined in your context and contains App IDs

# Open the CSV file in write mode
with open('steam_app20.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write header to the CSV file
    writer.writerow([
        'App ID', 'Description', 'Developer', 'Publisher', 'Genre', 'Category',
        'Release Date', 'Owners', 'Followers', 'Peak Concurrent Players', 'YouTube Stats'
    ])

    try:
        for app_id in app_ids:
            url = f'https://steamspy.com/app/{app_id}'
            driver.get(url)

            # Wait for the page to load completely
            time.sleep(2)  # Adjust as needed

            # Extract description
            try:
                game_description = driver.find_element("css selector", "p").text
            except Exception as e:
                print(f"Error fetching description for App ID {app_id}: {e}")
                game_description = "Description not found"

            # Extract 'Developer', 'Publisher', 'Genre', etc.
            try:
                developer = driver.find_element("xpath", "//strong[text()='Developer:']/following-sibling::a").text
            except Exception as e:
                developer = "Developer not found"

            try:
                publisher = driver.find_element("xpath", "//strong[text()='Publisher:']/following-sibling::a").text
            except Exception as e:
                publisher = "Publisher not found"

            try:
                genres = driver.find_elements("xpath", "//strong[text()='Genre:']/following-sibling::a")
                genre_text = ", ".join([genre.text for genre in genres])
            except Exception as e:
                genre_text = "Genre not found"

            # Extract additional fields
            try:
                category = driver.find_element("xpath", "//strong[text()='Category:']/following-sibling::text()").text
            except Exception as e:
                category = "Category not found"

            try:
                release_date = driver.find_element("xpath", "//strong[text()='Release date']/following-sibling::text()").text
            except Exception as e:
                release_date = "Release date not found"

            try:
                owners = driver.find_element("xpath", "//strong[text()='Owners']/following-sibling::text()").text
            except Exception as e:
                owners = "Owners not found"

            try:
                followers = driver.find_element("xpath", "//strong[text()='Followers']/following-sibling::text()").text
            except Exception as e:
                followers = "Followers not found"

            try:
                peak_players = driver.find_element("xpath", "//strong[text()='Peak concurrent players yesterday']/following-sibling::text()").text
            except Exception as e:
                peak_players = "Peak concurrent players not found"

            try:
                youtube_stats = driver.find_element("xpath", "//strong[text()='YouTube stats']/following-sibling::text()").text
            except Exception as e:
                youtube_stats = "YouTube stats not found"

            # Write the extracted data to the CSV file
            writer.writerow([
                app_id, game_description, developer, publisher, genre_text, category,
                release_date, owners, followers, peak_players, youtube_stats
            ])

    finally:
        # Close the driver
        driver.quit()
