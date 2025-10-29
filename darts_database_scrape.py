import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse
import time
import re
import os
import json

BASE_URL = "https://www.dartsdatabase.co.uk/events-all.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DartPredictorBot/1.0)"}
OUT_CSV_PREFIX = "darts_results"  # Will become results_2025.csv, results_2024.csv, etc.
CHECKPOINT = "processed_eids.json"
RATE_SECONDS = 1.0   # politeness delay between requests

# Helper to save checkpoint
def load_processed():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_processed(processed_set):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(list(processed_set), f)

def get_available_years():
    """Get all available years from the dropdown menu."""
    print(f"Requesting base page to get available years: {BASE_URL}")
    r = requests.get(BASE_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    
    years = []
    # Find the year dropdown
    year_select = soup.find("select", {"name": "yearselect"})
    if year_select:
        for option in year_select.find_all("option"):
            year_value = option.get("value")
            if year_value and year_value.isdigit():
                years.append(int(year_value))
    
    # Sort years in descending order (most recent first)
    years.sort(reverse=True)
    print(f"Found available years: {years}")
    return years

def get_event_links_for_year(year):
    """Get event links for a specific year by submitting the form."""
    print(f"Requesting events for year: {year}")
    
    # Prepare form data
    form_data = {
        "yearselect": str(year)
    }
    
    # Submit POST request with the year selection
    r = requests.post(BASE_URL, data=form_data, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    # Find anchors that contain display-event.php?eid=
    for a in soup.select("a[href*='display-event.php?eid=']"):
        href = a.get("href").strip()
        # Normalize to absolute URL
        if href.startswith("/"):
            href = "https://www.dartsdatabase.co.uk" + href
        elif not href.startswith("http"):
            href = "https://www.dartsdatabase.co.uk/" + href
        links.append(href)
    
    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for l in links:
        if l not in seen:
            uniq.append(l)
            seen.add(l)
    print(f"Found {len(uniq)} event links for year {year}.")
    return uniq

# Parse a score like "6–4", "3-2", "2 - 1", maybe with sets/legs like "3-2 (sets)"
def parse_score(score_str):
    # extract first pair of integers we find
    if not score_str:
        return None, None
    # replace weird dashes with normal hyphen
    s = score_str.replace("–", "-").replace("—", "-")
    # find two numbers separated by non-digit(s)
    m = re.search(r"(\d+)\s*[-–—]\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    # fallback: try to find two numbers anywhere
    nums = re.findall(r"(\d+)", s)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None, None

def scrape_event_matches(event_url):
    """Scrape every match (with event date + round tracking) from a darts event page."""
    r = requests.get(event_url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        print(f"Skipping {event_url} (status {r.status_code})")
        return []

    soup = BeautifulSoup(r.text, "html.parser")

    # --- Event name: prefer URL param 'tna' if present, else fall back to page content ---
    parsed = urllib.parse.urlparse(event_url)
    params = urllib.parse.parse_qs(parsed.query)
    event_name = ""
    if "tna" in params and params["tna"]:
        # decode e.g. "European%20Championship" -> "European Championship"
        event_name = urllib.parse.unquote_plus(params["tna"][0])
    else:
        # fallback: check headings or bold tags
        if soup.find("h1"):
            event_name = soup.find("h1").get_text(strip=True)
        elif soup.find("h2"):
            event_name = soup.find("h2").get_text(strip=True)
        else:
            # try some bold tags near top that look like event titles
            for b in soup.find_all("b"):
                txt = b.get_text(strip=True)
                if len(txt) > 3 and not re.search(r"(?i)(round|result|score|player|winner)", txt):
                    event_name = txt
                    break
    if not event_name:
        # fallback to url if everything else fails
        event_name = event_url

    # --- Event date: from URL parameter or from page text ---
    parsed = urllib.parse.urlparse(event_url)
    params = urllib.parse.parse_qs(parsed.query)
    event_date = params.get("eda", [""])[0]  # usually present

    # fallback: try to find a date inside the page
    if not event_date:
        possible_date = soup.find(string=re.compile(r"\d{1,2}/\d{1,2}/\d{4}"))
        if possible_date:
            event_date = re.search(r"\d{1,2}/\d{1,2}/\d{4}", possible_date).group(0)

    # --- Extract matches ---
    results = []
    current_round = ""

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")

            # detect round header rows (like "Quarter Final")
            if len(tds) == 1:
                txt = tds[0].get_text(" ", strip=True)
                if re.search(r"(?i)(round|final|semi|quarter|last\s*\d+)", txt):
                    current_round = txt
                    continue

            # normal match rows (3 or 4 columns)
            if len(tds) == 4:
                round_txt = tds[0].get_text(" ", strip=True)
                player1 = tds[1].get_text(" ", strip=True)
                score = tds[2].get_text(" ", strip=True)
                player2 = tds[3].get_text(" ", strip=True)

                # if the first column looks like "Last 16" etc, treat as round
                if re.search(r"(?i)(round|final|semi|quarter|last\s*\d+)", round_txt):
                    current_round = round_txt
                elif not round_txt and current_round:
                    round_txt = current_round

            elif len(tds) == 3:
                round_txt = current_round
                player1 = tds[0].get_text(" ", strip=True)
                score = tds[1].get_text(" ", strip=True)
                player2 = tds[2].get_text(" ", strip=True)
            else:
                continue

            # Skip headers
            lower_concat = " ".join([td.get_text(strip=True).lower() for td in tds])
            if "player" in lower_concat and "score" in lower_concat:
                continue

            def split_avg(player_text):
                """Return (name, avg) if avg present, else (name, '')"""
                if not player_text:
                    return player_text, ""
                # match 1-3 digits before decimal and exactly 2 after, e.g. "9.50", "95.34", "100.20"
                m = re.search(r"(\d{1,3}\.\d{2})\b", player_text)
                if m:
                    avg = m.group(1)
                    # remove trailing whitespace and common separators/opening brackets that may precede the avg
                    name_part = player_text[:m.start()]
                    name = re.sub(r"[\s\(\[\-–—:,_]+$", "", name_part).strip()
                    return name, avg
                return player_text.strip(), ""

            player1, avg1 = split_avg(player1)
            player2, avg2 = split_avg(player2)

            p1_score, p2_score = parse_score(score)
            if p1_score is not None and p2_score is not None:
                if p1_score > p2_score:
                    winner = player1
                elif p2_score > p1_score:
                    winner = player2
                else:
                    winner = "draw"
            else:
                winner = ""

            results.append({
                "Event": event_name,
                "EventDate": event_date,
                "EventURL": event_url,
                "Round": round_txt,
                "Player1": player1,
                "Player1Avg": avg1,
                "Player2": player2,
                "Player2Avg": avg2,
                "ScoreRaw": score,
                "Player1Score": p1_score,
                "Player2Score": p2_score,
                "Winner": winner
            })

    return results

def load_existing_data_for_year(year):
    """Load existing data for a specific year if the file exists."""
    filename = f"{OUT_CSV_PREFIX}_{year}.csv"
    if os.path.exists(filename):
        try:
            df_existing = pd.read_csv(filename)
            if not df_existing.empty:
                print(f"Loaded {len(df_existing)} existing rows from {filename}.")
                return df_existing.to_dict(orient="records")
        except Exception as e:
            print(f"Could not load existing CSV {filename}: {e}")
    return []

def save_year_data(year, data):
    """Save data for a specific year to its own CSV file."""
    filename = f"{OUT_CSV_PREFIX}_{year}.csv"
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(data)} rows to {filename}")

def main():
    # Get available years
    available_years = get_available_years()
    
    # You can modify this to scrape specific years
    # For example: years_to_scrape = [2025, 2024, 2023] or years_to_scrape = available_years
    # years_to_scrape = [2025, 2024]  # Example: only scrape these years
    years_to_scrape = available_years  # Scrape all years
    
    print(f"Scraping data for years: {years_to_scrape}")
    
    processed = load_processed()
    
    # Process each year
    for year_idx, year in enumerate(years_to_scrape, 1):
        print(f"\n=== Processing year {year} ({year_idx}/{len(years_to_scrape)}) ===")
        
        # Load existing data for this specific year
        year_rows = load_existing_data_for_year(year)
        
        event_links = get_event_links_for_year(year)
        if not event_links:
            print(f"No event links found for year {year}. Saving empty file.")
            save_year_data(year, year_rows)
            continue

        total_events = len(event_links)
        events_processed = 0
        
        for event_idx, link in enumerate(event_links, 1):
            # extract eid so we can checkpoint
            m = re.search(r"eid=(\d+)", link)
            eid = m.group(1) if m else link
            if eid in processed:
                print(f"[Year {year} - {event_idx}/{total_events}] Skipping already processed event {eid}")
                continue

            print(f"[Year {year} - {event_idx}/{total_events}] Scraping event {eid}: {link}")
            try:
                matches = scrape_event_matches(link)
                print(f"  → Found {len(matches)} match rows.")
                if matches:
                    year_rows.extend(matches)
                    events_processed += 1

                # Save this year's data after each event
                save_year_data(year, year_rows)
                processed.add(eid)
                save_processed(processed)
            except Exception as e:
                print(f"  Error scraping {link}: {e}")

            time.sleep(RATE_SECONDS)
        
        print(f"Year {year} completed: {events_processed} events processed, {len(year_rows)} total match rows")

    print(f"\nDone. Results saved to separate files: {OUT_CSV_PREFIX}_{year}.csv")

if __name__ == "__main__":
    main()