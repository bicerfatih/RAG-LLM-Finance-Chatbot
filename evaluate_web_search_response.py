"""
In this script, I'll evaluate the performance of the web search response"""

from web_search_2 import web_search #Import web_search from the web_search_2 module

# Function to test the web_search function's response
def test_web_search_response(prompt, expected_keywords):
    try:
        # Call the web_search function with the given prompt
        response = web_search(prompt)
        # Check if response is None or missing 'answer' key
        if response is None or 'answer' not in response:
            print(f"Error: No valid response or 'answer' is missing for prompt'{prompt}'")
            return False
        # Convert the response to lowercase for comparison
        answer = response['answer'].lower()
        # Check if all expected keywords are present in the response
        return all(keyword in answer for keyword in expected_keywords)
    # Handle exceptions
    except KeyError:
        print(f"Error: 'answer' is missing in response for prompt '{prompt}'")
        return False
    except TypeError as e:
        print(f"TypeError for prompt '{prompt}': {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error for prompt '{prompt}': {str(e)}")
        return False

# Define test cases with prompts and expected keywords
test_cases = [
    ("Startups receiving the most venture capital in 2024", ["venture capital", "startups", "2024"]),
    ("Impact of regulatory changes on fintech startups", ["regulatory changes", "fintech", "impact"]),
    ("Startups with the fastest growth in Europe", ["startups", "growth", "Europe"]),
    ("Most common reasons for startup failures", ["startup failures", "reasons", "common"]),
    ("Startup investment trends post-COVID", ["startup", "investment", "post-covid"]),
    ("Effectiveness of government grants for startups", ["government grants", "startups", "effectiveness"]),
    ("Risk factors for startups in the technology sector", ["risk factors", "technology", "startups"]),
    ("Startups leading in green technology", ["green technology", "startups"]),
    ("How to evaluate startup financial health", ["evaluate", "startup", "financial health"]),
    ("Current inflation rate", ["inflation", "rate"]),
    ("What are the latest stock prices?", ["stock", "prices"]),
    ("GDP growth rate trends", ["gdp", "growth", "rate", "trends"]),
    ("Latest unemployment statistics in the US", ["unemployment", "statistics", "us"]),
    ("Effects of climate change on agriculture", ["climate change", "agriculture", "effects"]),
    ("Top technological innovations 2024", ["technological", "innovations", "2024"]),
    ("Who is the current CEO of Tesla?", ["ceo", "tesla"]),
    ("Recent advances in artificial intelligence", ["advances", "artificial intelligence"]),
    ("Federal reserve's next meeting date", ["federal reserve", "meeting", "date"]),
    ("2024 Presidential candidates", ["presidential", "candidates", "2024"]),
    ("Trends in global oil prices", ["oil prices", "trends", "global"])
]

# Evaluate test cases
for prompt, keywords in test_cases:
    result = test_web_search_response(prompt, keywords)
    print(f"Prompt: '{prompt}' | Pass: {result}")
