import csv
import random
from datetime import datetime, timedelta

# First and last names for generating random names
first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", 
               "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
               "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
               "Matthew", "Betty", "Anthony", "Helen", "Mark", "Sandra", "Donald", "Donna",
               "Steven", "Carol", "Paul", "Ruth", "Andrew", "Sharon", "Kenneth", "Michelle"]

last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
              "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
              "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores"]

def generate_name():
    """Generate a random full name"""
    return f"{random.choice(first_names)} {random.choice(last_names)}"

# Predefined company names
companies = [
    "CloudSync Pro", "DataFlow Analytics", "SecureVault SaaS", "TeamCollab Plus",
    "AutoScale Solutions", "MetricsDash", "APIGateway Pro", "WorkflowEngine",
    "CustomerHub 360", "DevOps Pipeline"
]

# Review templates for different aspects
positive_templates = [
    "The {aspect} is excellent, really impressed with how {detail}.",
    "Love the {aspect}, it's {detail} and worth every penny.",
    "{aspect} exceeded our expectations, particularly {detail}.",
    "Outstanding {aspect}, the team finds it {detail}.",
    "The {aspect} has been a game-changer for us, especially {detail}."
]

negative_templates = [
    "Disappointed with the {aspect}, it's {detail} which causes issues.",
    "The {aspect} needs improvement, {detail} is a major concern.",
    "Struggling with {aspect}, finding it {detail} for our needs.",
    "{aspect} is below average, {detail} makes it hard to recommend.",
    "Having problems with {aspect}, it's {detail} and frustrating."
]

neutral_templates = [
    "The {aspect} is decent but {detail}.",
    "{aspect} works as expected, though {detail} could be better.",
    "Mixed feelings about {aspect}, while good overall, {detail}.",
    "The {aspect} is functional, however {detail} needs attention.",
    "{aspect} meets basic needs but {detail} is noticeable."
]

aspects = {
    "performance": ["blazing fast", "incredibly slow", "responsive", "laggy", "optimized well", "resource-heavy"],
    "pricing": ["very affordable", "overpriced", "competitive", "expensive for what you get", "great value", "not worth the cost"],
    "customer service": ["responsive and helpful", "slow to respond", "knowledgeable", "unhelpful", "proactive", "difficult to reach"],
    "usability": ["intuitive and user-friendly", "confusing interface", "easy to navigate", "steep learning curve", "well-designed", "clunky and outdated"],
    "features": ["comprehensive and powerful", "lacking essential functions", "innovative", "basic and limited", "constantly improving", "buggy and unreliable"],
    "uptime": ["rock-solid reliability", "frequent downtime", "99.9% availability", "unstable during peak hours", "dependable", "unreliable service"]
}

def generate_review():
    """Generate a single review with weighted sentiment"""
    sentiment_weight = random.random()
    
    # 50% positive, 30% negative, 20% neutral
    if sentiment_weight < 0.5:
        template = random.choice(positive_templates)
        sentiment = "positive"
    elif sentiment_weight < 0.8:
        template = random.choice(negative_templates)
        sentiment = "negative"
    else:
        template = random.choice(neutral_templates)
        sentiment = "neutral"
    
    aspect = random.choice(list(aspects.keys()))
    
    if sentiment == "positive":
        detail = random.choice([d for d in aspects[aspect] if any(word in d for word in ["fast", "affordable", "helpful", "friendly", "innovative", "reliable", "solid", "well", "great", "responsive", "proactive", "dependable", "competitive", "comprehensive", "powerful", "improving", "optimized"])])
    elif sentiment == "negative":
        detail = random.choice([d for d in aspects[aspect] if any(word in d for word in ["slow", "expensive", "unhelpful", "confusing", "lacking", "frequent", "difficult", "steep", "clunky", "basic", "limited", "buggy", "unreliable", "unstable", "laggy", "resource-heavy", "overpriced"])])
    else:
        detail = random.choice(aspects[aspect])
    
    return template.format(aspect=aspect, detail=detail)

def generate_date_with_bias():
    """Generate a date between 2022-01-01 and 2025-03-01 with bias towards last 3 months"""
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 3, 1)
    recent_start = end_date - timedelta(days=90)
    
    # 60% chance for last 3 months
    if random.random() < 0.6:
        days_diff = (end_date - recent_start).days
        random_days = random.randint(0, days_diff)
        return recent_start + timedelta(days=random_days)
    else:
        days_diff = (recent_start - start_date).days
        random_days = random.randint(0, days_diff)
        return start_date + timedelta(days=random_days)

# Generate data
data = []
for _ in range(100):
    row = {
        "Name": generate_name(),
        "Company": random.choice(companies),
        "Date": generate_date_with_bias().strftime("%Y-%m-%d"),
        "Comments": generate_review()
    }
    data.append(row)

# Write to CSV
with open("test.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Name", "Company", "Date", "Comments"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"Successfully created test.csv with {len(data)} rows")
print(f"\nSample of the data:")
for i, row in enumerate(data[:5]):
    print(f"Row {i+1}: {row['Name']} | {row['Company']} | {row['Date']}")
    print(f"  Comments: {row['Comments']}")
    print()

# Show date distribution
from collections import Counter
date_months = [row['Date'][:7] for row in data]
month_counts = Counter(date_months)
print(f"\nTop 10 months by review count:")
for month, count in month_counts.most_common(10):
    print(f"{month}: {count} reviews")