"""
Database Setup and Initialization Script
"""

import os
from database import init_database, get_prediction_statistics

print("=" * 60)
print("PostgreSQL Database Initialization")
print("=" * 60)

# Step 1: Check environment variables
database_url = os.getenv('DATABASE_URL')
if database_url:
    print(f"Using DATABASE_URL from environment variables")
else:
    print("WARNING: DATABASE_URL environment variable not set")
    print("   Using default: postgresql://postgres:password@localhost:5432/mbti_predictions")
    print()
    print("   How to set:")
    print("   PowerShell: $env:DATABASE_URL='your_database_url'")
    print("   Linux/Mac: export DATABASE_URL='your_database_url'")
    print()

# Step 2: Initialize database
print("\nCreating database tables...")
if init_database():
    print("Database initialization successful!")
    print()
    print("Tables created:")
    print("  - prediction_data (prediction records)")
    print("  - question_usage (question usage logs)")
    print("  - user_sessions (user sessions)")
else:
    print("ERROR: Database initialization failed")
    print()
    print("Please check:")
    print("  1. PostgreSQL is installed and running")
    print("  2. Database 'mbti_predictions' exists")
    print("  3. Username and password are correct")
    print()
    print("Create database command (PostgreSQL):")
    print("  psql -U postgres")
    print("  CREATE DATABASE mbti_predictions;")
    exit(1)

# Step 3: Test connection
print("\nTesting database connection...")
try:
    stats = get_prediction_statistics()
    if stats is not None:
        print("Database connection successful!")
        print(f"\nCurrent database statistics:")
        print(f"  Total predictions: {stats['total_predictions']}")
        if stats['most_common']:
            print(f"  Most common type: {stats['most_common']}")
    else:
        print("WARNING: Database connected but query failed")
except Exception as e:
    print(f"ERROR: Database connection failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Database setup complete!")
print("=" * 60)
print("\nYou can now run: python app.py")
