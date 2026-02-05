"""
Database Initialization Script for MBTI Prediction System
Creates tables and imports initial question data

Usage:
    python init_database.py --db postgresql://user:pass@localhost/mbti_db
    python init_database.py --db sqlite:///mbti.db  # For testing
"""

import argparse
import json
import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def read_sql_schema(schema_file):
    """Read SQL schema from file"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        return f.read()


def execute_schema(engine, schema_sql):
    """Execute schema creation SQL"""
    print("Creating database schema...")

    # Split by semicolon and execute each statement
    statements = [s.strip() for s in schema_sql.split(';') if s.strip()]

    with engine.connect() as conn:
        for i, statement in enumerate(statements, 1):
            try:
                # Skip comments
                if statement.startswith('--'):
                    continue

                conn.execute(text(statement))
                conn.commit()
                print(f"  ✅ Statement {i}/{len(statements)} executed")

            except SQLAlchemyError as e:
                # Some statements might fail if tables already exist
                if 'already exists' in str(e).lower():
                    print(f"  ⚠️  Statement {i}: Table/View already exists, skipping")
                else:
                    print(f"  ❌ Error in statement {i}: {e}")
                    raise

    print("✅ Schema created successfully!")


def import_questions(engine, questions_file, language):
    """Import questions from JSON file to database"""
    print(f"\nImporting questions from {questions_file} (language={language})...")

    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get('questions', [])

    with engine.connect() as conn:
        imported = 0
        skipped = 0

        for q in questions:
            try:
                # Check if question already exists
                result = conn.execute(
                    text("SELECT COUNT(*) FROM question_bank WHERE question_id = :qid AND language = :lang"),
                    {"qid": q['id'], "lang": language}
                )
                exists = result.scalar() > 0

                if exists:
                    skipped += 1
                    continue

                # Insert question
                conn.execute(
                    text("""
                        INSERT INTO question_bank
                        (question_id, title, description, dimension, version, language, active)
                        VALUES
                        (:qid, :title, :desc, :dim, :ver, :lang, :active)
                    """),
                    {
                        "qid": q['id'],
                        "title": q['question'],
                        "desc": q['description'],
                        "dim": q['dimension'],
                        "ver": q.get('version', '1.0'),
                        "lang": language,
                        "active": q.get('active', True)
                    }
                )
                conn.commit()
                imported += 1

            except SQLAlchemyError as e:
                print(f"  ❌ Error importing question {q['id']}: {e}")

        print(f"  ✅ Imported {imported} questions")
        if skipped > 0:
            print(f"  ⚠️  Skipped {skipped} existing questions")


def verify_database(engine):
    """Verify database setup"""
    print("\nVerifying database setup...")

    with engine.connect() as conn:
        # Check tables exist
        tables = ['users', 'predictions', 'question_bank', 'question_usage', 'user_feedback']

        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"  ✅ Table '{table}' exists ({count} rows)")
            except SQLAlchemyError as e:
                print(f"  ❌ Table '{table}' missing or error: {e}")

        # Check question count
        result = conn.execute(text("SELECT language, COUNT(*) FROM question_bank GROUP BY language"))
        for lang, count in result:
            print(f"  ✅ Language '{lang}': {count} questions loaded")


def main():
    parser = argparse.ArgumentParser(description='Initialize MBTI Prediction Database')
    parser.add_argument(
        '--db',
        default='postgresql://localhost/mbti_db',
        help='Database connection string (default: postgresql://localhost/mbti_db)'
    )
    parser.add_argument(
        '--schema',
        default='database_schema_proposal.sql',
        help='SQL schema file path'
    )
    parser.add_argument(
        '--questions-dir',
        default='mbti_web_app/data',
        help='Directory containing question JSON files'
    )
    parser.add_argument(
        '--skip-schema',
        action='store_true',
        help='Skip schema creation (only import questions)'
    )
    parser.add_argument(
        '--skip-questions',
        action='store_true',
        help='Skip question import (only create schema)'
    )

    args = parser.parse_args()

    print("="*60)
    print("MBTI Prediction System - Database Initialization")
    print("="*60)
    print(f"Database: {args.db}")
    print(f"Schema file: {args.schema}")
    print(f"Questions directory: {args.questions_dir}")
    print("="*60)

    try:
        # Create database engine
        print("\nConnecting to database...")
        engine = create_engine(args.db, echo=False)

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")

        # Create schema
        if not args.skip_schema:
            if not os.path.exists(args.schema):
                print(f"❌ Schema file not found: {args.schema}")
                sys.exit(1)

            schema_sql = read_sql_schema(args.schema)
            execute_schema(engine, schema_sql)

        # Import questions
        if not args.skip_questions:
            # Import English questions
            en_file = os.path.join(args.questions_dir, 'questions_en.json')
            if os.path.exists(en_file):
                import_questions(engine, en_file, 'en')
            else:
                print(f"⚠️  English questions file not found: {en_file}")

            # Import Chinese questions
            zh_file = os.path.join(args.questions_dir, 'questions_zh.json')
            if os.path.exists(zh_file):
                import_questions(engine, zh_file, 'zh')
            else:
                print(f"⚠️  Chinese questions file not found: {zh_file}")

        # Verify setup
        verify_database(engine)

        print("\n" + "="*60)
        print("✅ Database initialization completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Update app.py with database connection string")
        print("2. Test the API endpoints")
        print("3. Start collecting user data!")

    except SQLAlchemyError as e:
        print(f"\n❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
