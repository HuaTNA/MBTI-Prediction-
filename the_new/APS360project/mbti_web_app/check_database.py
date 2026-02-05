"""
Quick script to check database contents
Run this after completing a test
"""

from database import SessionLocal, PredictionData, QuestionUsage, UserSession

def check_database():
    db = SessionLocal()

    try:
        # Check prediction_data table
        predictions = db.query(PredictionData).all()
        print(f"\n=== prediction_data table ===")
        print(f"Total records: {len(predictions)}")

        if predictions:
            print("\nRecent predictions:")
            for p in predictions[-5:]:  # Show last 5
                print(f"  - {p.prediction_id[:8]}... | {p.predicted_mbti} | {p.timestamp}")
                print(f"    Text responses: {len(p.text_responses)} questions")
                print(f"    Emotion data: {len(p.emotion_data)} points")
        else:
            print("  ❌ No predictions found!")

        # Check question_usage table
        usages = db.query(QuestionUsage).all()
        print(f"\n=== question_usage table ===")
        print(f"Total records: {len(usages)}")

        if usages:
            print("\nRecent usages:")
            for u in usages[-5:]:  # Show last 5
                print(f"  - Q{u.question_id} | Order: {u.question_order} | Response length: {u.response_length}")
        else:
            print("  ❌ No question usages found!")

        # Check user_sessions table
        sessions = db.query(UserSession).all()
        print(f"\n=== user_sessions table ===")
        print(f"Total records: {len(sessions)}")

        if sessions:
            print("\nRecent sessions:")
            for s in sessions[-5:]:  # Show last 5
                status = "✓ Completed" if s.completed else "⏳ In progress"
                print(f"  - {s.session_id[:8]}... | {status} | {s.start_time}")
        else:
            print("  ❌ No sessions found!")

    except Exception as e:
        print(f"\n❌ Error checking database: {e}")
        print("\nMake sure your database connection is configured correctly.")
        print("Check DATABASE_URL in database.py")

    finally:
        db.close()

if __name__ == "__main__":
    check_database()
