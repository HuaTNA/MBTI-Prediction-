"""
Quick script to verify cloud database contents
"""
from database import engine
from sqlalchemy import text

with engine.connect() as conn:
    # Check prediction_data
    result = conn.execute(text("SELECT COUNT(*) FROM prediction_data"))
    pred_count = result.fetchone()[0]
    print(f"Predictions in cloud database: {pred_count}")

    # Check question_usage
    result = conn.execute(text("SELECT COUNT(*) FROM question_usage"))
    usage_count = result.fetchone()[0]
    print(f"Question usage records: {usage_count}")

    # Show question details
    if usage_count > 0:
        result = conn.execute(text("""
            SELECT question_id, question_order,
                   length(user_response_text) as text_length
            FROM question_usage
            ORDER BY question_order
            LIMIT 10
        """))
        print("\nQuestion responses:")
        for row in result:
            print(f"  Question {row[0]} (Order #{row[1]}): {row[2]} characters")

    print(f"\nCloud database test: SUCCESS!")
