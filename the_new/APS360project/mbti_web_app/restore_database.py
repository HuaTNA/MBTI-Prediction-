"""
Database Restore Script
Import database data from backup JSON file
"""

import json
import sys
from database import save_prediction

def restore_database(backup_file):
    """
    Restore database from JSON backup

    Args:
        backup_file: Path to backup JSON file
    """
    print(f"Loading backup from: {backup_file}")

    with open(backup_file, 'r', encoding='utf-8') as f:
        backup_data = json.load(f)

    predictions = backup_data.get('predictions', [])

    print(f"Found {len(predictions)} predictions to restore")
    print("⚠️  WARNING: This will add data to your existing database!")
    print("    Duplicate entries may cause errors.")

    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Restore cancelled.")
        return

    success_count = 0
    error_count = 0

    for pred in predictions:
        try:
            # Note: This is a simplified restore
            # In production, you'd want to check for duplicates first
            save_prediction(
                prediction_id=pred['prediction_id'],
                session_id=pred.get('session_id', pred['prediction_id']),
                predicted_mbti=pred['predicted_mbti'],
                dimension_scores=pred.get('dimension_scores', {}),
                text_responses=pred.get('text_responses', []),
                emotion_data=pred.get('emotion_data', []),
                language=pred.get('language', 'en'),
                question_version='1.0',
                model_version='1.0'
            )
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error restoring {pred['prediction_id'][:8]}...: {e}")

    print(f"\n✓ Restore complete!")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python restore_database.py <backup_file.json>")
        sys.exit(1)

    restore_database(sys.argv[1])
