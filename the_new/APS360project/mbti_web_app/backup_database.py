"""
Database Backup Script
Export database data to JSON for sharing with team members
"""

import json
from datetime import datetime
from database import get_recent_predictions, get_prediction_statistics

def backup_database(output_file=None):
    """
    Backup database to JSON file

    Args:
        output_file: Output file path (default: backup_YYYYMMDD_HHMMSS.json)
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"database_backup_{timestamp}.json"

    # Get statistics
    stats = get_prediction_statistics()

    # Get recent predictions (all of them)
    # Note: In production, you might want to limit this
    predictions = get_recent_predictions(limit=1000)

    backup_data = {
        "backup_time": datetime.now().isoformat(),
        "statistics": stats,
        "predictions": predictions
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Database backed up to: {output_file}")
    print(f"  Total predictions: {len(predictions)}")
    return output_file

if __name__ == "__main__":
    backup_database()
