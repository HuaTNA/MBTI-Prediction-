"""
MBTI Question Bank Management
Manages MBTI test questions with database support
"""

from datetime import datetime
from typing import List, Dict, Optional
import json
import random


class Question:
    """Single MBTI question"""

    def __init__(self,
                 question_id: int,
                 title: str,
                 description: str,
                 dimension: str,
                 version: str = "1.0",
                 language: str = "en",
                 active: bool = True):
        """
        Initialize a question

        Args:
            question_id: Unique question identifier
            title: Short question title
            description: Full question description
            dimension: MBTI dimension (I/E, S/N, T/F, J/P)
            version: Question version for A/B testing
            language: Question language (en, zh, etc.)
            active: Whether question is currently active
        """
        self.question_id = question_id
        self.title = title
        self.description = description
        self.dimension = dimension
        self.version = version
        self.language = language
        self.active = active

    def to_dict(self) -> Dict:
        """Convert question to dictionary"""
        return {
            'id': self.question_id,
            'question': self.title,
            'description': self.description,
            'dimension': self.dimension,
            'version': self.version,
            'language': self.language
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Question':
        """Create question from dictionary"""
        return cls(
            question_id=data['id'],
            title=data['question'],
            description=data['description'],
            dimension=data['dimension'],
            version=data.get('version', '1.0'),
            language=data.get('language', 'en'),
            active=data.get('active', True)
        )


class QuestionBank:
    """Question bank manager with support for multiple languages and versions"""

    def __init__(self, questions_file: Optional[str] = None):
        """
        Initialize question bank

        Args:
            questions_file: Path to JSON file containing questions
        """
        self.questions: List[Question] = []
        self.questions_by_id: Dict[int, Question] = {}
        self.questions_by_dimension: Dict[str, List[Question]] = {
            'I/E': [],
            'S/N': [],
            'T/F': [],
            'J/P': []
        }

        if questions_file:
            self.load_from_file(questions_file)

    def add_question(self, question: Question):
        """Add a question to the bank"""
        self.questions.append(question)
        self.questions_by_id[question.question_id] = question

        if question.dimension in self.questions_by_dimension:
            self.questions_by_dimension[question.dimension].append(question)

    def get_question(self, question_id: int) -> Optional[Question]:
        """Get question by ID"""
        return self.questions_by_id.get(question_id)

    def get_questions_by_dimension(self, dimension: str,
                                   language: str = 'en',
                                   active_only: bool = True) -> List[Question]:
        """Get all questions for a specific dimension"""
        questions = self.questions_by_dimension.get(dimension, [])

        if active_only:
            questions = [q for q in questions if q.active]

        if language:
            questions = [q for q in questions if q.language == language]

        return questions

    def get_random_questions(self,
                           count: int = 3,
                           language: str = 'en',
                           balanced: bool = True) -> List[Question]:
        """
        Get random questions for a test session

        Args:
            count: Number of questions to select
            language: Question language
            balanced: If True, ensure balanced coverage of all dimensions

        Returns:
            List of randomly selected questions
        """
        active_questions = [q for q in self.questions
                          if q.active and q.language == language]

        if not balanced:
            return random.sample(active_questions, min(count, len(active_questions)))

        # Balanced selection: ensure coverage of all dimensions
        dimensions = ['I/E', 'S/N', 'T/F', 'J/P']
        selected = []
        questions_per_dim = count // len(dimensions)
        remainder = count % len(dimensions)

        for i, dim in enumerate(dimensions):
            dim_questions = self.get_questions_by_dimension(dim, language)
            # Add extra question to first dimensions if count not divisible by 4
            num_to_select = questions_per_dim + (1 if i < remainder else 0)

            if dim_questions:
                selected.extend(random.sample(
                    dim_questions,
                    min(num_to_select, len(dim_questions))
                ))

        # Shuffle to avoid dimension order bias
        random.shuffle(selected)
        return selected[:count]

    def load_from_file(self, filepath: str):
        """Load questions from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for q_data in data.get('questions', []):
            question = Question.from_dict(q_data)
            self.add_question(question)

    def save_to_file(self, filepath: str):
        """Save questions to JSON file"""
        data = {
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'questions': [q.to_dict() for q in self.questions]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_for_frontend(self,
                          language: str = 'en',
                          question_ids: Optional[List[int]] = None) -> List[Dict]:
        """
        Export questions in format suitable for frontend

        Args:
            language: Question language
            question_ids: Specific question IDs to export (None = all)

        Returns:
            List of question dictionaries
        """
        if question_ids:
            questions = [self.get_question(qid) for qid in question_ids]
            questions = [q for q in questions if q and q.language == language]
        else:
            questions = [q for q in self.questions if q.language == language and q.active]

        return [q.to_dict() for q in questions]


# Default question bank (can be loaded from database later)
def create_default_question_bank() -> QuestionBank:
    """Create default English question bank"""
    bank = QuestionBank()

    # Note: These are the original 30 questions from the HTML
    # In production, these should come from the database
    default_questions = [
        {
            'id': 1,
            'question': "About Social Interactions",
            'description': "When you're in a social setting, how do you typically interact with others? Do you tend to initiate conversations and be active in groups, or do you prefer deep conversations with fewer people?",
            'dimension': "I/E",
            'version': "1.0",
            'language': "en"
        },
        {
            'id': 2,
            'question': "About Problem Solving",
            'description': "When faced with a complex problem, how do you typically look for solutions? Do you rely more on known facts and experience, or do you prefer exploring new possibilities and innovative methods?",
            'dimension': "S/N",
            'version': "1.0",
            'language': "en"
        },
        {
            'id': 3,
            'question': "About Planning",
            'description': "How do you plan your daily life and work? Do you like making detailed plans in advance and following them, or do you prefer to stay flexible and adjust as you go?",
            'dimension': "J/P",
            'version': "1.0",
            'language': "en"
        },
        # Add remaining 27 questions here...
        # (Omitted for brevity - would include all 30 questions)
    ]

    for q_data in default_questions:
        question = Question.from_dict(q_data)
        bank.add_question(question)

    return bank


if __name__ == "__main__":
    # Example usage
    bank = create_default_question_bank()

    # Get 3 random balanced questions
    selected = bank.get_random_questions(count=3, balanced=True)
    print(f"Selected {len(selected)} questions:")
    for q in selected:
        print(f"  - {q.title} ({q.dimension})")

    # Export for frontend
    frontend_data = bank.export_for_frontend()
    print(f"\nTotal questions available: {len(frontend_data)}")
