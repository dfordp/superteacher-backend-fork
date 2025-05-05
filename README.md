# AI Grader Project

## Overview
The AI Grader Project is designed to automate the grading of answer scripts using artificial intelligence. It extracts questions from a PDF question paper, performs OCR on student answer scripts, cleans and aligns the answers, and grades them based on a predefined rubric.

## Project Structure
```
ai-grader-project
├── src
│   ├── ai_grader_v8.py        # Main logic for grading answers using AI
│   ├── controllers             # Contains routing logic or controller classes
│   ├── models                  # Contains data models or database interaction logic
│   ├── services                # Contains service classes that encapsulate business logic
│   └── utils                   # Contains utility functions or helper methods
├── tests                       # Contains test cases for the application
├── question                    # Sample question paper in PDF format
│   └── question.pdf
├── answer                      # Sample answer script in PDF format
│   └── sample.pdf
├── requirements.txt            # Lists dependencies required for the project
├── .env                        # Contains environment variables and configuration settings
├── .gitignore                  # Specifies files and directories to be ignored by Git
└── README.md                   # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai-grader-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in the `.env` file.

## Usage
To run the grading process, execute the following command:
```
python src/ai_grader_v8.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.