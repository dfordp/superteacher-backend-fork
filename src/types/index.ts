// src/types/index.ts - Complete type definitions with CBSE extensions

// Define the conversation steps with more detailed states including CBSE flow
export enum ConversationStep {
  INITIAL = "initial", // Very first interaction
  WAITING_FOR_CLASS = "waiting_for_class", // CBSE: Waiting for class/grade selection
  WAITING_FOR_SUBJECT = "waiting_for_subject", // CBSE: Waiting for subject selection
  WAITING_FOR_QUESTION_PAPER = "waiting_for_question_paper", // CBSE: Waiting for question paper upload
  PROCESSING_QUESTION_PAPER = "processing_question_paper", // CBSE: Processing question paper
  EXTRACTING_QUESTION_MARKS = "extracting_question_marks", // CBSE: Extracting marks from question paper
  WAITING_FOR_MARKS_CONFIRMATION = "waiting_for_marks_confirmation", // CBSE: Waiting for marks confirmation
  WAITING_FOR_MARKS_UPDATE = "waiting_for_marks_update", // CBSE: Waiting for marks update
  WAITING_FOR_STUDENT_ANSWER = "waiting_for_student_answer", // CBSE: Waiting for student answer upload
  WAITING_FOR_QUESTION = "waiting_for_question", // Original: Waiting for question/topic to assess
  WAITING_FOR_ANSWER = "waiting_for_answer", // Original: Waiting for student's answer (image upload)
  WAITING_FOR_INSTRUCTION = "waiting_for_instruction", // Original: Waiting for grading instructions
  GRADING_IN_PROGRESS = "grading_in_progress", // Processing the grading request
  COMPLETE = "complete", // Grading is complete
  FOLLOW_UP = "follow_up", // Handling follow-up questions about the grading
}

// CBSE class/grade levels
export enum ClassLevel {
  CLASS_6 = "class_6",
  CLASS_7 = "class_7",
  CLASS_8 = "class_8",
  CLASS_9 = "class_9",
  CLASS_10 = "class_10",
  CLASS_11 = "class_11",
  CLASS_12 = "class_12",
}

// Enhanced Subject area detection with CBSE subjects
export enum SubjectArea {
  MATH = "mathematics",
  SCIENCE = "science",
  ENGLISH = "english_language_arts",
  HISTORY = "history",
  SOCIAL_STUDIES = "social_studies",
  ECONOMICS = "economics", // CBSE: Economics
  BUSINESS_STUDIES = "business_studies", // CBSE: Business Studies
  ACCOUNTANCY = "accountancy", // CBSE: Accountancy
  POLITICAL_SCIENCE = "political_science", // CBSE: Political Science
  GEOGRAPHY = "geography", // CBSE: Geography
  PHYSICS = "physics", // CBSE: Physics
  CHEMISTRY = "chemistry", // CBSE: Chemistry
  BIOLOGY = "biology", // CBSE: Biology
  COMPUTER_SCIENCE = "computer_science",
  FOREIGN_LANGUAGE = "foreign_language",
  ARTS = "arts",
  PHYSICAL_EDUCATION = "physical_education",
  GENERAL = "general",
}

// Enhanced user intent including CBSE-specific intents
export enum UserIntent {
  GREETING = "greeting", // General hello/hi
  HELP_REQUEST = "help_request", // Asking for help/guidance
  PROVIDE_QUESTION = "provide_question", // Providing a question to assess
  UPLOAD_ANSWER = "upload_answer", // Uploading student's answer
  GRADING_INSTRUCTION = "grading_instruction", // Giving grading instructions
  CLARIFICATION_REQUEST = "clarification_request", // Asking for clarification
  FEEDBACK_ON_FEEDBACK = "feedback_on_feedback", // Commenting on the grading
  NEW_SESSION = "new_session", // Starting a new session
  FOLLOW_UP_QUESTION = "follow_up_question", // Follow-up about the grading
  SET_CLASS = "set_class", // CBSE: Setting the class/grade level
  SET_SUBJECT = "set_subject", // CBSE: Setting the subject
  CONFIRM_MARKS = "confirm_marks", // CBSE: Confirming marks distribution
  UPDATE_MARKS = "update_marks", // CBSE: Updating marks for questions
  UNKNOWN = "unknown", // Cannot determine intent
}

// Grading approach detection
export enum GradingApproach {
  STRICT = "strict", // Emphasis on accuracy/correctness
  BALANCED = "balanced", // Balance between correctness and effort
  LENIENT = "lenient", // Emphasis on effort/improvement
  DETAILED = "detailed", // Highly detailed assessment
  QUICK = "quick", // Brief assessment
  CONCEPTUAL = "conceptual", // Focus on conceptual understanding
  TECHNICAL = "technical", // Focus on technical accuracy
  RUBRIC_BASED = "rubric_based", // Based on a specific rubric
  CBSE_STANDARD = "cbse_standard", // CBSE: Following CBSE marking scheme
}

// Enhanced session data with more contextual information
export interface SessionData {
  userId?: string; // Unique identifier for the user
  question: string | null; // The question/topic being assessed
  studentAnswer: string | null; // The extracted text from the student's answer
  originalImage: string | null; // Path or reference to the uploaded image
  subjectArea: SubjectArea | null; // Detected subject (math, science, english, etc.)
  contextualNotes: string | null; // Additional context provided by the teacher
  marks: number | null; // Maximum marks for this assessment
  step: ConversationStep; // Current state in the conversation
  gradingApproach: string | null; // Detected approach (strict, lenient, etc.)
  lastInteraction: Date | null; // Timestamp of last interaction
  previousGradingResults: GradingResult[] | null; // History of previous gradings
}

// CBSE-specific session data extends the base SessionData
export interface CbseSessionData extends SessionData {
  classLevel: ClassLevel | null;
  questionPaper: string | null;
  questionMarks: Map<number, number> | null;
  isMarkingConfirmed: boolean;
  conversationHistory?: { role: string; content: string }[]; // Add this line
}

// Enhanced chat request with more metadata
export interface ChatRequest {
  message: string; // The user's message
  userId: string; // User identifier
  timestamp: Date; // When the request was made
  imageAttached: boolean; // Whether an image is attached
  intentDetected?: string; // Optional detected intent of message
}

// Enhanced grading result with detailed feedback categories
export interface GradingResult {
  score: number; // Numerical score
  outOf: number; // Maximum possible score
  percentage: number; // Score as percentage
  letterGrade?: string; // Optional letter grade (A, B, C, etc.)

  // Overall assessment
  feedback: string; // Main feedback summary

  // Specific areas of assessment
  strengths: string[]; // What the student did well
  areas_for_improvement: string[]; // What needs improvement
  suggested_points: string[]; // Specific improvement suggestions
  correct_concepts: string; // Concepts correctly understood
  misconceptions: string; // Evident misconceptions

  // Additional assessment components
  critical_thinking: number | null; // Score for critical thinking (optional)
  organization: number | null; // Score for organization (optional)
  language_use: number | null; // Score for language/communication (optional)
  concept_application: number | null; // Score for applying concepts (optional)

  // Legacy support
  mistakes?: string[]; // Kept for backward compatibility

  // Metadata
  gradingApproach: string; // How the grading was approached
  subjectSpecificNotes?: string; // Notes specific to the subject area
  timeGraded: Date; // When the grading was completed

  // CBSE-specific fields
  cbseClass?: string; // CBSE: Class level
  cbseSubject?: string; // CBSE: Subject
  conceptsScore?: number; // CBSE: Score for understanding of concepts
  diagramScore?: number; // CBSE: Score for diagrams
  applicationScore?: number; // CBSE: Score for application of theories
  terminologyScore?: number; // CBSE: Score for use of terminology
  is_relevant?: boolean; // CBSE: Whether the answer is relevant to the question
}

// Enhanced OpenAI/LLM response structure
export interface LLMResponse {
  choices: {
    message: {
      content: string;
      role: string;
    };
    finish_reason: string;
  }[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  model?: string;
}
