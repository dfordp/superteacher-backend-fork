import {
  ConversationStep,
  SessionData,
  GradingResult,
  UserIntent,
  SubjectArea,
  GradingApproach,
} from "../types";
import { sessionStore } from "../utils/sessionStore";
import { ocrService } from "./ocrService";
import { openaiService } from "./openaiService";
import { storageService } from "./storageService";

export class ChatService {
  /**
   * Get or initialize a session for a user ID
   */
  getOrCreateSession(userId: string): SessionData {
    const existingSession = sessionStore.getSession(userId);

    if (!existingSession) {
      // Create a new session if none exists
      return sessionStore.updateSession(userId, {
        question: null,
        studentAnswer: null,
        originalImage: null,
        subjectArea: null,
        contextualNotes: null,
        marks: null,
        step: ConversationStep.INITIAL,
        gradingApproach: null,
        lastInteraction: new Date(),
        previousGradingResults: [],
      });
    }

    // Update last interaction time
    return sessionStore.updateSession(userId, {
      lastInteraction: new Date(),
    });
  }

  /**
   * Process text messages from users with enhanced intent detection
   */
  async processTextMessage(userId: string, message: string): Promise<string> {
    console.log(
      `Processing message for user ${userId}: "${message.substring(0, 50)}${
        message.length > 50 ? "..." : ""
      }"`
    );

    // Get or initialize session
    const session = this.getOrCreateSession(userId);

    // Detect user intent
    const userIntent = this.detectUserIntent(message, session);
    console.log(`Detected user intent: ${userIntent}`);

    // Handle based on current session state and user intent
    switch (session.step) {
      case ConversationStep.INITIAL:
      case ConversationStep.COMPLETE:
      case ConversationStep.FOLLOW_UP:
        return this.handleInitialOrCompletedState(userId, message, userIntent);

      case ConversationStep.WAITING_FOR_QUESTION:
        return this.handleQuestionInput(userId, message, userIntent);

      case ConversationStep.WAITING_FOR_ANSWER:
        // Text messages when expecting an image upload
        return this.handleTextDuringAnswerWait(userId, message, userIntent);

      case ConversationStep.WAITING_FOR_INSTRUCTION:
        return this.handleInstructionInput(userId, message, userIntent);

      case ConversationStep.GRADING_IN_PROGRESS:
        return "I'm still analyzing the student's work. This will take just a moment...";

      default:
        // Recover from unexpected states
        console.error(`Unknown session state: ${session.step}`);
        sessionStore.updateSession(userId, {
          step: ConversationStep.INITIAL,
        });
        return "I'm ready to help you grade student work. What would you like to assess?";
    }
  }

  /**
   * Detect user intent from message and session context
   */
  protected detectUserIntent(
    message: string,
    session: SessionData
  ): UserIntent {
    const lowerMsg = message.toLowerCase().trim();

    // Check for greetings or help requests
    if (
      (lowerMsg.length < 20 &&
        /^(hi|hello|hey|greetings|howdy)/i.test(lowerMsg)) ||
      lowerMsg === "start" ||
      lowerMsg === "begin"
    ) {
      return UserIntent.GREETING;
    }

    // Check for help requests
    if (
      lowerMsg.includes("help") ||
      lowerMsg.includes("how do") ||
      lowerMsg.includes("what can you") ||
      lowerMsg.includes("instructions") ||
      (lowerMsg.includes("how") && lowerMsg.includes("work"))
    ) {
      return UserIntent.HELP_REQUEST;
    }

    // Check for new session requests
    if (
      lowerMsg.includes("start over") ||
      lowerMsg.includes("restart") ||
      lowerMsg.includes("reset") ||
      lowerMsg.includes("new session") ||
      lowerMsg.includes("different question") ||
      lowerMsg.includes("another student")
    ) {
      return UserIntent.NEW_SESSION;
    }

    // Check for grading instructions based on session state
    if (
      session.step === ConversationStep.WAITING_FOR_INSTRUCTION &&
      (lowerMsg.includes("grade") ||
        lowerMsg.includes("assess") ||
        lowerMsg.includes("evaluate") ||
        lowerMsg.includes("score") ||
        lowerMsg.includes("mark") ||
        lowerMsg.includes("check") ||
        lowerMsg.includes("review") ||
        lowerMsg.includes("how did") ||
        lowerMsg.includes("feedback"))
    ) {
      return UserIntent.GRADING_INSTRUCTION;
    }

    // Check for follow-up questions about grading
    if (
      session.step === ConversationStep.COMPLETE &&
      (lowerMsg.includes("why") ||
        lowerMsg.includes("how") ||
        lowerMsg.includes("explain") ||
        lowerMsg.includes("clarify") ||
        lowerMsg.includes("?"))
    ) {
      return UserIntent.FOLLOW_UP_QUESTION;
    }

    // Handle question provision based on context and session state
    if (
      (session.step === ConversationStep.INITIAL ||
        session.step === ConversationStep.WAITING_FOR_QUESTION ||
        session.step === ConversationStep.COMPLETE) &&
      message.length > 20 &&
      !lowerMsg.startsWith("grade") &&
      !lowerMsg.startsWith("assess")
    ) {
      return UserIntent.PROVIDE_QUESTION;
    }

    // Default logic based on session state
    switch (session.step) {
      case ConversationStep.WAITING_FOR_INSTRUCTION:
        return UserIntent.GRADING_INSTRUCTION;
      case ConversationStep.WAITING_FOR_QUESTION:
        return UserIntent.PROVIDE_QUESTION;
      case ConversationStep.COMPLETE:
        return UserIntent.FOLLOW_UP_QUESTION;
      default:
        return UserIntent.UNKNOWN;
    }
  }

  /**
   * Handle messages in initial or completed state
   */
  private handleInitialOrCompletedState(
    userId: string,
    message: string,
    intent: UserIntent
  ): string {
    switch (intent) {
      case UserIntent.GREETING:
        return this.provideWelcomeMessage();

      case UserIntent.HELP_REQUEST:
        return this.provideHelpMessage();

      case UserIntent.NEW_SESSION:
        // Reset the session for a fresh start
        sessionStore.resetSession(userId);
        return "I've started a new grading session. What would you like to assess?";

      case UserIntent.PROVIDE_QUESTION:
        // Process as a new question to assess
        return this.handleQuestionInput(userId, message, intent);

      case UserIntent.FOLLOW_UP_QUESTION:
        // Handle follow-up about previous grading
        return this.handleFollowUpQuestion(userId, message);

      default:
        // Default to treating as a question if in doubt
        sessionStore.updateSession(userId, {
          step: ConversationStep.WAITING_FOR_QUESTION,
        });
        return this.handleQuestionInput(
          userId,
          message,
          UserIntent.PROVIDE_QUESTION
        );
    }
  }

  /**
   * Provide a welcome message with system capabilities
   */
  private provideWelcomeMessage(): string {
    return `
Hello! I'm your AI teaching assistant, ready to help grade student work.

I can:
• Analyze uploaded images of student responses
• Assess answers across various subjects
• Provide detailed feedback and scoring
• Identify strengths and areas for improvement
• Suggest ways students can enhance their work

To get started, simply share what question or topic you'd like me to assess, then upload an image of the student's work.
`.trim();
  }

  /**
   * Provide help on using the system
   */
  private provideHelpMessage(): string {
    return `
Here's how to use this grading assistant:

1. First, tell me the question or topic you're assessing
2. Upload an image of the student's work/answer
3. Tell me how you'd like it graded (e.g., "Grade this out of 10" or "Provide detailed feedback")

You can also:
• Start a new grading session anytime by saying "start over"
• Ask follow-up questions about my assessment
• Request focus on specific aspects (e.g., "Focus on critical thinking")

The more context you provide about what you're looking for, the better I can tailor my assessment to your needs.
`.trim();
  }

  /**
   * Handle the question input from the teacher
   */
  private handleQuestionInput(
    userId: string,
    question: string,
    _intent: UserIntent
  ): string {
    console.log(`Processing question input for user ${userId}: "${question}"`);

    // Validate the question
    if (!question || question.trim().length < 5) {
      return "Could you provide more details about what you'd like me to assess? The more context you give, the better I can help grade the student's work.";
    }

    try {
      // Detect subject area and appropriate max marks
      const subjectArea = this.detectSubjectArea(question);
      const suggestedMarks = this.suggestDefaultMarks(question, subjectArea);

      // Update session with the question and detected metadata
      sessionStore.updateSession(userId, {
        question,
        subjectArea,
        step: ConversationStep.WAITING_FOR_ANSWER,
        // Only store suggested marks if they were found in the question
        contextualNotes: suggestedMarks
          ? `Suggested marks: ${suggestedMarks}`
          : "Marks not specified",
      });

      console.log(
        `Session updated with question and metadata. Subject: ${subjectArea}`
      );

      // Use a generic response instead of subject-specific responses
      let response = "I'll help you assess this work.";

      // Add marks information if available
      if (suggestedMarks) {
        response += ` I'll use ${suggestedMarks} marks as specified in your question.`;
      } else {
        response += ` When you provide grading instructions later, please specify how many marks to grade out of.`;
      }

      response += ` Please upload an image of the student's work so I can analyze their response.`;

      return response;
    } catch (error) {
      console.error(`Error processing question for user ${userId}:`, error);
      return "I encountered an issue processing your request. Could you try rephrasing the question?";
    }
  }

  /**
   * Detect the subject area from the question
   */
  private detectSubjectArea(question: string): SubjectArea {
    const lowerQuestion = question.toLowerCase();

    // Math detection
    if (
      /\b(math|algebra|geometry|calculus|equation|formula|solve|graph|function|polynomial|triangle|circle|angle|theorem|proof)\b/i.test(
        lowerQuestion
      ) ||
      /[+\-*/=^√∫∑π]/.test(question) ||
      /\d+\s*[+\-*/=]\s*\d+/.test(question)
    ) {
      return SubjectArea.MATH;
    }

    // Science detection
    if (
      /\b(science|biology|chemistry|physics|atom|molecule|cell|organism|experiment|lab|hypothesis|theory|element|compound|reaction|force|energy|ecosystem)\b/i.test(
        lowerQuestion
      )
    ) {
      return SubjectArea.SCIENCE;
    }

    // English/Language Arts detection
    if (
      /\b(english|essay|writing|paragraph|grammar|literature|novel|poem|poetry|author|character|theme|plot|narrative|syntax|metaphor|simile|analyze|text)\b/i.test(
        lowerQuestion
      )
    ) {
      return SubjectArea.ENGLISH;
    }

    // History/Social Studies detection
    if (
      /\b(history|social studies|civilization|government|war|revolution|president|country|nation|empire|colony|politics|society|culture|century|decade|era|period)\b/i.test(
        lowerQuestion
      )
    ) {
      return SubjectArea.HISTORY;
    }

    // Computer Science detection
    if (
      /\b(computer science|programming|code|algorithm|function|variable|class|object|data structure|loop|conditional|if statement|database|web|software|hardware|binary)\b/i.test(
        lowerQuestion
      )
    ) {
      return SubjectArea.COMPUTER_SCIENCE;
    }

    // Default to general if no specific subject is detected
    return SubjectArea.GENERAL;
  }

  /**
   * Suggest default marks based on question and subject
   */
  private suggestDefaultMarks(
    question: string,
    _subject: SubjectArea
  ): number | null {
    // Check if the question explicitly mentions marks
    const marksMatch = question.match(/\b(\d+)\s*(?:mark|point|score)s?\b/i);
    if (marksMatch) {
      return parseInt(marksMatch[1], 10);
    }

    // Return null to indicate that marks need to be specified by the user
    return null;
  }

  /**
   * Handle text messages during the waiting for answer stage
   */
  private handleTextDuringAnswerWait(
    userId: string,
    message: string,
    intent: UserIntent
  ): string {
    // Check for intent to reset/start over
    if (intent === UserIntent.NEW_SESSION) {
      sessionStore.resetSession(userId);
      return "I've reset the grading session. What would you like to assess now?";
    }

    // Check if this might be a new question
    if (message.length > 30 && !message.toLowerCase().includes("upload")) {
      return `I notice you've shared what looks like another question or context. Would you like to:
      
1. Start over with this new question (just say "start over")
2. Continue with the current question (please upload an image of the student's work)

Please let me know how you'd like to proceed.`;
    }

    // Default response - remind to upload an image
    return "I'm waiting for an image of the student's work. Please upload an image so I can analyze their response.";
  }

  /**
   * Process an uploaded image with OCR
   */
  async processImageUpload(userId: string, imagePath: string): Promise<string> {
    const session = this.getOrCreateSession(userId);
    console.log(
      `Processing image upload for user ${userId}, session state: ${session.step}`
    );

    // Handle image upload for various session states
    if (
      session.step === ConversationStep.COMPLETE ||
      session.step === ConversationStep.FOLLOW_UP
    ) {
      console.log(
        `Received image in ${session.step} state, transitioning to new session`
      );
      // Keep the previous grading result for reference
      const previousResults = session.previousGradingResults || [];
      if (session.question) {
        previousResults.push({
          score: session.marks || 0,
          outOf: session.marks || 10,
          percentage: session.marks ? (session.marks / 10) * 100 : 0,
          feedback: "Previous grading session",
          strengths: [],
          areas_for_improvement: [],
          suggested_points: [],
          correct_concepts: "",
          misconceptions: "",
          gradingApproach: session.gradingApproach || "balanced",
          timeGraded: new Date(),
          critical_thinking: 0,
          organization: 0,
          language_use: 0,
          concept_application: 0,
        });
      }

      // Reset but preserve history
      sessionStore.updateSession(userId, {
        question: "Assessment of student work",
        studentAnswer: null,
        originalImage: null,
        marks: null,
        step: ConversationStep.WAITING_FOR_ANSWER,
        previousGradingResults: previousResults,
      });
    }

    // If at initial state without a question, create a generic one
    if (session.step === ConversationStep.INITIAL) {
      sessionStore.updateSession(userId, {
        question: "Assessment of student work",
        step: ConversationStep.WAITING_FOR_ANSWER,
      });
    }

    try {
      // Upload the image to Firebase Storage
      console.log(`Uploading image to Firebase Storage: ${imagePath}`);
      const imageUrl = await storageService.uploadFile(imagePath);

      // Update with the image information (storing the URL instead of the path)
      sessionStore.updateSession(userId, {
        originalImage: imageUrl,
        step: ConversationStep.GRADING_IN_PROGRESS,
      });

      // Extract text from the image URL
      console.log(`Extracting text from image URL: ${imageUrl}`);
      const ocrText = await ocrService.extractTextFromImageUrl(imageUrl);
      console.log(`Successfully extracted OCR text (${ocrText.length} chars)`);

      // Update session with the extracted text
      sessionStore.updateSession(userId, {
        studentAnswer: ocrText,
        step: ConversationStep.WAITING_FOR_INSTRUCTION,
      });

      // Analyze the text content for better response
      const contentType = this.analyzeContentType(
        ocrText,
        session.subjectArea as SubjectArea | null
      );
      const textPreview = this.createTextPreview(ocrText);

      // Prepare response based on content
      let response = this.generateImageProcessedResponse(
        contentType,
        textPreview
      );

      return response;
    } catch (error) {
      console.error(`Error processing image:`, error);

      // Reset to waiting for answer state to allow retrying
      sessionStore.updateSession(userId, {
        step: ConversationStep.WAITING_FOR_ANSWER,
      });

      return "I encountered an issue processing that image. This could be due to image quality or format. Could you try uploading it again, perhaps with better lighting or clarity?";
    }
  }

  /**
   * Process an image from URL with OCR
   */
  async processImageFromUrl(userId: string, imageUrl: string): Promise<string> {
    const session = this.getOrCreateSession(userId);
    console.log(
      `Processing image from URL for user ${userId}, session state: ${session.step}`
    );

    // Handle image upload for various session states
    if (
      session.step === ConversationStep.COMPLETE ||
      session.step === ConversationStep.FOLLOW_UP
    ) {
      console.log(
        `Received image in ${session.step} state, transitioning to new session`
      );
      // Keep the previous grading result for reference
      const previousResults = session.previousGradingResults || [];
      if (session.question) {
        previousResults.push({
          score: session.marks || 0,
          outOf: session.marks || 10,
          percentage: session.marks ? (session.marks / 10) * 100 : 0,
          feedback: "Previous grading session",
          strengths: [],
          areas_for_improvement: [],
          suggested_points: [],
          correct_concepts: "",
          misconceptions: "",
          gradingApproach: session.gradingApproach || "balanced",
          timeGraded: new Date(),
          critical_thinking: 0,
          organization: 0,
          language_use: 0,
          concept_application: 0,
        });
      }

      // Reset but preserve history
      sessionStore.updateSession(userId, {
        question: "Assessment of student work",
        studentAnswer: null,
        originalImage: null,
        marks: null,
        step: ConversationStep.WAITING_FOR_ANSWER,
        previousGradingResults: previousResults,
      });
    }

    // If at initial state without a question, create a generic one
    if (!session.question) {
      console.log(
        `No question found for user ${userId}, setting default question`
      );
      sessionStore.updateSession(userId, {
        question: "Assessment of student work",
      });
    }

    try {
      // Store the image URL directly (no need to upload)
      sessionStore.updateSession(userId, {
        originalImage: imageUrl,
        step: ConversationStep.GRADING_IN_PROGRESS,
      });

      // Extract text from the image URL
      console.log(`Extracting text from image URL: ${imageUrl}`);
      const ocrText = await ocrService.extractTextFromImageUrl(imageUrl);
      console.log(`Successfully extracted OCR text (${ocrText.length} chars)`);

      // Update session with the extracted text
      const updatedSession = sessionStore.updateSession(userId, {
        studentAnswer: ocrText,
        step: ConversationStep.WAITING_FOR_INSTRUCTION,
      });

      // Double-check that question exists
      if (!updatedSession.question) {
        console.log(
          `Still no question after updating session, explicitly setting one`
        );
        sessionStore.updateSession(userId, {
          question: "Assessment of student work",
        });
      }

      // Log the current state for debugging
      const finalSession = sessionStore.getSession(userId);
      console.log(
        `Final session state - Question: ${!!finalSession?.question}, Answer: ${!!finalSession?.studentAnswer}`
      );

      // Analyze the text content for better response
      const contentType = this.analyzeContentType(
        ocrText,
        session.subjectArea as SubjectArea | null
      );
      const textPreview = this.createTextPreview(ocrText);

      // Prepare response based on content
      let response = this.generateImageProcessedResponse(
        contentType,
        textPreview
      );

      return response;
    } catch (error) {
      console.error(`Error processing image URL:`, error);

      // Reset to waiting for answer state to allow retrying
      sessionStore.updateSession(userId, {
        step: ConversationStep.WAITING_FOR_ANSWER,
      });

      return "I encountered an issue processing that image. This could be due to image quality or the URL format. Could you try uploading the image again?";
    }
  }

  /**
   * Create a preview of the extracted text
   */
  private createTextPreview(text: string): string {
    // If text is short, return it completely
    if (text.length <= 150) return text;

    // For longer text, create a meaningful preview
    const firstPart = text.substring(0, 100).trim();

    // Try to find sentence boundaries to create a cleaner preview
    const lastPeriodPos = firstPart.lastIndexOf(".");
    const lastQuestionPos = firstPart.lastIndexOf("?");
    const lastExclamationPos = firstPart.lastIndexOf("!");

    // Find the latest sentence boundary
    const endPos = Math.max(
      lastPeriodPos !== -1 ? lastPeriodPos + 1 : 0,
      lastQuestionPos !== -1 ? lastQuestionPos + 1 : 0,
      lastExclamationPos !== -1 ? lastExclamationPos + 1 : 0
    );

    // If we found a sentence boundary, use it; otherwise use the truncated preview
    const preview = endPos > 0 ? text.substring(0, endPos) : firstPart;

    return `${preview}... (${text.length} characters total)`;
  }

  /**
   * Analyze the content type of the extracted text
   */
  private analyzeContentType(
    text: string,
    subjectArea: SubjectArea | null
  ): string {
    if (!text || text.trim().length === 0) {
      return "empty";
    }

    // Check for math content
    if (
      /[+\-*/=^√∫∑π]{2,}|\d+\s*[+\-*/=]\s*\d+|equation|formula|=|solve|graph|calculate/i.test(
        text
      ) ||
      subjectArea === SubjectArea.MATH
    ) {
      return "math";
    }

    // Check for code
    if (
      /function|class|var |let |const |if\s*\(|for\s*\(|while\s*\(|import |public |private |def |print\(|#include/i.test(
        text
      ) ||
      subjectArea === SubjectArea.COMPUTER_SCIENCE
    ) {
      return "code";
    }

    // Determine if it's short answer or essay based on length
    if (text.length < 300) {
      return "short_answer";
    } else {
      return "essay";
    }
  }

  /**
   * Generate a response for processed image based on content type
   */
  private generateImageProcessedResponse(
    contentType: string,
    textPreview: string
  ): string {
    switch (contentType) {
      case "empty":
        return "I couldn't extract any text from this image. Is the text clear and legible? You might want to try uploading the image again or with better lighting.";

      case "math":
        return `I've analyzed the math work in the image. How would you like me to evaluate it? I can check the solution approach, identify errors, or provide a grade based on specific criteria.

Here's what I extracted:
"${textPreview}"`;

      case "code":
        return `I've processed the code in the image. How would you like me to assess it? I can evaluate correctness, efficiency, style, or provide an overall grade.

Here's what I extracted:
"${textPreview}"`;

      case "short_answer":
        return `I've extracted the student's response. How would you like me to evaluate it? I can check for accuracy, completeness, or provide a grade.

Here's what I extracted:
"${textPreview}"`;

      case "essay":
        return `I've processed the student's written response. How would you like me to evaluate it? I can assess organization, argumentation, evidence use, or provide an overall grade.

Here's a preview of what I extracted:
"${textPreview}"`;

      default:
        return `I've analyzed the student's work. How would you like me to grade or assess it?

Here's what I extracted:
"${textPreview}"`;
    }
  }

  /**
   * Handle the instruction input from the teacher
   */
  private async handleInstructionInput(
    userId: string,
    instruction: string,
    _intent: UserIntent
  ): Promise<string> {
    const session = sessionStore.getSession(userId);
    console.log(
      `Processing grading instruction for user ${userId}: "${instruction}"`
    );

    // Validate we have the necessary data
    if (!session.studentAnswer) {
      console.log(
        `Missing student answer for grading. Question: ${!!session.question}, Answer: ${!!session.studentAnswer}`
      );
      return "I'm missing the student's work. Could you please upload the student's work again?";
    }

    // If question is missing but we have the student answer, create a generic question
    if (!session.question && session.studentAnswer) {
      console.log(
        `Question missing but student answer available. Creating generic question for grading.`
      );
      sessionStore.updateSession(userId, {
        question: "Assessment of student work",
      });
      // Reload the session data
      const updatedSession = sessionStore.getSession(userId);
      if (updatedSession) {
        console.log(
          `Updated session with generic question: "${updatedSession.question}"`
        );
      }
    }

    try {
      // Detect grading approach and extract marks
      const gradingApproach = this.detectGradingApproach(instruction);
      const marks = this.extractMarksFromInstruction(instruction, session);
      console.log(
        `Detected grading approach: ${gradingApproach}, marks: ${marks}`
      );

      // If no marks were found or extracted, ask the user to specify
      if (marks === null) {
        sessionStore.updateSession(userId, {
          gradingApproach,
          step: ConversationStep.WAITING_FOR_INSTRUCTION,
        });
        return "Please specify how many marks this answer should be graded out of (e.g., 'Grade out of 15 marks').";
      }

      // Update session with grading parameters
      sessionStore.updateSession(userId, {
        marks,
        gradingApproach,
        step: ConversationStep.GRADING_IN_PROGRESS,
      });

      // Call AI service to grade the answer
      console.log(`Sending to AI for grading...`);
      const gradingResult = await openaiService.gradeAnswer(
        session.question || "Assessment of student work",
        session.studentAnswer,
        instruction,
        marks
      );

      console.log(
        `Received grading result: Score ${gradingResult.score}/${marks}`
      );

      // Calculate percentage
      gradingResult.outOf = marks;
      gradingResult.percentage = (gradingResult.score / marks) * 100;
      gradingResult.gradingApproach = gradingApproach;
      gradingResult.timeGraded = new Date();

      // Add letter grade if appropriate
      if (gradingApproach !== GradingApproach.QUICK) {
        gradingResult.letterGrade = this.calculateLetterGrade(
          gradingResult.percentage
        );
      }

      // Update session with completed status
      const previousResults = session.previousGradingResults || [];
      sessionStore.updateSession(userId, {
        step: ConversationStep.COMPLETE,
        previousGradingResults: [...previousResults, gradingResult],
      });

      // Format the response
      const response = this.formatGradingResponse(
        gradingResult,
        marks,
        gradingApproach
      );
      return response;
    } catch (error) {
      console.error("Error during grading:", error);

      // Move back to waiting for instruction to allow retrying
      sessionStore.updateSession(userId, {
        step: ConversationStep.WAITING_FOR_INSTRUCTION,
      });

      return `
I encountered an issue while grading this response. This could be due to:

1. The complexity of the student's answer
2. Technical difficulties with the grading process
3. Unusual formatting or content in the response

Could you try providing more specific grading instructions? For example, specify the number of marks (e.g., "Grade out of 10 points") or the aspects to focus on (e.g., "Evaluate their understanding of the water cycle").
      `.trim();
    }
  }

  /**
   * Handle follow-up questions about previous grading
   */
  private handleFollowUpQuestion(userId: string, question: string): string {
    const session = sessionStore.getSession(userId);

    // Move to follow-up state
    sessionStore.updateSession(userId, {
      step: ConversationStep.FOLLOW_UP,
    });

    // Get the most recent grading result if available
    const results = session.previousGradingResults;
    if (!results || results.length === 0) {
      return "I don't have any previous grading information to reference. Would you like to start a new assessment?";
    }

    const latestResult = results[results.length - 1];

    // Handle specific types of follow-up questions
    if (
      question.toLowerCase().includes("why") &&
      question.toLowerCase().includes("score")
    ) {
      return `
The score of ${latestResult.score}/${
        latestResult.outOf
      } was determined based on:

1. ${
        latestResult.strengths.length > 0
          ? latestResult.strengths[0]
          : "Understanding of key concepts"
      }
2. ${
        latestResult.areas_for_improvement.length > 0
          ? "Areas needing improvement: " +
            latestResult.areas_for_improvement[0]
          : "Areas that could be strengthened"
      }
3. ${
        latestResult.correct_concepts
          ? "Demonstrated knowledge: " +
            latestResult.correct_concepts.substring(0, 100)
          : "Overall approach to the question"
      }

The main factors affecting the score were the student's conceptual understanding and their ability to apply it to the specific problem.
      `.trim();
    }

    if (question.toLowerCase().includes("improve")) {
      return `
To improve their score, the student should focus on:

${latestResult.suggested_points.map((point) => `• ${point}`).join("\n")}

${
  latestResult.misconceptions
    ? `They should also address this misconception: ${latestResult.misconceptions}`
    : ""
}

Working on these areas would strengthen their understanding and likely result in a higher score on similar assessments.
      `.trim();
    }

    // Default follow-up response
    return `
Regarding the previous assessment (${latestResult.score}/${latestResult.outOf}):

The student demonstrated strengths in:
${latestResult.strengths.map((s) => `• ${s}`).join("\n")}

Areas that need improvement include:
${latestResult.areas_for_improvement.map((a) => `• ${a}`).join("\n")}

${latestResult.feedback}

Is there something specific about this assessment you'd like me to explain further?
    `.trim();
  }

  /**
   * Detect the grading approach from the instruction
   */
  private detectGradingApproach(instruction: string): GradingApproach {
    const lowerInstruction = instruction.toLowerCase();

    // Check for strict approach indicators
    if (
      lowerInstruction.includes("strict") ||
      lowerInstruction.includes("rigorous") ||
      lowerInstruction.includes("thorough") ||
      lowerInstruction.includes("exact") ||
      lowerInstruction.includes("precise")
    ) {
      return GradingApproach.STRICT;
    }

    // Check for lenient approach indicators
    if (
      lowerInstruction.includes("lenient") ||
      lowerInstruction.includes("generous") ||
      lowerInstruction.includes("forgiving") ||
      lowerInstruction.includes("effort") ||
      lowerInstruction.includes("attempt")
    ) {
      return GradingApproach.LENIENT;
    }

    // Check for detailed assessment requests
    if (
      lowerInstruction.includes("detailed") ||
      lowerInstruction.includes("in-depth") ||
      lowerInstruction.includes("comprehensive") ||
      lowerInstruction.includes("thorough") ||
      lowerInstruction.includes("elaborate")
    ) {
      return GradingApproach.DETAILED;
    }

    // Check for quick assessment requests
    if (
      lowerInstruction.includes("quick") ||
      lowerInstruction.includes("brief") ||
      lowerInstruction.includes("short") ||
      lowerInstruction.includes("just the score") ||
      lowerInstruction.includes("only the grade")
    ) {
      return GradingApproach.QUICK;
    }

    // Check for conceptual focus
    if (
      lowerInstruction.includes("concept") ||
      lowerInstruction.includes("understanding") ||
      lowerInstruction.includes("grasp") ||
      lowerInstruction.includes("comprehension")
    ) {
      return GradingApproach.CONCEPTUAL;
    }

    // Check for technical focus
    if (
      lowerInstruction.includes("technical") ||
      lowerInstruction.includes("accuracy") ||
      lowerInstruction.includes("precision") ||
      lowerInstruction.includes("correctness") ||
      lowerInstruction.includes("calculation")
    ) {
      return GradingApproach.TECHNICAL;
    }

    // Default to balanced approach
    return GradingApproach.BALANCED;
  }

  /**
   * Extract marks from instruction or use defaults
   */
  private extractMarksFromInstruction(
    instruction: string,
    session: SessionData
  ): number | null {
    // Try to extract marks from the instruction
    const marksMatch = instruction.match(
      /(\d+)\s*(?:mark|point|score|grade)s?/i
    );
    if (marksMatch) {
      return parseInt(marksMatch[1], 10);
    }

    // Check if we have suggested marks from earlier
    if (
      session.contextualNotes &&
      session.contextualNotes.includes("Suggested marks:")
    ) {
      const suggestedMatch = session.contextualNotes.match(
        /Suggested marks: (\d+)/
      );
      if (suggestedMatch) {
        return parseInt(suggestedMatch[1], 10);
      }
    }

    // Return null to indicate marks need to be specified
    return null;
  }

  /**
   * Calculate letter grade based on percentage
   */
  private calculateLetterGrade(percentage: number): string {
    if (percentage >= 97) return "A+";
    if (percentage >= 93) return "A";
    if (percentage >= 90) return "A-";
    if (percentage >= 87) return "B+";
    if (percentage >= 83) return "B";
    if (percentage >= 80) return "B-";
    if (percentage >= 77) return "C+";
    if (percentage >= 73) return "C";
    if (percentage >= 70) return "C-";
    if (percentage >= 67) return "D+";
    if (percentage >= 63) return "D";
    if (percentage >= 60) return "D-";
    return "F";
  }

  /**
   * Format the grading result into a user-friendly response
   */
  private formatGradingResponse(
    result: GradingResult,
    maxMarks: number,
    approach: GradingApproach
  ): string {
    // Quick format for quick assessment approach
    if (approach === GradingApproach.QUICK) {
      return `
## Quick Assessment

🏆 **Score: ${result.score}/${maxMarks}**)

${result.feedback}

*For more detailed feedback, just ask.*
    `.trim();
    }

    // Choose appropriate formatting based on approach
    let formattedResponse = "";

    // Header section
    formattedResponse += `## Student Assessment\n\n`;

    // Score section (all approaches include this)
    formattedResponse += `🏆 **Score: ${result.score}/${maxMarks}** `;
    if (result.letterGrade) {
      formattedResponse += `(${Math.round(result.percentage)}% - ${
        result.letterGrade
      })\n\n`;
    } else {
      formattedResponse += `(${Math.round(result.percentage)}%)\n\n`;
    }

    // Main feedback (all approaches include this)
    formattedResponse += `📝 **Feedback:**\n${result.feedback}\n\n`;

    // Strengths (detailed, balanced, conceptual, lenient approaches)
    if (
      approach === GradingApproach.DETAILED ||
      approach === GradingApproach.BALANCED ||
      approach === GradingApproach.CONCEPTUAL ||
      approach === GradingApproach.LENIENT
    ) {
      if (result.strengths && result.strengths.length > 0) {
        formattedResponse += `💪 **Strengths:**\n`;
        result.strengths.forEach((strength) => {
          formattedResponse += `- ${strength}\n`;
        });
        formattedResponse += `\n`;
      }
    }

    // Areas for improvement (most approaches except lenient)
    if (
      approach !== GradingApproach.LENIENT &&
      result.areas_for_improvement &&
      result.areas_for_improvement.length > 0
    ) {
      formattedResponse += `🔍 **Areas for Improvement:**\n`;
      result.areas_for_improvement.forEach((area) => {
        formattedResponse += `- ${area}\n`;
      });
      formattedResponse += `\n`;
    }

    // Suggestions (detailed, balanced approaches)
    if (
      (approach === GradingApproach.DETAILED ||
        approach === GradingApproach.BALANCED) &&
      result.suggested_points &&
      result.suggested_points.length > 0
    ) {
      formattedResponse += `💡 **Suggestions to Improve:**\n`;
      result.suggested_points.forEach((suggestion) => {
        formattedResponse += `- ${suggestion}\n`;
      });
      formattedResponse += `\n`;
    }

    // Conceptual understanding (conceptual, detailed approaches)
    if (
      (approach === GradingApproach.CONCEPTUAL ||
        approach === GradingApproach.DETAILED) &&
      result.correct_concepts
    ) {
      formattedResponse += `✅ **Correct Concepts:**\n${result.correct_concepts}\n\n`;
    }

    // Misconceptions (technical, strict, detailed approaches)
    if (
      (approach === GradingApproach.TECHNICAL ||
        approach === GradingApproach.STRICT ||
        approach === GradingApproach.DETAILED) &&
      result.misconceptions
    ) {
      formattedResponse += `⚠️ **Misconceptions:**\n${result.misconceptions}\n\n`;
    }

    // Final prompt for next steps
    formattedResponse += `Would you like to grade another answer or have any questions about this assessment?`;

    return formattedResponse;
  }
}

export const chatService = new ChatService(); // lowercase object, uppercase class
