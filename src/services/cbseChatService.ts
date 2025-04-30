// src/services/cbseChatService.ts
import {
  ConversationStep,
  GradingResult,
  SubjectArea,
  ClassLevel,
  CbseSessionData,
  GradingApproach,
} from "../types";
import { sessionStore } from "../utils/sessionStore";
import { openaiService } from "./openaiService";
import { storageService } from "./storageService";

/**
 * CBSE-specific ChatService with natural conversational flow
 * that guides the user through the CBSE grading workflow.
 */
export class CbseChatService {
  /**
   * Get or initialize a session for a user ID with CBSE-specific data
   */
  getOrCreateSession(userId: string): CbseSessionData {
    const existingSession = sessionStore.getSession(userId) as CbseSessionData;

    // For new sessions, add CBSE-specific fields and conversation history
    if (!existingSession.classLevel) {
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
        classLevel: null,
        questionPaper: null,
        questionMarks: null,
        isMarkingConfirmed: false,
        conversationHistory: [], // Initialize conversation history
      }) as CbseSessionData;
    }

    // Update last interaction time
    return sessionStore.updateSession(userId, {
      lastInteraction: new Date(),
    }) as CbseSessionData;
  }

  /**
   * Process text messages with conversational CBSE-specific flow
   */
  async processTextMessage(userId: string, message: string): Promise<string> {
    console.log(
      `Processing message for user ${userId}: "${message.substring(0, 50)}${
        message.length > 50 ? "..." : ""
      }"`
    );

    // Get or initialize session
    const session = this.getOrCreateSession(userId);
    console.log(`Current session state: ${session.step}`);

    // Store the original message for context
    const userMessage = message;

    // Process intents and update session state
    await this.processMessageIntent(userId, message, session);

    // Generate a conversational response
    return this.generateConversationalResponse(userId, userMessage);
  }

  /**
   * Process message intent and update session state
   */
  private async processMessageIntent(
    userId: string,
    message: string,
    session: CbseSessionData
  ): Promise<void> {
    // Extract class level if present in message
    const classLevel = this.extractClassLevel(message);
    if (
      classLevel &&
      (session.step === ConversationStep.INITIAL ||
        session.step === ConversationStep.WAITING_FOR_CLASS)
    ) {
      console.log(`Extracted class level: ${classLevel}`);
      sessionStore.updateSession(userId, {
        classLevel,
        step: ConversationStep.WAITING_FOR_SUBJECT,
      });
    }

    // Extract subject if present in message
    const subjectArea = this.extractSubjectArea(message);
    if (
      subjectArea &&
      (session.step === ConversationStep.WAITING_FOR_SUBJECT ||
        (session.step === ConversationStep.INITIAL && classLevel))
    ) {
      console.log(`Extracted subject: ${subjectArea}`);
      sessionStore.updateSession(userId, {
        subjectArea,
        step: ConversationStep.WAITING_FOR_QUESTION_PAPER,
      });
    }

    // Handle marks confirmation
    const lowerMsg = message.toLowerCase().trim();
    if (session.step === ConversationStep.WAITING_FOR_MARKS_CONFIRMATION) {
      if (
        lowerMsg.includes("yes") ||
        lowerMsg.includes("correct") ||
        lowerMsg.includes("right") ||
        lowerMsg.includes("good")
      ) {
        console.log("Marks confirmed");
        sessionStore.updateSession(userId, {
          isMarkingConfirmed: true,
          step: ConversationStep.WAITING_FOR_STUDENT_ANSWER,
        });
      } else if (
        lowerMsg.includes("no") ||
        lowerMsg.includes("incorrect") ||
        lowerMsg.includes("wrong") ||
        lowerMsg.includes("not right")
      ) {
        console.log("Marks correction needed");
        sessionStore.updateSession(userId, {
          step: ConversationStep.WAITING_FOR_MARKS_UPDATE,
        });
      }
    }

    // Handle marks update requests
    if (session.step === ConversationStep.WAITING_FOR_MARKS_UPDATE) {
      const updatePattern =
        /(?:question|q)\s*\.?\s*(\d+)(?:[a-z])?\s*(?:should|to|is|has|have|update|change|set)\s*(?:be|have|to)?\s*(\d+)\s*(?:mark|marks)?/i;
      const match = message.match(updatePattern);

      if (match && session.questionMarks) {
        const questionNumber = parseInt(match[1], 10);
        const newMarks = parseInt(match[2], 10);

        console.log(
          `Updating marks for question ${questionNumber} to ${newMarks}`
        );

        // Update the question marks
        session.questionMarks.set(questionNumber, newMarks);

        // Move back to confirmation state
        sessionStore.updateSession(userId, {
          questionMarks: session.questionMarks,
          step: ConversationStep.WAITING_FOR_MARKS_CONFIRMATION,
        });
      }
    }

    // Handle general intents - greeting, help, new session
    if (lowerMsg.match(/^(hi|hello|hey|greetings|start|begin)\b/i)) {
      console.log("Detected greeting intent");
      sessionStore.updateSession(userId, {
        step: ConversationStep.WAITING_FOR_CLASS,
      });
    } else if (
      lowerMsg.includes("start over") ||
      lowerMsg.includes("restart") ||
      lowerMsg.includes("reset") ||
      lowerMsg.includes("new session") ||
      lowerMsg.includes("another paper")
    ) {
      console.log("Starting new session");
      sessionStore.resetSession(userId);
      sessionStore.updateSession(userId, {
        step: ConversationStep.WAITING_FOR_CLASS,
        conversationHistory: session.conversationHistory || [],
      });
    }

    // Always record that we've updated the session after intent processing
    console.log(
      `Session updated, current state: ${
        (sessionStore.getSession(userId) as CbseSessionData).step
      }`
    );
  }

  /**
   * Extract class level from message using a more flexible approach
   */
  private extractClassLevel(message: string): ClassLevel | null {
    const lowerMsg = message.toLowerCase().trim();

    // Check for class level mentions using regex
    const classMatch = lowerMsg.match(/\b(?:class|grade)?\s*(\d+)(?:\s|$|\b)/i);
    if (classMatch) {
      const classNum = parseInt(classMatch[1], 10);

      switch (classNum) {
        case 6:
          return ClassLevel.CLASS_6;
        case 7:
          return ClassLevel.CLASS_7;
        case 8:
          return ClassLevel.CLASS_8;
        case 9:
          return ClassLevel.CLASS_9;
        case 10:
          return ClassLevel.CLASS_10;
        case 11:
          return ClassLevel.CLASS_11;
        case 12:
          return ClassLevel.CLASS_12;
      }
    }

    return null;
  }

  /**
   * Extract subject area from message with improved pattern matching
   */
  private extractSubjectArea(message: string): SubjectArea | null {
    const lowerMsg = message.toLowerCase().trim();

    if (lowerMsg.includes("math") || lowerMsg.includes("mathematics")) {
      return SubjectArea.MATH;
    } else if (lowerMsg.includes("econ")) {
      return SubjectArea.ECONOMICS;
    } else if (lowerMsg.includes("science") && !lowerMsg.includes("social")) {
      return SubjectArea.SCIENCE;
    } else if (lowerMsg.includes("english")) {
      return SubjectArea.ENGLISH;
    } else if (lowerMsg.includes("history")) {
      return SubjectArea.HISTORY;
    } else if (
      lowerMsg.includes("social") ||
      lowerMsg.includes("social studies")
    ) {
      return SubjectArea.SOCIAL_STUDIES;
    } else if (lowerMsg.includes("business")) {
      return SubjectArea.BUSINESS_STUDIES;
    } else if (lowerMsg.includes("account")) {
      return SubjectArea.ACCOUNTANCY;
    } else if (
      lowerMsg.includes("politi") ||
      lowerMsg.includes("political science")
    ) {
      return SubjectArea.POLITICAL_SCIENCE;
    } else if (lowerMsg.includes("geo")) {
      return SubjectArea.GEOGRAPHY;
    } else if (lowerMsg.includes("physics")) {
      return SubjectArea.PHYSICS;
    } else if (lowerMsg.includes("chem")) {
      return SubjectArea.CHEMISTRY;
    } else if (lowerMsg.includes("bio")) {
      return SubjectArea.BIOLOGY;
    } else if (lowerMsg.includes("computer")) {
      return SubjectArea.COMPUTER_SCIENCE;
    }

    return null;
  }

  /**
   * Generate a conversational response using OpenAI
   */
  private async generateConversationalResponse(
    userId: string,
    userMessage: string,
    additionalContext: string = ""
  ): Promise<string> {
    const session = this.getOrCreateSession(userId);

    // Build conversation history
    const history = session.conversationHistory || [];

    // Build a detailed system prompt for the LLM
    const systemPrompt = `
You are SuperTeacher AI, a CBSE grading assistant that helps teachers grade student papers.
You have a warm, conversational, helpful personality like ChatGPT while still maintaining expertise in CBSE grading standards.

CURRENT GRADING SESSION STATE:
- Current workflow step: ${session.step}
- Class level: ${session.classLevel || "Not yet specified"}
- Subject: ${session.subjectArea || "Not yet specified"}
- Question paper uploaded: ${session.questionPaper ? "Yes" : "No"}
- Marks confirmed: ${session.isMarkingConfirmed ? "Yes" : "No"}
- Student answer uploaded: ${session.studentAnswer ? "Yes" : "No"}

${additionalContext}

The user's message is: "${userMessage}"

CONVERSATION GUIDELINES:
1. Maintain a warm, natural, conversational tone - don't sound robotic or like you're following a script
2. Respond to the specific content of the user's message first before guiding them
3. Use conversational phrases and informal language where appropriate
4. Detect and acknowledge multiple intents in a single message (e.g., "I'm grading Class 12 Economics")
5. Be flexible - if they provide information out of the expected sequence, acknowledge it and use it

WORKFLOW GUIDANCE (maintain subtly in your response):
- If class level is not set: Guide them to specify which class they're grading (e.g., Class 10, Class 12)
- If class is set but subject isn't: Ask which subject they're grading
- If class and subject are set but no question paper: Ask them to upload the question paper
- If question paper is uploaded but marks aren't confirmed: Ask if extracted marks are correct
- If marks are confirmed but no student answer: Ask them to upload the student's answer
- If they ask off-topic questions: Answer helpfully, then gently guide back to the workflow

For Economics subject specifically, show your expertise in CBSE Economics assessment standards.

Your response should feel like a natural conversation with a knowledgeable colleague, not a rigid system following steps.
`;

    try {
      // Call OpenAI for a conversational response
      let response = await openaiService.generateConversation(
        systemPrompt,
        userMessage,
        history
      );

      // Add special context for showing marks if needed
      if (
        session.step === ConversationStep.WAITING_FOR_MARKS_CONFIRMATION &&
        session.questionMarks &&
        session.questionMarks.size > 0
      ) {
        // Format marks for display
        const formattedMarks = Array.from(session.questionMarks.entries())
          .sort((a, b) => a[0] - b[0])
          .map(
            ([qNum, marks]) =>
              `Question ${qNum}: ${marks} mark${marks !== 1 ? "s" : ""}`
          )
          .join("\n");

        // Insert marks information if not already present
        if (!response.includes("Question 1:") && !response.includes("marks")) {
          response = response.replace(
            /(\. |,|\?) (Are|Do|Could|Would|Is|Can|Have|Will)/,
            `$1 Here are the questions and their marks I've extracted:\n\n${formattedMarks}\n\n$2`
          );
        }
      }

      // Update conversation history
      const updatedHistory = [
        ...history,
        { role: "user", content: userMessage },
        { role: "assistant", content: response },
      ];

      // Keep history at a reasonable size
      const trimmedHistory = updatedHistory.slice(-15);

      sessionStore.updateSession(userId, {
        conversationHistory: trimmedHistory,
      });

      return response;
    } catch (error) {
      console.error("Error generating conversational response:", error);

      // Fallback if OpenAI fails
      return this.getFallbackResponse(session);
    }
  }

  /**
   * Provide a fallback response if conversational generation fails
   */
  private getFallbackResponse(session: CbseSessionData): string {
    switch (session.step) {
      case ConversationStep.INITIAL:
      case ConversationStep.WAITING_FOR_CLASS:
        return "Welcome to the CBSE Grading Assistant! I'm here to help you grade papers. Could you tell me which class you're grading for?";

      case ConversationStep.WAITING_FOR_SUBJECT:
        return `Great! You're grading for ${
          session.classLevel?.replace("_", " ").toUpperCase() || "a class"
        }. What subject are you grading?`;

      case ConversationStep.WAITING_FOR_QUESTION_PAPER:
        return `Perfect! I'll help you grade ${
          session.subjectArea?.replace("_", " ") || "your subject"
        } for ${
          session.classLevel?.replace("_", " ").toUpperCase() || "your class"
        }. Could you upload the question paper so I can analyze the questions and marks?`;

      case ConversationStep.WAITING_FOR_MARKS_CONFIRMATION:
        return "I've analyzed the question paper and extracted the marks. Do these look correct to you?";

      case ConversationStep.WAITING_FOR_MARKS_UPDATE:
        return "Which question needs marks correction? You can tell me something like 'Question 3 should be 5 marks'.";

      case ConversationStep.WAITING_FOR_STUDENT_ANSWER:
        return "Great! Now please upload the student's answer paper so I can grade it according to CBSE standards.";

      case ConversationStep.GRADING_IN_PROGRESS:
        return "I'm analyzing the student's answer and preparing a detailed assessment based on CBSE guidelines.";

      case ConversationStep.COMPLETE:
        return "I've completed the assessment. Would you like to grade another paper or do you have questions about this grading?";

      default:
        return "I'm here to help with CBSE grading. What would you like to do next?";
    }
  }

  /**
   * Process a question paper URL using GPT-4 Vision
   */
  async processQuestionPaperUrl(
    userId: string,
    imageUrl: string
  ): Promise<string> {
    // Update session state
    sessionStore.updateSession(userId, {
      originalImage: imageUrl,
      step: ConversationStep.PROCESSING_QUESTION_PAPER,
    });

    // Use GPT-4 Vision to extract and analyze the question paper
    const analysisPrompt = `
  You are analyzing a CBSE question paper.
  1. Extract all text exactly as it appears.
  2. Identify all questions and their marks.
  3. Pay special attention to the question numbers and marks allocated to each question.
  4. Look for any text marked "MM" or "Maximum Marks" to find the total marks.
  `;

    try {
      // Extract text and analyze it in one step
      const ocrText = await openaiService.analyzeImage(
        imageUrl,
        analysisPrompt
      );
      console.log(
        `Extracted and analyzed question paper text: ${ocrText.length} chars`
      );

      // Update session with the text
      sessionStore.updateSession(userId, {
        questionPaper: ocrText,
        step: ConversationStep.EXTRACTING_QUESTION_MARKS,
      });

      // Now use GPT to structure the extracted data
      const structurePrompt = `
Given this extracted text from a CBSE question paper:

${ocrText}

Extract and structure the following information in JSON format:
1. totalMarks: The total marks for the paper
2. questions: An array of objects, each with:
   - number: The question number
   - text: The full text of the question
   - marks: The marks allocated to this question

Return ONLY valid JSON without explanation.
`;

      const structuredData = await openaiService.generateText(structurePrompt);
      let cleanedData = structuredData;
      if (cleanedData.includes("```json")) {
        cleanedData = cleanedData.replace(/```json\s*|\s*```/g, "");
      }
      try {
        // Parse the structured data
        const parsedData = JSON.parse(cleanedData);

        // Convert to question marks map
        const questionMarks = new Map<number, number>();
        const questionText = new Map<number, string>();

        parsedData.questions.forEach(
          (q: { number: number; text: string; marks: number }) => {
            questionMarks.set(q.number, q.marks);
            questionText.set(q.number, q.text);
          }
        );

        // Update session with extracted marks
        sessionStore.updateSession(userId, {
          questionMarks,
          step: ConversationStep.WAITING_FOR_MARKS_CONFIRMATION,
        });

        // Add additional context with the extracted questions for the response
        const additionalContext = `
        EXTRACTED QUESTIONS AND MARKS:
        ${parsedData.questions
          .map(
            (q: { number: number; text: string; marks: number }) =>
              `Question ${q.number}: ${q.text.substring(0, 100)}${
                q.text.length > 100 ? "..." : ""
              }\nMarks: ${q.marks}`
          )
          .join("\n\n")}

      Guide the user to confirm if these extracted marks are correct. If they say yes, update isMarkingConfirmed to true and move to waiting for student answer. If they say no, help them correct the marks.
      `;

        // Generate a conversational response
        return await this.generateConversationalResponse(
          userId,
          "I've uploaded the question paper",
          additionalContext
        );
      } catch (error) {
        console.error("Error parsing structured data:", error);

        // Fallback to simpler extraction
        const extractionResult = this.extractQuestionMarks(ocrText);

        sessionStore.updateSession(userId, {
          questionMarks: extractionResult.marks,
          step: ConversationStep.WAITING_FOR_MARKS_CONFIRMATION,
        });

        const additionalContext = `
I had some trouble structuring the question paper data, but I've extracted the following marks:
${Array.from(extractionResult.marks.entries())
  .sort((a, b) => a[0] - b[0])
  .map(
    ([qNum, marks]) =>
      `Question ${qNum}: ${marks} mark${marks !== 1 ? "s" : ""}`
  )
  .join("\n")}

Ask the user if these marks look correct.
`;

        return await this.generateConversationalResponse(
          userId,
          "I've uploaded the question paper",
          additionalContext
        );
      }
    } catch (error) {
      console.error("Error using GPT Vision:", error);

      // Fall back to more generic handling
      return this.generateConversationalResponse(
        userId,
        "I've uploaded the question paper",
        "I had trouble processing the question paper. Could you please try uploading a clearer image?"
      );
    }
  }

  /**
   * Process student answer from URL using GPT-4 Vision
   */
  async processStudentAnswerUrl(
    userId: string,
    imageUrl: string
  ): Promise<string> {
    const session = this.getOrCreateSession(userId);
    console.log(`Processing student answer from URL for user ${userId}`);

    // Ensure we have question paper and marks first
    if (
      !session.questionPaper ||
      !session.questionMarks ||
      !session.isMarkingConfirmed
    ) {
      return await this.generateConversationalResponse(
        userId,
        "I've uploaded a student answer",
        "The user has uploaded a student answer, but we don't have the question paper or confirmed marks yet. Gently remind them that we need to complete those steps first."
      );
    }

    // Update session state
    sessionStore.updateSession(userId, {
      originalImage: imageUrl,
      step: ConversationStep.GRADING_IN_PROGRESS,
    });

    try {
      // Extract text from the student answer using GPT-4 Vision
      const analysisPrompt = `
You are analyzing a student's answer to a CBSE ${session.subjectArea} exam.
Extract all text from this student's answer exactly as it appears, preserving formatting as much as possible.
Pay special attention to:
1. All written text in the image
2. Any diagrams or figures (describe them briefly where they appear)
3. Mathematical formulas or equations (if present)
4. Numbered points or sections in the answer
`;

      const studentAnswer = await openaiService.analyzeImage(
        imageUrl,
        analysisPrompt
      );
      console.log(
        `Successfully extracted student answer text (${studentAnswer.length} chars)`
      );

      // Update session with the extracted text
      sessionStore.updateSession(userId, {
        studentAnswer,
        step: ConversationStep.GRADING_IN_PROGRESS,
      });

      // Calculate total marks
      let totalMarks = 0;
      for (const marks of session.questionMarks.values()) {
        totalMarks += marks;
      }

      // Create subject-specific grading instructions
      let subjectSpecificInstructions = "";
      const subjectString = session.subjectArea?.replace("_", " ") || "general";
      const classString =
        session.classLevel?.replace("_", " ").toUpperCase() || "";

      if (session.subjectArea === SubjectArea.ECONOMICS) {
        subjectSpecificInstructions = `
        For Economics, follow these CBSE guidelines:
        - Award full marks for complete explanations of economic concepts
        - Award partial marks for partial understanding
        - Check for correct diagrams when required
        - Look for proper economic terminology usage
        - Evaluate application of economic theories to real-world scenarios
        - Consider logical structure and flow of the answer
        - Reward proper use of economic data and statistical information
        - Assess ability to analyze economic policies and their impacts
        - Consider understanding of both microeconomic and macroeconomic principles
        - Evaluate comprehension of Indian economic development and challenges
        `;
      } else if (session.subjectArea === SubjectArea.ACCOUNTANCY) {
        subjectSpecificInstructions = `
        For Accountancy, follow these CBSE guidelines:
        - Award full marks for correct numerical calculations and procedures
        - Award marks for proper accounting formats and presentations
        - Check for accurate application of accounting principles
        - Look for proper use of accounting terminology
        - Evaluate understanding of double-entry bookkeeping system
        - Consider precision in financial statement preparation
        - Assess understanding of accounting standards and conventions
        - Check for proper journal entries, ledger posting, and trial balance
        - Evaluate ability to analyze financial statements
        - Consider comprehension of partnership accounts, company accounts, and not-for-profit organization accounts
        `;
      } else if (session.subjectArea === SubjectArea.BUSINESS_STUDIES) {
        subjectSpecificInstructions = `
        For Business Studies, follow these CBSE guidelines:
        - Award full marks for thorough explanation of business concepts
        - Award marks for relevant examples from business world
        - Check for understanding of management principles and functions
        - Look for proper business terminology usage
        - Evaluate application of theoretical frameworks to business cases
        - Consider organization and structure of the answer
        - Assess knowledge of business environment and its components
        - Check understanding of marketing, finance, and human resource concepts
        - Evaluate comprehension of entrepreneurship development
        - Consider understanding of consumer protection and business ethics
        `;
      }

      // Grade the student's answer
      const gradingPrompt = `
You are a CBSE examiner grading a ${classString} ${subjectString} exam.

QUESTION PAPER:
${session.questionPaper}

STUDENT'S ANSWER:
${studentAnswer}

GRADING INSTRUCTIONS:
${subjectSpecificInstructions}
- VERY IMPORTANT: First check if the student's answer is actually relevant to the questions. If it's completely off-topic or from a different subject, give a score of 0.
- Grade according to CBSE marking scheme standards
- Total marks for this answer: ${totalMarks}
- Allocate marks per question according to the confirmed marks distribution
- Be fair and consistent in your evaluation
- Verify that the answer addresses the specific concepts from the question paper

Return your assessment as a valid JSON object with these fields:
{
  "score": (a number from 0 to ${totalMarks} representing the total score, give 0 if content is irrelevant to the subject),
  "feedback": (professional explanation of the grade with specific examples from the student's work, note if content is irrelevant),
  "strengths": (array of 3-4 specific strengths demonstrated in the work, or ["None"] if completely irrelevant),
  "areas_for_improvement": (array of 3-4 specific areas needing improvement),
  "suggested_points": (array of 2-3 actionable suggestions for improvement),
  "correct_concepts": (key concepts the student understood correctly, or "None" if irrelevant),
  "misconceptions": (any evident misconceptions in the student's answer),
  "conceptsScore": (a number from 0-10 rating conceptual understanding),
  "diagramScore": (a number from 0-10 rating diagram accuracy if applicable),
  "applicationScore": (a number from 0-10 rating application of theories),
  "terminologyScore": (a number from 0-10 rating use of terminology),
  "is_relevant": (true or false, indicating if the answer is relevant to the subject and question)
}

Ensure your response is ONLY valid JSON without any additional text.
`;
      console.log(`Sending to AI for grading...`);
      let gradingResponseText = await openaiService.generateText(
        gradingPrompt,
        0.2
      );
      // Clean up JSON if it's wrapped in markdown code blocks
      if (gradingResponseText.includes("```json")) {
        gradingResponseText = gradingResponseText.replace(
          /```json\s*|\s*```/g,
          ""
        );
      }

      try {
        // Parse the grading result
        const gradingResult = JSON.parse(gradingResponseText) as GradingResult;

        // Add metadata
        gradingResult.outOf = totalMarks;
        gradingResult.percentage = (gradingResult.score / totalMarks) * 100;
        gradingResult.gradingApproach = GradingApproach.CBSE_STANDARD;
        gradingResult.timeGraded = new Date();
        gradingResult.cbseClass = classString;
        gradingResult.cbseSubject = subjectString;

        // Update session with completed status
        const previousResults = session.previousGradingResults || [];
        sessionStore.updateSession(userId, {
          step: ConversationStep.COMPLETE,
          previousGradingResults: [...previousResults, gradingResult],
        });

        // Format the response with CBSE-specific grading
        const formattedGrading = this.formatCbseGradingResponse(
          gradingResult,
          totalMarks,
          session
        );

        return await this.generateConversationalResponse(
          userId,
          "I've uploaded the student answer",
          `The grading has been completed. Please incorporate this assessment into your response:
          
${formattedGrading}

Respond conversationally, but make sure to include all the assessment details.`
        );
      } catch (error) {
        console.error("Error parsing grading result:", error);

        // Fallback to simple grading
        const fallbackResult = this.getFallbackCbseGrading(
          studentAnswer,
          totalMarks,
          session.questionPaper || "",
          session.subjectArea
        );

        // Update session
        const previousResults = session.previousGradingResults || [];
        sessionStore.updateSession(userId, {
          step: ConversationStep.COMPLETE,
          previousGradingResults: [...previousResults, fallbackResult],
        });

        // Format the fallback response
        const formattedFallback = this.formatCbseGradingResponse(
          fallbackResult,
          totalMarks,
          session
        );

        return await this.generateConversationalResponse(
          userId,
          "I've uploaded the student answer",
          `I had some trouble with the detailed grading, but I've prepared a basic assessment:
          
${formattedFallback}

Respond conversationally while including this assessment.`
        );
      }
    } catch (error) {
      console.error(`Error processing student answer:`, error);

      // Reset to appropriate state to allow retrying
      sessionStore.updateSession(userId, {
        step: ConversationStep.WAITING_FOR_STUDENT_ANSWER,
      });

      return await this.generateConversationalResponse(
        userId,
        "I've uploaded the student answer",
        "There was a problem processing the student's answer. Ask the user to try uploading it again with better lighting or clarity."
      );
    }
  }

  /**
   * Format the grading result with CBSE-specific format but in a more conversational style
   */
  private formatCbseGradingResponse(
    result: GradingResult,
    totalMarks: number,
    session: CbseSessionData
  ): string {
    const subjectName =
      session.subjectArea?.replace("_", " ").toUpperCase() || "GENERAL";
    const className = session.classLevel?.replace("_", " ").toUpperCase() || "";

    // Add a check for completely irrelevant answers
    const isRelevant = result.is_relevant !== false && result.score > 0;

    let formattedResponse = `
  ## CBSE ${className} ${subjectName} ASSESSMENT
  `;

    if (!isRelevant) {
      formattedResponse += `
  ⚠️ **IRRELEVANT CONTENT DETECTED**
  🏆 **TOTAL SCORE: 0/${totalMarks}** (0%)
  
  📝 **EXAMINER'S REMARKS:**
  ${result.feedback}
  
  The submitted answer does not address the question/subject matter. The content appears to be related to a different subject or topic entirely.
  
  💡 **SUGGESTIONS:**
  - Please ensure you're answering the correct question
  - Review the ${subjectName} syllabus and course materials
  - Practice understanding question requirements before answering
  `;
    } else {
      // Original formatting for relevant answers
      formattedResponse += `
  🏆 **TOTAL SCORE: ${result.score}/${totalMarks}** (${Math.round(
        result.percentage
      )}%)
  
  📝 **EXAMINER'S REMARKS:**
  ${result.feedback}
  
  💪 **STRENGTHS:**
  ${result.strengths.map((s) => `- ${s}`).join("\n")}
  
  🔍 **AREAS FOR IMPROVEMENT:**
  ${result.areas_for_improvement.map((a) => `- ${a}`).join("\n")}
  
  💡 **SUGGESTIONS TO IMPROVE:**
  ${result.suggested_points.map((s) => `- ${s}`).join("\n")}
  `;

      // Add economics-specific feedback if applicable
      if (session.subjectArea === SubjectArea.ECONOMICS) {
        formattedResponse += `
      📊 **ECONOMICS-SPECIFIC FEEDBACK:**
      - Economic Concepts: ${
        result.conceptsScore || "Not explicitly evaluated"
      }/10
      - Diagram Accuracy: ${
        result.diagramScore || "Not explicitly evaluated"
      }/10
      - Application of Theories: ${
        result.applicationScore || "Not explicitly evaluated"
      }/10
      - Use of Terminology: ${
        result.terminologyScore || "Not explicitly evaluated"
      }/10
      `;
      } else if (session.subjectArea === SubjectArea.ACCOUNTANCY) {
        formattedResponse += `
      📒 **ACCOUNTANCY-SPECIFIC FEEDBACK:**
      - Accounting Principles: ${
        result.conceptsScore || "Not explicitly evaluated"
      }/10
      - Numerical Accuracy: ${
        result.diagramScore || "Not explicitly evaluated"
      }/10
      - Format & Presentation: ${
        result.applicationScore || "Not explicitly evaluated"
      }/10
      - Financial Terminology: ${
        result.terminologyScore || "Not explicitly evaluated"
      }/10
      `;
      } else if (session.subjectArea === SubjectArea.BUSINESS_STUDIES) {
        formattedResponse += `
      💼 **BUSINESS STUDIES-SPECIFIC FEEDBACK:**
      - Management Concepts: ${
        result.conceptsScore || "Not explicitly evaluated"
      }/10
      - Case Application: ${
        result.diagramScore || "Not explicitly evaluated"
      }/10
      - Business Examples: ${
        result.applicationScore || "Not explicitly evaluated"
      }/10
      - Business Terminology: ${
        result.terminologyScore || "Not explicitly evaluated"
      }/10
      `;
      }
    }

    return formattedResponse;
  }

  /**
   * Fallback grading specifically for CBSE with more human-like assessment
   */
  private getFallbackCbseGrading(
    studentAnswer: string,
    totalMarks: number,
    _questionPaper: string,
    subjectArea: SubjectArea | null
  ): GradingResult {
    // Calculate a score based on length and structure
    const textLength = studentAnswer.length;
    const sentenceCount = (studentAnswer.match(/[.!?]+\s/g) || []).length + 1;
    const wordCount = studentAnswer.split(/\s+/).length;

    let score = Math.min(
      Math.round((textLength / 300) * totalMarks),
      totalMarks
    );
    score = Math.min(
      Math.max(Math.round(totalMarks * 0.4), score),
      Math.round(totalMarks * 0.8)
    );

    // Generate subject-specific feedback
    // Generate subject-specific feedback
    let subjectSpecificStrengths: string[] = [];
    let subjectSpecificAreas: string[] = [];

    if (subjectArea === SubjectArea.ECONOMICS) {
      subjectSpecificStrengths = [
        "Shows understanding of fundamental economic concepts",
        "Attempts to connect economic theory to real-world examples",
      ];

      subjectSpecificAreas = [
        "Could develop economic terminology more precisely",
        "Economic diagrams would benefit from clearer labeling",
      ];
    } else if (subjectArea === SubjectArea.ACCOUNTANCY) {
      subjectSpecificStrengths = [
        "Demonstrates basic understanding of accounting principles",
        "Shows an attempt at proper accounting formats",
      ];

      subjectSpecificAreas = [
        "Numerical procedures could be more clearly presented",
        "Financial statement formats need more precision",
      ];
    } else if (subjectArea === SubjectArea.BUSINESS_STUDIES) {
      subjectSpecificStrengths = [
        "Shows familiarity with business management concepts",
        "Attempts to relate theory to business scenarios",
      ];

      subjectSpecificAreas = [
        "Could use more specific business examples",
        "Management principles could be applied more effectively",
      ];
    } else if (subjectArea === SubjectArea.MATH) {
      subjectSpecificStrengths = [
        "Demonstrates basic mathematical problem-solving skills",
        "Shows work in a somewhat organized manner",
      ];

      subjectSpecificAreas = [
        "Step-by-step workings could be more clearly presented",
        "Mathematical notation could be more precise",
      ];
    } else if (subjectArea === SubjectArea.SCIENCE) {
      subjectSpecificStrengths = [
        "Demonstrates basic understanding of scientific concepts",
        "Attempts to use scientific terminology appropriately",
      ];

      subjectSpecificAreas = [
        "Scientific explanations could be more thorough",
        "Could better connect theory to experimental evidence",
      ];
    }

    // Combine with general strengths
    const strengths = [
      ...(wordCount > 100
        ? ["Provides a substantive response with reasonable detail"]
        : []),
      ...(sentenceCount > 5
        ? ["Organizes thoughts in a somewhat structured manner"]
        : []),
      "Makes an effort to address the key points in the question",
      ...subjectSpecificStrengths,
    ].slice(0, 4);

    const areasForImprovement = [
      "Could benefit from more detailed explanations",
      "Additional specific examples would strengthen the answer",
      "More explicit connections to the core question would improve clarity",
      ...subjectSpecificAreas,
    ].slice(0, 4);

    // Calculate percentage
    const scorePercentage = (score / totalMarks) * 100;

    return {
      score,
      outOf: totalMarks,
      percentage: scorePercentage,
      feedback: `This answer shows a basic understanding of the concepts covered in the question. The response addresses some key points but would benefit from more comprehensive and detailed explanations. According to CBSE standards, the answer demonstrates partial mastery of the required knowledge.`,
      strengths,
      areas_for_improvement: areasForImprovement,
      suggested_points: [
        "Review NCERT textbooks to strengthen conceptual understanding",
        "Practice more detailed explanations with specific examples",
        "Focus on making clearer connections between concepts and applications",
      ],
      correct_concepts:
        "The response shows a foundation of understanding related to the core concepts of the topic.",
      misconceptions:
        "There are some minor misconceptions that could be addressed with more precise explanations and examples.",
      gradingApproach: GradingApproach.CBSE_STANDARD,
      timeGraded: new Date(),

      // CBSE-specific fields
      cbseClass: subjectArea?.toString() || "general",
      cbseSubject: subjectArea?.toString() || "general",
      conceptsScore: Math.round((scorePercentage * 0.6) / 10),
      diagramScore: Math.round((scorePercentage * 0.5) / 10),
      applicationScore: Math.round((scorePercentage * 0.55) / 10),
      terminologyScore: Math.round((scorePercentage * 0.65) / 10),
      critical_thinking: Math.round((scorePercentage * 0.6) / 10),
      organization: Math.round((scorePercentage * 0.7) / 10),
      language_use: Math.round((scorePercentage * 0.65) / 10),
      concept_application: Math.round((scorePercentage * 0.5) / 10),
    };
  }

  /**
   * Extract questions and their marks from OCR text
   */
  private extractQuestionMarks(ocrText: string): {
    marks: Map<number, number>;
    questionText: Map<number, string>;
  } {
    console.log("Extracting question marks from OCR text");
    const questionMarks = new Map<number, number>();
    const questionText = new Map<number, string>();
    const lines = ocrText.split("\n");

    // Find total marks in header
    let totalMarksInHeader = 0;
    const mmPattern =
      /MM:\s*(\d+)|Maximum\s+Marks\s*:?\s*(\d+)|MM\s*(\d+)|Total\s+Marks\s*:?\s*(\d+)/i;
    for (let i = 0; i < Math.min(15, lines.length); i++) {
      const mmMatch = lines[i].match(mmPattern);
      if (mmMatch) {
        totalMarksInHeader = parseInt(
          mmMatch[1] || mmMatch[2] || mmMatch[3] || mmMatch[4],
          10
        );
        console.log(`Total marks found in header: ${totalMarksInHeader}`);
        break;
      }
    }

    // Pattern to find question numbers and marks
    const questionPattern =
      /^\s*(\d+)\s*\.\s*(.*?)(?:\s*\[(\d+)\s*(?:marks?|m)\]|\s*\((\d+)\s*(?:marks?|m)\)|\s+(\d+)\s*(?:marks?|m))?$/i;

    let currentQuestionNumber = 0;
    let currentQuestionText = "";

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.length === 0) continue;

      const questionMatch = line.match(questionPattern);
      if (questionMatch) {
        // Save previous question if there was one
        if (currentQuestionNumber > 0) {
          questionText.set(currentQuestionNumber, currentQuestionText.trim());
        }

        // Get new question details
        currentQuestionNumber = parseInt(questionMatch[1], 10);
        currentQuestionText = questionMatch[2] || "";

        // Get marks if they're in the question line
        const marksValue =
          questionMatch[3] || questionMatch[4] || questionMatch[5];
        if (marksValue) {
          questionMarks.set(currentQuestionNumber, parseInt(marksValue, 10));
        }
      } else if (currentQuestionNumber > 0) {
        // This might be a continuation of the current question
        // Look for marks pattern at the end
        const marksPattern =
          /\s*\[(\d+)\s*(?:marks?|m)\]|\s*\((\d+)\s*(?:marks?|m)\)|\s+(\d+)\s*(?:marks?|m)$/i;
        const marksMatch = line.match(marksPattern);

        if (marksMatch && !questionMarks.has(currentQuestionNumber)) {
          const marksValue = marksMatch[1] || marksMatch[2] || marksMatch[3];
          questionMarks.set(currentQuestionNumber, parseInt(marksValue, 10));
          currentQuestionText += " " + line.replace(marksPattern, "");
        } else {
          currentQuestionText += " " + line;
        }
      }
    }

    // Save the last question
    if (currentQuestionNumber > 0) {
      questionText.set(currentQuestionNumber, currentQuestionText.trim());
    }

    // Special case for Economics papers that often follow a standard format
    if (totalMarksInHeader === 40 && questionMarks.size === 0) {
      console.log(
        "Applying special case for CBSE Economics paper with 40 marks"
      );

      // Check if the extracted questions match the standard format
      // Standard format: Questions 1-5 (2 marks each), 6-10 (3 marks each), 11-13 (5 marks each)

      if (questionText.has(1) && (questionText.has(5) || questionText.has(6))) {
        // Apply the standard distribution: first set of questions (usually 1-5) at 2 marks each
        for (let q = 1; q <= 5; q++) {
          if (questionText.has(q)) {
            questionMarks.set(q, 2);
          }
        }
      }

      if (
        questionText.has(6) &&
        (questionText.has(10) || questionText.has(11))
      ) {
        // Apply the standard distribution: second set of questions (usually 6-10) at 3 marks each
        for (let q = 6; q <= 10; q++) {
          if (questionText.has(q)) {
            questionMarks.set(q, 3);
          }
        }
      }

      if (questionText.has(11) && questionText.has(12)) {
        // Apply the standard distribution: third set of questions (usually 11-13) at 5 marks each
        for (let q = 11; q <= 13; q++) {
          if (questionText.has(q)) {
            questionMarks.set(q, 5);
          }
        }
      }
    }

    // Log the results
    console.log(
      `Extracted ${questionMarks.size} questions with marks and ${questionText.size} with text`
    );

    return { marks: questionMarks, questionText: questionText };
  }

  /**
   * Process an uploaded image with GPT-4 Vision
   */
  async processImageUpload(userId: string, imagePath: string): Promise<string> {
    const session = this.getOrCreateSession(userId);
    console.log(
      `Processing image upload for user ${userId}, session state: ${session.step}`
    );

    // Upload the image to get a URL
    console.log(`Uploading image to Firebase Storage: ${imagePath}`);
    const imageUrl = await storageService.uploadFile(imagePath);

    // Handle image based on current session state in CBSE flow
    switch (session.step) {
      case ConversationStep.WAITING_FOR_QUESTION_PAPER:
        // Process as question paper and return conversational response
        return this.processQuestionPaperUrl(userId, imageUrl);

      case ConversationStep.WAITING_FOR_STUDENT_ANSWER:
        // Process as student answer and return conversational response
        return this.processStudentAnswerUrl(userId, imageUrl);

      case ConversationStep.COMPLETE:
      case ConversationStep.FOLLOW_UP:
        // Start a new session but keep it conversational
        sessionStore.resetSession(userId);
        sessionStore.updateSession(userId, {
          step: ConversationStep.WAITING_FOR_CLASS,
        });

        return await this.generateConversationalResponse(
          userId,
          "I've uploaded an image",
          "The user has uploaded an image after completing a previous assessment. Start a fresh grading session and ask for the class level."
        );

      default:
        // If not in expected state, guide user to correct flow conversationally
        sessionStore.updateSession(userId, {
          step: ConversationStep.WAITING_FOR_CLASS,
        });

        return await this.generateConversationalResponse(
          userId,
          "I've uploaded an image",
          "The user has uploaded an image at an unexpected point in the workflow. Gently guide them to follow the proper sequence, starting with the class level."
        );
    }
  }

  /**
   * Process an image from URL with GPT-4 Vision
   */
  async processImageFromUrl(userId: string, imageUrl: string): Promise<string> {
    const session = this.getOrCreateSession(userId);
    console.log(
      `Processing image from URL for user ${userId}, session state: ${session.step}`
    );

    // Handle image based on current session state in CBSE flow
    switch (session.step) {
      case ConversationStep.WAITING_FOR_QUESTION_PAPER:
        console.log("Directing to processQuestionPaperUrl");
        return this.processQuestionPaperUrl(userId, imageUrl);

      case ConversationStep.WAITING_FOR_STUDENT_ANSWER:
        console.log("Directing to processStudentAnswerUrl");
        return this.processStudentAnswerUrl(userId, imageUrl);

      default:
        console.log(
          `Unexpected session state for image upload: ${session.step}`
        );
        // If not in expected state, guide user conversationally
        sessionStore.updateSession(userId, {
          step: ConversationStep.WAITING_FOR_CLASS,
        });

        return await this.generateConversationalResponse(
          userId,
          "I've uploaded an image",
          "The user has uploaded an image at an unexpected point in the workflow. Gently guide them to follow the proper sequence, starting with the class level."
        );
    }
  }
}
