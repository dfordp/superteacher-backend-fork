import { SessionData, ConversationStep, CbseSessionData } from "../types";

/**
 * A simple in-memory store for user sessions
 * In production, this would be replaced with Redis or another session store
 */
class SessionStore {
  private sessions: Map<string, SessionData>;

  constructor() {
    this.sessions = new Map<string, SessionData>();
  }

  /**
   * Initialize or retrieve a session for a user
   */
  getSession(userId: string): SessionData | CbseSessionData {
    if (!this.sessions.has(userId)) {
      this.sessions.set(userId, {
        userId,
        question: null,
        studentAnswer: null,
        originalImage: null,
        subjectArea: null,
        contextualNotes: null,
        marks: null,
        step: ConversationStep.WAITING_FOR_QUESTION,
        gradingApproach: null,
        lastInteraction: null,
        previousGradingResults: null,
      });
    }

    return this.sessions.get(userId)!;
  }

  /**
   * Update a session with new data
   */
  updateSession(
    userId: string,
    data: Partial<SessionData> | Partial<CbseSessionData>
  ): SessionData {
    const currentSession = this.getSession(userId);
    const updatedSession = { ...currentSession, ...data };
    this.sessions.set(userId, updatedSession);
    return updatedSession;
  }

  /**
   * Reset a session to initial state
   */
  resetSession(userId: string): void {
    this.sessions.set(userId, {
      userId,
      question: null,
      studentAnswer: null,
      originalImage: null,
      subjectArea: null,
      contextualNotes: null,
      marks: null,
      step: ConversationStep.INITIAL,
      gradingApproach: null,
      lastInteraction: null,
      previousGradingResults: [],
    });
  }

  /**
   * Reset a session to initial state
   */
  resetCbseSession(userId: string): void {
    const cbseSession: CbseSessionData = {
      userId,
      question: null,
      studentAnswer: null,
      originalImage: null,
      subjectArea: null,
      contextualNotes: null,
      marks: null,
      step: ConversationStep.INITIAL, // Start at INITIAL for CBSE flow
      gradingApproach: null,
      lastInteraction: null,
      previousGradingResults: [],
      classLevel: null,
      questionPaper: null,
      questionMarks: null,
      isMarkingConfirmed: false,
    };
    this.sessions.set(userId, cbseSession);
  }

  /**
   * Update a CBSE session with new data
   */
  updateCbseSession(
    userId: string,
    data: Partial<CbseSessionData>
  ): CbseSessionData {
    const currentSession = this.getSession(userId) as CbseSessionData;
    // Initialize CBSE fields if they're not there
    if (!currentSession.classLevel) {
      currentSession.classLevel = null;
      currentSession.questionPaper = null;
      currentSession.questionMarks = null;
      currentSession.isMarkingConfirmed = false;
    }
    const updatedSession = { ...currentSession, ...data };
    this.sessions.set(userId, updatedSession);
    return updatedSession as CbseSessionData;
  }
}

// Export a singleton instance
export const sessionStore = new SessionStore();
