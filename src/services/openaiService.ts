// src/services/openaiService.ts
import axios from "axios";
import { OPENAI_CONFIG } from "../config/openai";
import { GradingResult, GradingApproach } from "../types";

/**
 * Service for interacting with the OpenAI API
 */
class OpenAIService {
  private apiKey: string;
  private model: string;
  private apiEndpoint: string = "https://api.openai.com/v1/chat/completions";

  constructor() {
    this.apiKey = OPENAI_CONFIG.apiKey;
    this.model = OPENAI_CONFIG.model || "gpt-4o";
    console.log(`OpenAI Service initialized with model: ${this.model}`);
  }

  /**
   * Grade a student's answer using OpenAI
   */
  async gradeAnswer(
    question: string,
    studentAnswer: string,
    gradingInstruction: string,
    maxMarks: number
  ): Promise<GradingResult> {
    try {
      console.log(`Grading answer with model: ${this.model}`);
      console.log(`Question length: ${question.length}, Answer length: ${studentAnswer.length}`);
      console.log(`Max marks: ${maxMarks}`);

      const systemPrompt = `
      # CBSE Examination Grading System
      
      You are an expert CBSE examiner with 15+ years of experience grading academic papers. Your assessment must follow official CBSE marking schemes while providing valuable, actionable feedback.
      
      ## Assessment Guidelines
      - Evaluate according to CBSE subject criteria
      - Maximum marks available: ${maxMarks}
      - Apply a balanced approach to grading, rewarding both conceptual understanding and application
      - Ensure consistency with CBSE standards for partial credit
      - Evaluate the answer based EXCLUSIVELY on its academic merit and alignment with the question
      
      ## Specific Grading Instructions
      ${gradingInstruction}
      
      ## Question Context
      ${question || "The student has provided an answer to the assigned question."}
      
      ## Output Format Requirements
      Return ONLY a valid JSON object with the following structure:
      {
        "score": ${maxMarks > 0 ? "number between 0-" + maxMarks : "number"}, 
        "feedback": "detailed explanation of assessment with specific references to answer content",
        "strengths": ["3-4 specific strengths identified in the answer"],
        "areas_for_improvement": ["3-4 specific areas where improvement is needed"],
        "suggested_points": ["2-3 concrete, actionable suggestions for improvement"],
        "correct_concepts": "detailed analysis of concepts correctly understood",
        "misconceptions": "identification of any misconceptions or errors",
        "critical_thinking": number 1-10,
        "organization": number 1-10,
        "language_use": number 1-10,
        "concept_application": number 1-10,
        "is_relevant": boolean (indicates whether answer addresses the question appropriately)
      }
      
      ## IMPORTANT
      - If the answer is completely off-topic or from an unrelated subject area, assign a score of 0 and set is_relevant to false
      - Provide constructive, specific feedback that would help the student improve
      - Base your assessment solely on the content of the response, not on extraneous factors
      - Maintain strict objectivity and fairness in your evaluation
      `;

      const userPrompt = `
QUESTION:
${question}

STUDENT'S ANSWER:
${studentAnswer}

Please grade this answer according to the instructions.
`;

      const response = await axios.post(
        this.apiEndpoint,
        {
          model: this.model,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          max_tokens: 2000,
          temperature: OPENAI_CONFIG.temperature,
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
        }
      );

      // Parse the response as JSON
      const aiResponseContent = response.data.choices[0].message.content;
      
      try {
        const gradingResult = JSON.parse(aiResponseContent) as GradingResult;
        gradingResult.timeGraded = new Date();
        gradingResult.gradingApproach = GradingApproach.BALANCED;
        
        return gradingResult;
      } catch (parseError) {
        console.error("Error parsing AI response as JSON:", parseError);
        
        // Extract score if possible using regex
        const scoreMatch = aiResponseContent.match(/score["']?\s*:\s*(\d+)/i);
        const score = scoreMatch ? parseInt(scoreMatch[1], 10) : Math.floor(maxMarks / 2);
        
        // Create a fallback result
        return {
          score,
          outOf: maxMarks,
          percentage: (score / maxMarks) * 100,
          feedback: aiResponseContent.substring(0, 500),
          strengths: ["The student demonstrates some understanding of the topic"],
          areas_for_improvement: ["More detail and structure would improve the answer"],
          suggested_points: ["Review the key concepts and provide more examples"],
          correct_concepts: "Some basic understanding is demonstrated",
          misconceptions: "There may be some misconceptions that need addressing",
          gradingApproach: GradingApproach.BALANCED,
          timeGraded: new Date(),
          critical_thinking: 5,
          organization: 5,
          language_use: 5,
          concept_application: 5
        };
      }
    } catch (error) {
      console.error("Error in OpenAI API call:", error);
      throw error;
    }
  }

  /**
   * Generate a conversational response
   */
  async generateConversation(
    systemPrompt: string,
    userMessage: string,
    history: { role: string; content: string }[] = []
  ): Promise<string> {  
    try {
      console.log(`Generating conversational response with model: ${this.model}`);
      
      // Prepare messages array with system prompt, history and user message
      const messages = [
        { role: "system", content: systemPrompt },
        ...history,
        { role: "user", content: userMessage }
      ];
      
      // Call the API
      const response = await axios.post(
        this.apiEndpoint,
        {
          model: this.model,
          messages: messages,
          max_tokens: 2000,
          temperature: 0.7, // Higher for more natural conversation
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
          timeout: 60000, // 1 minute timeout
        }
      );
      
      console.log(`Received conversational response from AI API`);
      return response.data.choices[0].message.content;
    } catch (error) {
      console.error("Error generating conversational response:", error);
      throw error;
    }
  }

  /**
   * Generate text with a specific prompt
   */
  async generateText(
    prompt: string,
    temperature: number = 0.1
  ): Promise<string> {
    try {
      console.log(`Generating text with prompt: ${prompt.substring(0, 50)}...`);
      
      const response = await axios.post(
        this.apiEndpoint,
        {
          model: this.model,
          messages: [
            { role: "user", content: prompt }
          ],
          max_tokens: 2000,
          temperature: temperature,
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
          timeout: 60000,
        }
      );
      
      return response.data.choices[0].message.content;
    } catch (error) {
      console.error("Error generating text:", error);
      throw error;
    }
  }

  /**
   * Analyze an image using GPT-4 Vision
   */
  async analyzeImage(
    imageUrl: string,
    prompt: string = "Extract all text from this image exactly as it appears."
  ): Promise<string> {
    try {
      console.log(`Analyzing image with GPT-4 Vision: ${imageUrl}`);
      
      const response = await axios.post(
        this.apiEndpoint,
        {
          model: "gpt-4o", // Make sure to use a model that supports vision
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: prompt },
                { type: "image_url", image_url: { url: imageUrl } }
              ]
            }
          ],
          max_tokens: 3000,
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
          timeout: 120000, // 2 minute timeout for image processing
        }
      );
      
      console.log(`Received image analysis from Vision API`);
      return response.data.choices[0].message.content;
    } catch (error) {
      console.error("Error analyzing image:", error);
      throw error;
    }
  }
}

export const openaiService = new OpenAIService();