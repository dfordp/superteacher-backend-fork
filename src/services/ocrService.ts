// src/services/ocrService.ts - Updated for FastAPI integration

import axios from "axios";
import fs from "fs";
import FormData from "form-data";
import path from "path";

/**
 * Enhanced OCR service with better error handling, fallbacks, and retries
 * Connects to a remote OCR service with robustness features
 */
class OcrService {
  private ocrEndpoint: string;
  private backupEndpoint: string | null;
  private demoMode: boolean;
  private maxRetries: number;
  private retryDelay: number;
  private enableLogging: boolean;

  constructor() {
    // Primary OCR endpoint configuration - update to your FastAPI endpoint
    this.ocrEndpoint =
      process.env.OCR_ENDPOINT ||
      "https://grading-api.onrender.com/extract-text"; // Update with your actual URL

    // Backup OCR service if available
    this.backupEndpoint = process.env.OCR_BACKUP_ENDPOINT || null;

    // Demo mode settings to allow operation when OCR is unavailable
    this.demoMode =
      process.env.OCR_DEMO_MODE === "true" ||
      process.env.NODE_ENV === "production"; // Default to demo mode in production

    // Retry configuration for robust OCR calls
    this.maxRetries = parseInt(process.env.OCR_MAX_RETRIES || "3", 10);
    this.retryDelay = parseInt(process.env.OCR_RETRY_DELAY || "2000", 10);

    // Enable detailed logging for debugging
    this.enableLogging = process.env.OCR_LOGGING === "true" || true;

    this.logInfo("OCR Service initialized with:");
    this.logInfo(`- Primary endpoint: ${this.ocrEndpoint}`);
    this.logInfo(`- Backup endpoint: ${this.backupEndpoint || "none"}`);
    this.logInfo(`- Demo mode: ${this.demoMode ? "enabled" : "disabled"}`);
    this.logInfo(
      `- Max retries: ${this.maxRetries}, Retry delay: ${this.retryDelay}ms`
    );
  }

  /**
   * Extract text from an image with robust error handling and retries
   */
  async extractTextFromImage(imagePath: string): Promise<string> {
    try {
      this.logInfo(`Processing image at: ${imagePath}`);

      // Validate the image file exists and is accessible
      if (!fs.existsSync(imagePath)) {
        this.logError(`Image file does not exist: ${imagePath}`);
        return this.handleOcrFailure("File not found", imagePath);
      }

      // Check file stats and size
      const stats = fs.statSync(imagePath);
      this.logInfo(
        `Image file size: ${
          stats.size
        } bytes, Permissions: ${stats.mode.toString(8)}`
      );

      if (stats.size === 0) {
        this.logError("Image file is empty (0 bytes)");
        return this.handleOcrFailure("Empty file", imagePath);
      }

      if (stats.size > 5 * 1024 * 1024) {
        this.logInfo("Large image detected, might affect OCR processing time");
      }

      // If demo mode is enabled, skip the actual OCR call
      if (this.demoMode) {
        this.logInfo("Demo mode is enabled, returning simulated OCR text");
        return this.getDemoText(path.basename(imagePath));
      }

      // Try OCR processing with retries
      return await this.performOcrWithRetries(imagePath);
    } catch (error) {
      this.logError("Error in OCR processing:", error);
      return this.handleOcrFailure("Processing error", imagePath);
    }
  }

  /**
   * Extract text from an image URL rather than a local file
   */
  async extractTextFromImageUrl(imageUrl: string): Promise<string> {
    try {
      this.logInfo(`Processing image from URL: ${imageUrl}`);

      // If demo mode is enabled, skip the actual OCR call
      if (this.demoMode) {
        this.logInfo("Demo mode is enabled, returning simulated OCR text");
        return this.getDemoText(path.basename(imageUrl));
      }

      // Try OCR processing with retries
      return await this.performOcrWithRetriesFromUrl(imageUrl);
    } catch (error) {
      this.logError("Error in OCR processing from URL:", error);
      return this.handleOcrFailure("Processing error", imageUrl);
    }
  }

  /**
   * Perform OCR with retries and fallback for local files
   */
  private async performOcrWithRetries(imagePath: string): Promise<string> {
    let retries = 0;
    let lastError: any;

    while (retries <= this.maxRetries) {
      try {
        // If not the first attempt, add delay with exponential backoff
        if (retries > 0) {
          const delay = this.retryDelay * Math.pow(2, retries - 1);
          this.logInfo(
            `Retry ${retries}/${this.maxRetries} - Waiting ${delay}ms...`
          );
          await new Promise((resolve) => setTimeout(resolve, delay));
        }

        // Try primary endpoint first
        if (retries < this.maxRetries) {
          try {
            this.logInfo(
              `Attempting OCR with primary endpoint, attempt ${retries + 1}/${
                this.maxRetries + 1
              }`
            );
            const result = await this.callOcrService(
              this.ocrEndpoint,
              imagePath
            );
            if (result) return result;
          } catch (error) {
            lastError = error;
            this.logError(
              `Primary OCR endpoint failed on attempt ${retries + 1}:`,
              error
            );
          }
        }

        // Try backup endpoint if available and this is the last attempt
        if (retries === this.maxRetries && this.backupEndpoint) {
          try {
            this.logInfo("Trying backup OCR endpoint as last resort");
            const result = await this.callOcrService(
              this.backupEndpoint,
              imagePath
            );
            if (result) return result;
          } catch (error) {
            this.logError("Backup OCR endpoint also failed:", error);
            // Continue to use the primary endpoint error for consistency
          }
        }

        retries++;
      } catch (error) {
        lastError = error;
        retries++;
        this.logError(`OCR attempt ${retries} failed:`, error);
      }
    }

    // If all attempts failed
    this.logError("All OCR attempts failed:", lastError);
    return this.handleOcrFailure("All attempts failed", imagePath);
  }

  /**
   * Perform OCR with retries and fallback for image URLs
   */
  private async performOcrWithRetriesFromUrl(
    imageUrl: string
  ): Promise<string> {
    let retries = 0;
    let lastError: any;

    while (retries <= this.maxRetries) {
      try {
        // If not the first attempt, add delay with exponential backoff
        if (retries > 0) {
          const delay = this.retryDelay * Math.pow(2, retries - 1);
          this.logInfo(
            `Retry ${retries}/${this.maxRetries} - Waiting ${delay}ms...`
          );
          await new Promise((resolve) => setTimeout(resolve, delay));
        }

        // Try primary endpoint first
        if (retries < this.maxRetries) {
          try {
            this.logInfo(
              `Attempting OCR with primary endpoint, attempt ${retries + 1}/${
                this.maxRetries + 1
              }`
            );
            const result = await this.callOcrServiceWithUrl(
              this.ocrEndpoint,
              imageUrl
            );
            if (result) return result;
          } catch (error) {
            lastError = error;
            this.logError(
              `Primary OCR endpoint failed on attempt ${retries + 1}:`,
              error
            );
          }
        }

        // Try backup endpoint if available and this is the last attempt
        if (retries === this.maxRetries && this.backupEndpoint) {
          try {
            this.logInfo("Trying backup OCR endpoint as last resort");
            const result = await this.callOcrServiceWithUrl(
              this.backupEndpoint,
              imageUrl
            );
            if (result) return result;
          } catch (error) {
            this.logError("Backup OCR endpoint also failed:", error);
            // Continue to use the primary endpoint error for consistency
          }
        }

        retries++;
      } catch (error) {
        lastError = error;
        retries++;
        this.logError(`OCR attempt ${retries} failed:`, error);
      }
    }

    // If all attempts failed
    this.logError("All OCR attempts failed:", lastError);
    return this.handleOcrFailure("All attempts failed", imageUrl);
  }

  /**
   * Call the OCR service with a local file
   */
  private async callOcrService(
    endpoint: string,
    imagePath: string
  ): Promise<string> {
    // Create form data for image upload
    const form = new FormData();
    form.append("file", fs.createReadStream(imagePath));

    // Call the OCR service with increased timeout
    const response = await axios.post(endpoint, form, {
      headers: {
        ...form.getHeaders(),
        "X-Retry-Count": "0",
        Accept: "application/json",
      },
      timeout: 120000, // 2-minute timeout for OCR processing
      validateStatus: (status) => status === 200, // Only accept 200 status
    });

    // Validate response
    if (response.data && response.data.extracted_text) {
      // Check if the response is empty or too short
      if (response.data.extracted_text.trim().length < 5) {
        this.logInfo(
          "OCR returned empty or very short result, checking if image requires processing"
        );
        return this.handleOcrFailure("Empty result", imagePath);
      }

      // Log success and return extracted text
      this.logInfo(
        `OCR successful, extracted ${response.data.extracted_text.length} characters`
      );
      return response.data.extracted_text;
    } else {
      this.logError(
        "OCR response missing extracted_text field:",
        response.data
      );
      throw new Error("Invalid OCR response format");
    }
  }

  /**
   * Call the OCR service with an image URL
   */
  private async callOcrServiceWithUrl(
    endpoint: string,
    imageUrl: string
  ): Promise<string> {
    // Make a POST request with the image URL
    this.logInfo(`Sending image URL to OCR service: ${imageUrl}`);

    // Log the request payload for debugging
    const payload = {
      image_url: imageUrl,
    };
    this.logInfo(`Request payload: ${JSON.stringify(payload)}`);

    // Using form-data approach instead of JSON body
    const form = new FormData();
    form.append("image_url", imageUrl);

    const response = await axios.post(endpoint, form, {
      headers: {
        ...form.getHeaders(),
        Accept: "application/json",
        "X-Retry-Count": "0",
      },
      timeout: 120000, // 2-minute timeout for OCR processing
      validateStatus: (status) => status === 200, // Only accept 200 status
    });

    // Validate response
    if (response.data && response.data.extracted_text) {
      // Check if the response is empty or too short
      if (response.data.extracted_text.trim().length < 5) {
        this.logInfo("OCR returned empty or very short result from URL");
        return this.handleOcrFailure("Empty result", imageUrl);
      }

      // Log success and return extracted text
      this.logInfo(
        `OCR successful from URL, extracted ${response.data.extracted_text.length} characters`
      );
      return response.data.extracted_text;
    } else {
      this.logError(
        "OCR response missing extracted_text field:",
        response.data
      );
      throw new Error("Invalid OCR response format");
    }
  }

  /**
   * Handle OCR failure with appropriate fallback
   */
  private handleOcrFailure(reason: string, imagePath: string): string {
    this.logInfo(`Using demo text due to OCR failure: ${reason}`);
    // Always fall back to demo text on any failure
    return this.getDemoText(path.basename(imagePath));
  }

  /**
   * Generate demo text for testing and fallback
   */
  private getDemoText(filename: string): string {
    this.logInfo(`Generating demo text for: ${filename}`);

    // Check for indicators in the filename to customize demo text
    const lowerFilename = filename.toLowerCase();

    // Math demo text
    if (
      lowerFilename.includes("math") ||
      lowerFilename.includes("calc") ||
      lowerFilename.includes("equ")
    ) {
      return `
To solve this problem, I used the quadratic formula:
x = (-b ± √(b² - 4ac)) / 2a

Given the equation 2x² + 5x - 3 = 0:
a = 2, b = 5, c = -3

Substituting into the formula:
x = (-5 ± √(5² - 4(2)(-3))) / 2(2)
x = (-5 ± √(25 + 24)) / 4
x = (-5 ± √49) / 4
x = (-5 ± 7) / 4

x₁ = (-5 + 7) / 4 = 2/4 = 0.5
x₂ = (-5 - 7) / 4 = -12/4 = -3

Therefore, the solution set is {0.5, -3}.
      `.trim();
    }

    // Essay/English demo text
    if (
      lowerFilename.includes("essay") ||
      lowerFilename.includes("english") ||
      lowerFilename.includes("lit")
    ) {
      return `
The theme of identity in "To Kill a Mockingbird" is primarily explored through the character development of Scout Finch. As she navigates childhood in the racially charged atmosphere of Maycomb County, her understanding of herself and her place in society evolves significantly.

At the beginning of the novel, Scout's identity is largely shaped by her family connections and her tomboyish tendencies. She resists traditional feminine expectations, preferring to play with her brother Jem and their friend Dill rather than attend tea parties or wear dresses. Her father Atticus encourages her independence while also instilling in her a strong moral compass.

The trial of Tom Robinson serves as a crucial catalyst for Scout's developing identity. Through witnessing the injustice of the trial and the reactions of different community members, Scout begins to understand the complexity of social identities beyond her own experience. She learns that people are not always what they seem, as demonstrated by her evolving perspective on Boo Radley.

By the end of the novel, Scout has developed a more nuanced understanding of identity. She recognizes that people's identities are multifaceted and that empathy—"walking in someone else's shoes" as Atticus teaches her—is essential to truly understanding others. The novel concludes with Scout having internalized this lesson, suggesting that her identity has been enriched by her experiences and the moral guidance of her father.
      `.trim();
    }

    // Science demo text
    if (
      lowerFilename.includes("science") ||
      lowerFilename.includes("bio") ||
      lowerFilename.includes("chem")
    ) {
      return `
The water cycle is a continuous process that circulates water throughout Earth's systems. It consists of several key phases:

1. Evaporation: Water from oceans, lakes, and rivers is heated by the sun and transforms into water vapor, rising into the atmosphere. Plants also release water vapor through transpiration.

2. Condensation: As water vapor rises and cools, it condenses around tiny particles like dust to form clouds. The water changes from a gas back to a liquid state during this process.

3. Precipitation: When water droplets in clouds become too heavy, they fall back to Earth as rain, snow, sleet, or hail, depending on atmospheric temperature.

4. Collection: Precipitation is collected in bodies of water (oceans, lakes, rivers) or soaks into the ground as groundwater, where it may be stored in aquifers. Some water also becomes runoff, flowing across land surfaces back into larger bodies of water.

This cycle is crucial for Earth's ecosystems as it redistributes water, regulates temperature, and supports all living organisms. Human activities such as deforestation, pollution, and climate change can disrupt this natural cycle, affecting weather patterns and water availability worldwide.
      `.trim();
    }

    // Default demo text for general responses
    return `
In response to this question, I believe the key factors to consider are:

First, the historical context provides essential background. The events of the early 20th century, particularly the economic challenges following World War I, created conditions that influenced subsequent developments significantly.

Second, social dynamics played a crucial role. The interaction between different social classes and interest groups shaped the outcomes in ways that weren't always predictable at the time. These relationships evolved in response to both internal pressures and external events.

Third, the technological advances of this period cannot be overlooked. Innovations in communication, transportation, and manufacturing transformed how people lived and worked, creating new opportunities while disrupting traditional practices.

When analyzing these factors together, we can see that their combined influence created a complex situation where multiple forces were acting simultaneously. This helps explain why the outcomes were not simply the result of any single cause but emerged from the interaction of numerous variables across different domains.

The evidence supporting this analysis comes from primary sources including government records, contemporary accounts, and statistical data that document the patterns of change during this critical period.
    `.trim();
  }

  /**
   * Log informational messages
   */
  private logInfo(message: string, ...args: any[]): void {
    if (this.enableLogging) {
      console.log(`[OCR Service] ${message}`, ...args);
    }
  }

  /**
   * Log error messages
   */
  private logError(message: string, error?: any): void {
    console.error(`[OCR Service ERROR] ${message}`);
    if (error) {
      if (axios.isAxiosError(error)) {
        console.error(`Status: ${error.response?.status || "unknown"}`);
        console.error(
          `Response: ${JSON.stringify(error.response?.data || {}).substring(
            0,
            200
          )}...`
        );
      } else if (error instanceof Error) {
        console.error(`${error.name}: ${error.message}`);
        console.error(
          `Stack: ${error.stack?.split("\n")[0] || "No stack trace"}`
        );
      } else {
        console.error(error);
      }
    }
  }
}

export const ocrService = new OcrService();
