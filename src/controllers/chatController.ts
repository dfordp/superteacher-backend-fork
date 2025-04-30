import { Request, Response } from "express";
import { IncomingForm } from "formidable";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import path from "path";
import { storageService } from "../services/storageService";
import { cbseChatService as chatService } from "../server";
// Configure formidable options - use /tmp for Render compatibility
const uploadDir =
  process.env.NODE_ENV === "production"
    ? "/tmp"
    : path.join(__dirname, "../../uploads");
// Ensure uploads directory exists
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

export const chatController = {
  /**
   * Handle chat messages and file uploads
   */
  async handleChatRequest(req: Request, res: Response): Promise<void> {
    try {
      // Check if it's a multipart form (file upload) or regular JSON request
      const contentType = req.headers["content-type"] || "";
      console.log("Content-Type:", contentType);

      if (contentType.includes("multipart/form-data")) {
        // Handle file upload
        await handleFileUpload(req, res);
      } else {
        // Handle text message
        await handleTextMessage(req, res);
      }
    } catch (error) {
      console.error("Error handling chat request:", error);
      res.status(500).json({ error: "Failed to process request" });
    }
  },
};

/**
 * Process a text message request
 */
async function handleTextMessage(req: Request, res: Response): Promise<void> {
  try {
    const { message, userId, image_url } = req.body;

    // Check if this is a text message with an image URL
    if (image_url) {
      const userIdToUse = userId || uuidv4();
      console.log(`Processing image URL for user ${userIdToUse}: ${image_url}`);

      // Process the image URL directly with the OCR service
      const response = await chatService.processImageFromUrl(
        userIdToUse,
        image_url
      );

      // Send the response
      res.json({
        userId: userIdToUse,
        message: response,
      });
      return;
    }

    if (!message) {
      res.status(400).json({ error: "Message is required" });
      return;
    }

    const userIdToUse = userId || uuidv4();
    console.log(`Processing text message for user ${userIdToUse}`);

    // Process the message
    const response = await chatService.processTextMessage(userIdToUse, message);

    // Send the response
    res.json({
      userId: userIdToUse,
      message: response,
    });
  } catch (error) {
    console.error("Error processing text message:", error);
    res.status(500).json({ error: "Failed to process message" });
  }
}

/**
 * Process a file upload request
 */
async function handleFileUpload(req: Request, res: Response): Promise<void> {
  console.log("Starting file upload process...");
  console.log("Upload directory:", uploadDir);

  // Create a temporary file upload handler
  const form = new IncomingForm({
    uploadDir,
    keepExtensions: true,
    maxFileSize: 10 * 1024 * 1024, // 10MB limit
  });

  // Parse the form
  form.parse(req, async (err, fields, files) => {
    if (err) {
      console.error("Error parsing form:", err);
      res.status(400).json({ error: "Failed to upload file" });
      return;
    }

    console.log("Form parsed successfully");
    console.log("Fields received:", JSON.stringify(fields));
    console.log("Files received:", Object.keys(files));

    // Extract user ID from form fields or generate one
    const rawUserId =
      fields.userId &&
      (Array.isArray(fields.userId) ? fields.userId[0] : fields.userId);

    // Make sure we have a valid userId format
    const userId =
      rawUserId && typeof rawUserId === "string" && rawUserId.trim().length > 0
        ? rawUserId.trim()
        : uuidv4();

    console.log(`Processing file upload for user ${userId}`);

    // Check if any file was uploaded
    if (!files || Object.keys(files).length === 0) {
      console.error("No files were uploaded");
      res.status(400).json({ error: "No files were uploaded" });
      return;
    }

    // Process the uploaded file
    let fileField = null;

    // Check for 'image' field first (our primary field name)
    if (files.image) {
      console.log("'image' field found");
      fileField = Array.isArray(files.image) ? files.image[0] : files.image;
    }
    // Then check for 'file' field
    else if (files.file) {
      console.log("'file' field found");
      fileField = Array.isArray(files.file) ? files.file[0] : files.file;
    }
    // If neither specific field is found, try to get the first available file
    else {
      const fileKeys = Object.keys(files);
      console.log(
        `Neither 'image' nor 'file' field found. Available keys: ${fileKeys.join(
          ", "
        )}`
      );

      for (const key of fileKeys) {
        fileField = files[key];
        // If it's an array, get the first element
        if (Array.isArray(fileField) && fileField.length > 0) {
          fileField = fileField[0];
        }

        if (fileField) {
          console.log(`Using file from key: ${key}`);
          break;
        }
      }
    }

    if (!fileField) {
      console.error("Missing or invalid image file");
      res.status(400).json({ error: "Missing or invalid image file" });
      return;
    }

    // @ts-ignore - handle formidable types
    const imagePath = fileField.filepath || fileField.path;
    console.log(`Image path: ${imagePath}`);

    if (!imagePath || !fs.existsSync(imagePath)) {
      console.error(`File doesn't exist at path: ${imagePath}`);
      res.status(400).json({ error: "File upload failed" });
      return;
    }

    try {
      console.log(`Processing image for user ${userId}...`);

      // Upload to Firebase Storage first to get a URL
      const imageUrl = await storageService.uploadFile(imagePath);
      console.log(`Image uploaded to storage, URL: ${imageUrl}`);

      // Process the image URL instead of the local file
      const response = await chatService.processImageFromUrl(userId, imageUrl);
      console.log("Image processed successfully");

      // Send response
      res.json({
        userId,
        message: response,
      });

      // Clean up the temporary file AFTER processing is complete
      try {
        if (fs.existsSync(imagePath)) {
          fs.unlinkSync(imagePath);
          console.log(`Cleaned up temporary file: ${imagePath}`);
        }
      } catch (cleanupError) {
        console.error("Error cleaning up file:", cleanupError);
      }
    } catch (error) {
      console.error("Error processing image:", error);
      res.status(500).json({ error: "Failed to process image" });
    }
  });
}
