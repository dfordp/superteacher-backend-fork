// src/server.ts - Updated for CBSE Grading Assistant
import app from "./app";
import dotenv from "dotenv";
import { CbseChatService } from "./services/cbseChatService";

// Load environment variables
dotenv.config();

// Set port

// Register the CBSE chat service
console.log("Initializing CBSE Grading Assistant...");

// Create an instance of the CBSE chat service
export const cbseChatService = new CbseChatService();
(global as any).chatService = cbseChatService;
console.log("CBSE Grading Assistant initialized successfully");

// Start the server
app.listen(3000, () => {
  console.log(`CBSE Grading Assistant server is running on port ${3000}`);
  console.log(`Health check: http://localhost:${3000}/health`);
  console.log(`CBSE Grading API: http://localhost:${3000}/api/chat`);
  console.log(`Environment: ${process.env.NODE_ENV || "development"}`);
});

// Add graceful shutdown
process.on("SIGTERM", () => {
  console.log("SIGTERM signal received: closing server");
  process.exit(0);
});

process.on("SIGINT", () => {
  console.log("SIGINT signal received: closing server");
  process.exit(0);
});
