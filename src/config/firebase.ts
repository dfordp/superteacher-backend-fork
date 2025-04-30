// src/config/firebase.ts
import admin from "firebase-admin";
import { getStorage } from "firebase-admin/storage";
import { Bucket } from "@google-cloud/storage";
import dotenv from "dotenv";
import path from "path";
import fs from "fs";

dotenv.config();

// Fix for private key format issues
function formatPrivateKey(key: string | undefined): string | undefined {
  if (!key) return undefined;

  // If the key doesn't start with the expected prefix, assume it needs formatting
  if (!key.startsWith("-----BEGIN PRIVATE KEY-----")) {
    // Handle the case where JSON.stringify has added extra escaping
    key = key.replace(/\\n/g, "\n");
  }

  // Remove any extra quotes that might have been added
  key = key.replace(/^"(.*)"$/, "$1");

  return key;
}

// Initialize Firebase Admin if not already initialized
if (!admin.apps.length) {
  try {
    // Try environment variables first (for production/Render deployment)
    if (
      process.env.FIREBASE_PROJECT_ID &&
      process.env.FIREBASE_CLIENT_EMAIL &&
      process.env.FIREBASE_PRIVATE_KEY
    ) {
      console.log("Using Firebase credentials from environment variables");

      // Format the private key to handle common environment variable issues
      const privateKey = formatPrivateKey(process.env.FIREBASE_PRIVATE_KEY);

      admin.initializeApp({
        credential: admin.credential.cert({
          projectId: process.env.FIREBASE_PROJECT_ID,
          clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
          privateKey: privateKey,
        }),
        storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
      });

      console.log(
        "Firebase credentials loaded from environment variables successfully"
      );
    }
    // Fallback to local JSON file (for local development)
    else {
      const serviceAccountPath = path.join(
        __dirname,
        "../../firebase-service-account.json"
      );

      if (fs.existsSync(serviceAccountPath)) {
        console.log(
          "Using Firebase credentials from local service account file"
        );
        const serviceAccount = require(serviceAccountPath);
        admin.initializeApp({
          credential: admin.credential.cert(serviceAccount),
          storageBucket: serviceAccount.project_id + ".appspot.com",
        });

        console.log(
          "Firebase credentials loaded from service account file successfully"
        );
      } else {
        throw new Error(
          "No Firebase credentials found in environment or local file. Cannot initialize without credentials."
        );
      }
    }

    console.log("Firebase Admin initialized successfully");
  } catch (error) {
    console.error("Firebase Admin initialization error:", error);
    console.error(
      "Error details:",
      error instanceof Error ? error.message : "Unknown error"
    );
    throw error; // Re-throw to prevent app from starting without Firebase
  }
}

// Get and export the storage bucket
let storage: Bucket;
try {
  storage = getStorage().bucket();
  console.log("Firebase Storage bucket obtained successfully");
} catch (error) {
  console.error("Error getting Firebase Storage bucket:", error);
  throw error;
}

export { storage };
export default admin;
