import { storage } from "../config/firebase";
import path from "path";
import fs from "fs";
import { v4 as uuidv4 } from "uuid";

class StorageService {
  /**
   * Upload a file to Firebase Storage and return a public URL
   */
  async uploadFile(filePath: string): Promise<string> {
    try {
      if (!fs.existsSync(filePath)) {
        throw new Error(`File does not exist at path: ${filePath}`);
      }

      // Generate a unique filename to prevent collisions
      const fileName = `${uuidv4()}${path.extname(filePath)}`;
      const destination = `uploads/${fileName}`;

      console.log(`Uploading file to Firebase Storage: ${destination}`);
      console.log(`Storage service type: ${typeof storage}`);
      console.log(
        `Storage service methods: ${Object.keys(storage).join(", ")}`
      );

      // Upload the file to Firebase Storage
      await storage.upload(filePath, {
        destination,
        metadata: {
          contentType: this.getContentType(filePath),
          cacheControl: "public, max-age=31536000",
        },
      });

      // Make the file publicly accessible and get the URL
      const [url] = await storage.file(destination).getSignedUrl({
        action: "read",
        expires: Date.now() + 1000 * 60 * 60 * 24 * 7, // 7 days
      });

      console.log(`File uploaded, public URL: ${url}`);
      return url;
    } catch (error) {
      console.error("Error uploading file to Firebase Storage:", error);
      throw error;
    }
  }

  /**
   * Determine content type based on file extension
   */
  private getContentType(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();

    switch (ext) {
      case ".jpg":
      case ".jpeg":
        return "image/jpeg";
      case ".png":
        return "image/png";
      case ".gif":
        return "image/gif";
      case ".bmp":
        return "image/bmp";
      case ".webp":
        return "image/webp";
      case ".heic":
        return "image/heic";
      default:
        return "application/octet-stream";
    }
  }
}

export const storageService = new StorageService();
