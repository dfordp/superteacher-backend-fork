import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import chatRoutes from "./routes/chatRoutes";
import fs from "fs";
import path from "path";
import os from "os";

// Create Express app
const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Better CORS configuration
app.use(
  cors({
    origin: "*", // For testing, allow all origins
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Accept", "Authorization"],
    credentials: true,
    maxAge: 86400, // Cache preflight requests for 24 hours
  })
);
// Routes
app.use("/api", chatRoutes);

// Enhanced health check endpoint with diagnostics
app.get("/health", (_req: express.Request, res: express.Response) => {
  try {
    // Check if tmp directory is writable
    const tmpDir = os.tmpdir();
    const testFile = path.join(tmpDir, `test-${Date.now()}.txt`);
    fs.writeFileSync(testFile, "test", "utf8");
    fs.unlinkSync(testFile);

    // Check custom upload directory
    const uploadDir =
      process.env.NODE_ENV === "production"
        ? "/tmp"
        : path.join(__dirname, "../../uploads");

    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    const uploadTestFile = path.join(uploadDir, `test-${Date.now()}.txt`);
    fs.writeFileSync(uploadTestFile, "test", "utf8");
    fs.unlinkSync(uploadTestFile);

    res.status(200).json({
      status: "ok",
      environment: process.env.NODE_ENV || "development",
      timestamp: new Date().toISOString(),
      fs_access: {
        tmp_dir: tmpDir,
        tmp_writable: true,
        upload_dir: uploadDir,
        upload_writable: true,
      },
      system: {
        platform: process.platform,
        arch: process.arch,
        node_version: process.version,
      },
      ocr: {
        demo_mode:
          process.env.OCR_DEMO_MODE === "true" ||
          process.env.NODE_ENV === "production",
        endpoint:
          process.env.OCR_ENDPOINT ||
          "https://grading-api.onrender.com/extract-text",
      },
    });
  } catch (error) {
    console.error("Health check error:", error);
    res.status(500).json({
      status: "error",
      message: error instanceof Error ? error.message : "Unknown error",
      timestamp: new Date().toISOString(),
    });
  }
});

// Error handling middleware
app.use(
  (
    err: Error,
    _req: express.Request,
    res: express.Response,
    _next: express.NextFunction
  ) => {
    console.error(err.stack);
    res.status(500).json({
      error: "An unexpected error occurred",
      message: process.env.NODE_ENV === "development" ? err.message : undefined,
    });
  }
);

export default app;
