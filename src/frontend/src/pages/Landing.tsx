import React from "react";
import { Link, Navigate } from "react-router-dom";
import { isAuthed } from "../api/auth.ts";

export default function Landing() {
  if (isAuthed()) return <Navigate to="/app" replace />;
  return (
    <div className="landing-background">
      <div
        style={{
          background: "rgba(255, 255, 255, 0.9)",
          backdropFilter: "blur(10px)",
          padding: 40,
          borderRadius: 16,
          maxWidth: 520,
          width: "90%",
          boxShadow: "0 20px 40px rgba(0,0,0,0.2)",
          textAlign: "center",
        }}
      >
        <h1 style={{ margin: "0 0 4px", fontSize: 32, color: "#1e293b" }}>
          Mingle AI
        </h1>
        <p style={{ color: "#475569", marginTop: 12 }}>
          Upload PDFs and chat with your documents using AI.
        </p>
        <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
          <Link
            to="/login"
            style={{
              flex: 1,
              textAlign: "center",
              padding: "14px 20px",
              background: "#0ea5e9",
              color: "white",
              borderRadius: 8,
              textDecoration: "none",
              fontWeight: "bold",
            }}
          >
            Log in
          </Link>
          <Link
            to="/register"
            style={{
              flex: 1,
              textAlign: "center",
              padding: "14px 20px",
              background: "#10b981",
              color: "white",
              borderRadius: 8,
              textDecoration: "none",
              fontWeight: "bold",
            }}
          >
            Register
          </Link>
        </div>
      </div>
    </div>
  );
}
