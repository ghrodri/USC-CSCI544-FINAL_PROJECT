import React, { useState } from "react";
import { Link, useNavigate, Navigate } from "react-router-dom";
import { register, isAuthed } from "../api/auth.ts";

export default function Register() {
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const navigate = useNavigate();
  const [showPopup, setShowPopup] = useState(false);

  // Prevent redirect until popup acknowledged
  if (isAuthed() && !showPopup) return <Navigate to="/app" replace />;

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    try {
      await register(email, username, password);
      setShowPopup(true); // show modal instead of immediate redirect
    } catch (e: any) {
      setErr(e.message || "Registration failed");
    }
  };

  const handleContinue = () => {
    setShowPopup(false);
    navigate("/app");
  };

  return (
    <div
      style={{
        maxWidth: 400,
        margin: "80px auto",
        padding: 24,
        background: "white",
        borderRadius: 12,
        boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
        position: "relative",
      }}
    >
      <h2 style={{ marginTop: 0 }}>Register</h2>
      <form onSubmit={onSubmit}>
        <input
          placeholder="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          style={{
            width: "100%",
            padding: 10,
            marginBottom: 12,
            borderRadius: 6,
            border: "1px solid #ccc",
          }}
        />
        <input
          placeholder="Username"
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          minLength={3}
          style={{
            width: "100%",
            padding: 10,
            marginBottom: 12,
            borderRadius: 6,
            border: "1px solid #ccc",
          }}
        />
        <input
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          style={{
            width: "100%",
            padding: 10,
            marginBottom: 12,
            borderRadius: 6,
            border: "1px solid #ccc",
          }}
        />
        <button
          type="submit"
          style={{
            width: "100%",
            padding: 12,
            background: "#10b981",
            color: "white",
            border: "none",
            borderRadius: 6,
            fontWeight: "bold",
            cursor: "pointer",
          }}
        >
          Create account
        </button>
      </form>
      {err && <p style={{ color: "red", marginTop: 12 }}>{err}</p>}
      <p style={{ marginTop: 16, textAlign: "center" }}>
        Already have an account? <Link to="/login">Login</Link>
      </p>
      {showPopup && (
        <div className="popup-backdrop">
          <div className="popup-container">
            <h3 style={{ marginTop: 0 }}>Account created</h3>
            <p style={{ margin: "8px 0 16px" }}>
              Your user was created successfully.
            </p>
            <button className="popup-button" onClick={handleContinue}>
              Continue
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
