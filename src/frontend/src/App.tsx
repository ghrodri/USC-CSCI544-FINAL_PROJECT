import React from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  Outlet,
  useNavigate,
} from "react-router-dom";
import Landing from "./pages/Landing.tsx";
import Login from "./pages/Login.tsx";
import Register from "./pages/Register.tsx";
import ChatUI from "./components/ChatUI.tsx";
import { isAuthed, logout, getUserUsername } from "./api/auth.ts";
import "./App.css";

function RequireAuth() {
  return isAuthed() ? <Outlet /> : <Navigate to="/login" replace />;
}

function ChatPage() {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = React.useState(false);
  const username = getUserUsername() || "User";

  return (
    <div>
      <div className="topnav">
        <div className="topnav-left">
          <span className="brand">Mingle AI</span>
        </div>
        <div className="topnav-right">
          <div className="profile">
            <button
              className="avatar-button"
              aria-haspopup="true"
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen((v) => !v)}
              title={username}
            >
              <span className="avatar-letter">
                {(username || "U").charAt(0).toUpperCase()}
              </span>
            </button>
            {menuOpen && (
              <div className="dropdown" role="menu">
                <div className="dropdown-email">{username}</div>
                <button
                  className="dropdown-item"
                  onClick={() => {
                    logout();
                    setMenuOpen(false);
                    navigate("/");
                  }}
                >
                  Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
      <ChatUI />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route element={<RequireAuth />}>
          <Route path="/app" element={<ChatPage />} />
        </Route>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
