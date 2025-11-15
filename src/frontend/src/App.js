import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedPdfs, setUploadedPdfs] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load PDFs and chats on component mount
  useEffect(() => {
    loadPdfs();
    loadChats();
  }, []);

  const loadPdfs = async () => {
    try {
      const response = await axios.get("http://localhost:8000/pdfs");
      setUploadedPdfs(response.data.pdfs);
    } catch (error) {
      console.error("Error loading PDFs:", error);
    }
  };

  const loadChats = async () => {
    try {
      const response = await axios.get("http://localhost:8000/chats");
      setChats(response.data.chats);
    } catch (error) {
      console.error("Error loading chats:", error);
    }
  };

  const loadChatMessages = async (chatId) => {
    try {
      const response = await axios.get(
        `http://localhost:8000/chats/${chatId}/messages`
      );
      setMessages(response.data.messages);
    } catch (error) {
      console.error("Error loading chat messages:", error);
    }
  };

  const createNewChat = async () => {
    try {
      const response = await axios.post("http://localhost:8000/chats", {
        title: "New Chat",
      });
      const newChatId = response.data.chat_id;
      setCurrentChatId(newChatId);
      setMessages([]);
      await loadChats();
    } catch (error) {
      console.error("Error creating chat:", error);
    }
  };

  const selectChat = async (chatId) => {
    setCurrentChatId(chatId);
    await loadChatMessages(chatId);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (file.type !== "application/pdf") {
      alert("Please upload a PDF file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setIsUploading(true);
      const response = await axios.post(
        "http://localhost:8000/upload-pdf",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setIsUploading(false);

      // Reload PDFs list
      await loadPdfs();

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          content: `PDF "${file.name}" uploaded successfully! You can now ask questions about it.`,
          sender: "system",
        },
      ]);
    } catch (error) {
      console.error("Upload error:", error);
      alert("Error uploading PDF. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const deletePdf = async (pdfId) => {
    try {
      await axios.delete(`http://localhost:8000/pdfs/${pdfId}`);
      await loadPdfs();
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          content: "PDF deleted successfully.",
          sender: "system",
        },
      ]);
    } catch (error) {
      console.error("Delete error:", error);
      alert("Error deleting PDF. Please try again.");
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      content: inputMessage,
      sender: "user",
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await axios.post("http://localhost:8000/chat", {
        message: inputMessage,
        chat_id: currentChatId,
      });

      // Update current chat ID if it was created
      if (response.data.chat_id && !currentChatId) {
        setCurrentChatId(response.data.chat_id);
        await loadChats();
      }

      const botMessage = {
        id: Date.now() + 1,
        content: response.data.response,
        sender: "bot",
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage = {
        id: Date.now() + 1,
        content:
          "Sorry, there was an error processing your message. Please try again.",
        sender: "bot",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <div className="header">
        <div className="header-left">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            ☰
          </button>
          <button className="new-chat-button" onClick={createNewChat}>
            + New Chat
          </button>
          <h1>AI Finance Assistant</h1>
        </div>
        <div className="upload-section">
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            disabled={isUploading}
            id="file-upload"
            style={{ display: "none" }}
          />
          <label htmlFor="file-upload" className="upload-button">
            {isUploading ? "Uploading..." : "Upload PDF"}
          </label>
        </div>
      </div>

      <div className="main-content">
        {sidebarOpen && (
          <div className="sidebar">
            <div className="sidebar-header">
              <h3>Chats</h3>
              <button
                className="close-sidebar"
                onClick={() => setSidebarOpen(false)}
              >
                ×
              </button>
            </div>
            <div className="chat-list">
              {chats.length === 0 ? (
                <p className="no-chats">No chats yet</p>
              ) : (
                chats.map((chat) => (
                  <div
                    key={chat.id}
                    className={`chat-item ${
                      currentChatId === chat.id ? "active" : ""
                    }`}
                    onClick={() => selectChat(chat.id)}
                  >
                    <div className="chat-info">
                      <span className="chat-title">{chat.title}</span>
                      <span className="chat-date">
                        {new Date(chat.last_updated).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="sidebar-footer">
              <h4>PDFs</h4>
              <div className="pdf-list">
                {uploadedPdfs.length === 0 ? (
                  <p className="no-pdfs">No PDFs uploaded</p>
                ) : (
                  uploadedPdfs.map((pdf) => (
                    <div key={pdf.id} className="pdf-item">
                      <span className="pdf-name">📄 {pdf.filename}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome-message">
                <p>
                  Welcome! You can start chatting immediately or upload PDF
                  documents for enhanced context.
                </p>
              </div>
            )}
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.sender}`}>
                <div
                  className="message-content"
                  dangerouslySetInnerHTML={{
                    __html: message.content
                      .replace(/\n/g, "<br>")
                      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                      .replace(/\*(.*?)\*/g, "<em>$1</em>"),
                  }}
                ></div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              disabled={isUploading}
              rows="1"
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="send-button"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
