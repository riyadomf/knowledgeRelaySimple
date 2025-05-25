import React, { useState, useEffect, useRef } from 'react';
import { Upload, MessageCircle, Users, FileText, Brain, Send, Plus, ChevronDown, ExternalLink } from 'lucide-react';

const KnowledgeRelay = () => {
  const [currentView, setCurrentView] = useState('dashboard');
  const [teams, setTeams] = useState([]);
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [qaSession, setQaSession] = useState({
    responses: {},
    currentQuestionIndex: 0,
    adaptiveQuestions: []
  });
  const [predefinedQuestions, setPredefinedQuestions] = useState([]);
  const messagesEndRef = useRef(null);

  // API Base URL
  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    loadTeams();
    loadPredefinedQuestions();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadTeams = async () => {
    try {
      const response = await fetch(`${API_BASE}/teams`);
      const data = await response.json();
      setTeams(data);
    } catch (error) {
      console.error('Error loading teams:', error);
    }
  };

  const loadPredefinedQuestions = async () => {
    try {
      const response = await fetch(`${API_BASE}/predefined-questions`);
      const data = await response.json();
      setPredefinedQuestions(data.questions);
    } catch (error) {
      console.error('Error loading questions:', error);
    }
  };

  const createTeam = async (teamName) => {
    try {
      const response = await fetch(`${API_BASE}/teams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: teamName })
      });
      const newTeam = await response.json();
      setTeams([...teams, newTeam]);
      return newTeam;
    } catch (error) {
      console.error('Error creating team:', error);
    }
  };

  const uploadDocument = async (file, teamId, developerName) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('team_id', teamId);
    formData.append('developer_name', developerName);

    try {
      setUploadProgress(0);
      const response = await fetch(`${API_BASE}/upload-document`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        setUploadProgress(100);
        setTimeout(() => setUploadProgress(null), 2000);
        return await response.json();
      }
    } catch (error) {
      console.error('Error uploading document:', error);
      setUploadProgress(null);
    }
  };

  const askQuestion = async (question, teamId) => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ team_id: teamId, question })
      });
      const data = await response.json();
      setIsLoading(false);
      return data;
    } catch (error) {
      console.error('Error asking question:', error);
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !selectedTeam) return;

    const userMessage = { type: 'user', content: inputMessage, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    
    const question = inputMessage;
    setInputMessage('');

    const response = await askQuestion(question, selectedTeam.id);
    
    if (response) {
      const botMessage = {
        type: 'bot',
        content: response.answer,
        sources: response.sources,
        confidence: response.confidence,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    }
  };

  const getAdaptiveQuestions = async (teamId, responses) => {
    try {
      const response = await fetch(`${API_BASE}/adaptive-questions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ team_id: teamId, previous_responses: responses })
      });
      const data = await response.json();
      return data.adaptive_questions || [];
    } catch (error) {
      console.error('Error getting adaptive questions:', error);
      return [];
    }
  };

  const saveQASession = async (teamId, developerName, responses) => {
    try {
      const response = await fetch(`${API_BASE}/qa-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          team_id: teamId,
          developer_name: developerName,
          responses
        })
      });
      return await response.json();
    } catch (error) {
      console.error('Error saving QA session:', error);
    }
  };

  // Components
  const Dashboard = () => (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">KnowledgeRelay</h1>
        <p className="text-gray-600">AI-powered knowledge transfer for development teams</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center mb-4">
            <Upload className="h-8 w-8 text-blue-600 mr-3" />
            <h2 className="text-xl font-semibold">For Outgoing Developers</h2>
          </div>
          <p className="text-gray-600 mb-4">Share your project knowledge through documents and guided Q&A sessions.</p>
          <button
            onClick={() => setCurrentView('upload')}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Start Knowledge Transfer
          </button>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center mb-4">
            <MessageCircle className="h-8 w-8 text-green-600 mr-3" />
            <h2 className="text-xl font-semibold">For Incoming Developers</h2>
          </div>
          <p className="text-gray-600 mb-4">Ask questions and get instant answers from your team's knowledge base.</p>
          <button
            onClick={() => setCurrentView('chat')}
            className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
          >
            Ask Questions
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <Users className="h-6 w-6 text-gray-600 mr-2" />
            <h3 className="text-lg font-semibold">Teams</h3>
          </div>
          <CreateTeamButton />
        </div>
        
        <div className="grid md:grid-cols-3 gap-4">
          {teams.map(team => (
            <div key={team.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
              <h4 className="font-semibold mb-2">{team.name}</h4>
              <p className="text-sm text-gray-600 mb-3">
                Created: {new Date(team.created_at).toLocaleDateString()}
              </p>
              <button
                onClick={() => setSelectedTeam(team)}
                className="text-blue-600 hover:text-blue-800 text-sm font-medium"
              >
                Select Team →
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const CreateTeamButton = () => {
    const [showForm, setShowForm] = useState(false);
    const [teamName, setTeamName] = useState('');

    const handleSubmit = async (e) => {
      e.preventDefault();
      if (teamName.trim()) {
        await createTeam(teamName);
        setTeamName('');
        setShowForm(false);
      }
    };

    return (
      <div>
        {!showForm ? (
          <button
            onClick={() => setShowForm(true)}
            className="flex items-center text-blue-600 hover:text-blue-800"
          >
            <Plus className="h-4 w-4 mr-1" />
            New Team
          </button>
        ) : (
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={teamName}
              onChange={(e) => setTeamName(e.target.value)}
              placeholder="Team name"
              className="px-3 py-1 border rounded text-sm"
              autoFocus
            />
            <button type="submit" className="px-3 py-1 bg-blue-600 text-white rounded text-sm">
              Create
            </button>
            <button
              type="button"
              onClick={() => setShowForm(false)}
              className="px-3 py-1 text-gray-600 text-sm"
            >
              Cancel
            </button>
          </form>
        )}
      </div>
    );
  };

  const UploadView = () => {
    const [developerName, setDeveloperName] = useState('');
    const [dragActive, setDragActive] = useState(false);

    const handleDrop = (e) => {
      e.preventDefault();
      setDragActive(false);
      const files = e.dataTransfer.files;
      if (files[0] && selectedTeam && developerName) {
        uploadDocument(files[0], selectedTeam.id, developerName);
      }
    };

    const handleFileSelect = (e) => {
      const file = e.target.files[0];
      if (file && selectedTeam && developerName) {
        uploadDocument(file, selectedTeam.id, developerName);
      }
    };

    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="mb-6">
          <button
            onClick={() => setCurrentView('dashboard')}
            className="text-blue-600 hover:text-blue-800 mb-4"
          >
            ← Back to Dashboard
          </button>
          <h2 className="text-2xl font-bold mb-2">Knowledge Transfer</h2>
          <p className="text-gray-600">Upload documents and share your project knowledge</p>
        </div>

        {!selectedTeam ? (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <p className="text-yellow-800">Please select a team first from the dashboard.</p>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <h3 className="text-lg font-semibold mb-4">Selected Team: {selectedTeam.name}</h3>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Your Name
                </label>
                <input
                  type="text"
                  value={developerName}
                  onChange={(e) => setDeveloperName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter your name"
                />
              </div>

              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center ${
                  dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
                }`}
                onDragOver={(e) => e.preventDefault()}
                onDragEnter={() => setDragActive(true)}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
              >
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg mb-2">Drop files here or click to upload</p>
                <p className="text-sm text-gray-600 mb-4">Supports PDF, Word documents, and text files</p>
                <input
                  type="file"
                  onChange={handleFileSelect}
                  accept=".pdf,.docx,.doc,.txt"
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="bg-blue-600 text-white px-4 py-2 rounded-lg cursor-pointer hover:bg-blue-700 transition-colors"
                >
                  Choose Files
                </label>
              </div>

              {uploadProgress !== null && (
                <div className="mt-4">
                  <div className="bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">Upload progress: {uploadProgress}%</p>
                </div>
              )}
            </div>

            <div className="bg-white rounded-lg shadow-sm border p-6">
              <h3 className="text-lg font-semibold mb-4">Q&A Knowledge Capture</h3>
              <p className="text-gray-600 mb-4">Answer guided questions to capture your project knowledge</p>
              <button
                onClick={() => setCurrentView('qa-session')}
                disabled={!developerName}
                className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Start Q&A Session
              </button>
            </div>
          </div>
        )}
      </div>
    );
  };

  const QASessionView = () => {
    const [developerName, setDeveloperName] = useState('');
    const [currentQuestions, setCurrentQuestions] = useState([]);
    const [currentAnswers, setCurrentAnswers] = useState({});
    const [sessionPhase, setSessionPhase] = useState('setup'); // setup, predefined, adaptive, complete

    const startQASession = () => {
      setCurrentQuestions(predefinedQuestions);
      setSessionPhase('predefined');
    };

    const handleAnswerChange = (question, answer) => {
      setCurrentAnswers(prev => ({
        ...prev,
        [question]: answer
      }));
    };

    const proceedToAdaptive = async () => {
      if (!selectedTeam) return;
      
      setIsLoading(true);
      const adaptiveQuestions = await getAdaptiveQuestions(selectedTeam.id, currentAnswers);
      setCurrentQuestions(adaptiveQuestions);
      setSessionPhase('adaptive');
      setIsLoading(false);
    };

    const completeSession = async () => {
      if (!selectedTeam || !developerName) return;
      
      setIsLoading(true);
      await saveQASession(selectedTeam.id, developerName, currentAnswers);
      setSessionPhase('complete');
      setIsLoading(false);
    };

    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="mb-6">
          <button
            onClick={() => setCurrentView('upload')}
            className="text-blue-600 hover:text-blue-800 mb-4"
          >
            ← Back to Upload
          </button>
          <h2 className="text-2xl font-bold mb-2">Q&A Knowledge Capture</h2>
          <p className="text-gray-600">Share your expertise through guided questions</p>
        </div>

        {sessionPhase === 'setup' && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold mb-4">Setup Q&A Session</h3>
            {selectedTeam && (
              <p className="text-gray-600 mb-4">Team: {selectedTeam.name}</p>
            )}
            
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Your Name
              </label>
              <input
                type="text"
                value={developerName}
                onChange={(e) => setDeveloperName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your name"
              />
            </div>

            <button
              onClick={startQASession}
              disabled={!developerName || !selectedTeam}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Start Q&A Session
            </button>
          </div>
        )}

        {sessionPhase === 'predefined' && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold mb-4">Standard Questions</h3>
            <p className="text-gray-600 mb-6">Please answer these questions about your project:</p>
            
            <div className="space-y-6">
              {currentQuestions.map((question, index) => (
                <div key={index} className="border-b pb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {question}
                  </label>
                  <textarea
                    value={currentAnswers[question] || ''}
                    onChange={(e) => handleAnswerChange(question, e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows={3}
                    placeholder="Your answer..."
                  />
                </div>
              ))}
            </div>

            <div className="flex gap-4 mt-6">
              <button
                onClick={proceedToAdaptive}
                disabled={isLoading}
                className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
              >
                {isLoading ? 'Processing...' : 'Generate Follow-up Questions'}
              </button>
              <button
                onClick={completeSession}
                disabled={isLoading}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                {isLoading ? 'Saving...' : 'Complete Session'}
              </button>
            </div>
          </div>
        )}

        {sessionPhase === 'adaptive' && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold mb-4">Follow-up Questions</h3>
            <p className="text-gray-600 mb-6">Based on your answers, here are some additional questions:</p>
            
            <div className="space-y-6">
              {currentQuestions.map((question, index) => (
                <div key={index} className="border-b pb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {question}
                  </label>
                  <textarea
                    value={currentAnswers[question] || ''}
                    onChange={(e) => handleAnswerChange(question, e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows={3}
                    placeholder="Your answer..."
                  />
                </div>
              ))}
            </div>

            <button
              onClick={completeSession}
              disabled={isLoading}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 mt-6"
            >
              {isLoading ? 'Saving...' : 'Complete Session'}
            </button>
          </div>
        )}

        {sessionPhase === 'complete' && (
          <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
            <div className="text-green-600 mb-4">
              <Brain className="h-12 w-12 mx-auto mb-2" />
              <h3 className="text-lg font-semibold">Session Complete!</h3>
            </div>
            <p className="text-gray-600 mb-6">Your knowledge has been captured and added to the team's knowledge base.</p>
            <button
              onClick={() => setCurrentView('dashboard')}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Back to Dashboard
            </button>
          </div>
        )}
      </div>
    );
  };

  const ChatView = () => {
    const handleKeyPress = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    };

    return (
      <div className="max-w-4xl mx-auto p-6 h-screen flex flex-col">
        <div className="mb-6">
          <button
            onClick={() => setCurrentView('dashboard')}
            className="text-blue-600 hover:text-blue-800 mb-4"
          >
            ← Back to Dashboard
          </button>
          <h2 className="text-2xl font-bold mb-2">Ask Questions</h2>
          {selectedTeam ? (
            <p className="text-gray-600">Team: {selectedTeam.name}</p>
          ) : (
            <p className="text-red-600">Please select a team first</p>
          )}
        </div>

        <div className="flex-1 bg-white rounded-lg shadow-sm border flex flex-col">
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 mt-8">
                <MessageCircle className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>Start a conversation by asking a question about your project.</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-3xl rounded-lg p-4 ${
                      message.type === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-900'
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <p className="font-semibold text-sm mb-2">Sources:</p>
                        <div className="space-y-2">
                          {message.sources.map((source, idx) => (
                            <div key={idx} className="bg-white rounded p-2 text-sm">
                              <p className="text-gray-700">{source.content}</p>
                              {source.metadata.source_file && (
                                <p className="text-gray-500 text-xs mt-1">
                                  From: {source.metadata.source_file}
                                </p>
                              )}
                              <div className="flex items-center mt-1">
                                <span className="text-xs text-gray-500">
                                  Confidence: {Math.round(source.similarity * 100)}%
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    <div className="text-xs opacity-75 mt-2">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))
            )}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-lg p-4">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <span className="text-gray-600">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <div className="border-t p-4">
            <div className="flex space-x-2">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={selectedTeam ? "Ask a question about your project..." : "Please select a team first"}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                rows={1}
                disabled={!selectedTeam}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || !selectedTeam || isLoading}
                className="bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const Navigation = () => (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-6xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Brain className="h-8 w-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-900">KnowledgeRelay</span>
          </div>
          
          <div className="flex items-center space-x-4">
            {selectedTeam && (
              <div className="flex items-center space-x-2">
                <Users className="h-4 w-4 text-gray-600" />
                <span className="text-sm text-gray-600">{selectedTeam.name}</span>
              </div>
            )}
            
            <div className="flex space-x-2">
              <button
                onClick={() => setCurrentView('dashboard')}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  currentView === 'dashboard'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setCurrentView('upload')}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  currentView === 'upload' || currentView === 'qa-session'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Upload
              </button>
              <button
                onClick={() => setCurrentView('chat')}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  currentView === 'chat'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Chat
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );

  const renderCurrentView = () => {
    switch (currentView) {
      case 'upload':
        return <UploadView />;
      case 'qa-session':
        return <QASessionView />;
      case 'chat':
        return <ChatView />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      {renderCurrentView()}
    </div>
  );
};

export default KnowledgeRelay;