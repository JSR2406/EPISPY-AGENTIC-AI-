import React, { useState, useEffect } from 'react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [apiStatus, setApiStatus] = useState({ online: false, text: 'Checking...' });
  const [systemStats, setSystemStats] = useState(null);
  const [formData, setFormData] = useState({
    location_id: 'LOC002', cases: '75', severity_score: '7.8',
    population_density: '1000', weather_temp: '36'
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [activeTab, setActiveTab] = useState('analytics');
  const [heatmapData, setHeatmapData] = useState(null);
  const [outbreakProbability, setOutbreakProbability] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [voiceEnabled, setVoiceEnabled] = useState(false);

  useEffect(() => {
    checkAPIStatus();
    loadSystemStats();
    const statsInterval = setInterval(loadSystemStats, 30000);
    const alertInterval = setInterval(checkForAlerts, 30000);

    return () => {
      clearInterval(statsInterval);
      clearInterval(alertInterval);
    };
  }, []);

  useEffect(() => {
    if (activeTab === 'analytics') {
      loadHeatmapData();
      loadOutbreakProbability();
    }
  }, [activeTab]);

  const checkAPIStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      if (data.overall === 'healthy') {
        setApiStatus({ online: true, text: 'API Online' });
      } else {
        setApiStatus({ online: false, text: 'API Degraded' });
      }
    } catch (error) {
      setApiStatus({ online: false, text: 'API Offline' });
    }
  };
  
  const loadSystemStats = async () => {
     try {
       const [modelInfo, modelHealth, modelMetrics] = await Promise.all([
         fetch(`${API_URL}/model/info`).then(r => r.json()),
         fetch(`${API_URL}/model/health`).then(r => r.json()),
         fetch(`${API_URL}/model/metrics`).then(r => r.json())
       ]);
       setSystemStats({
         accuracy: (modelInfo.performance_metrics.accuracy * 100).toFixed(1),
         predictions: modelHealth.total_predictions,
         responseTime: modelHealth.average_response_time_ms.toFixed(0),
         status: modelHealth.status,
         predictionsByRisk: modelMetrics.predictions_by_risk_level
       });
     } catch (error) { console.error('Stats Error:', error); }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPredictionResult(null);
    try {
      const payload = {
        location_id: formData.location_id,
        cases: parseInt(formData.cases),
        severity_score: parseFloat(formData.severity_score),
        population_density: parseFloat(formData.population_density),
        weather_temp: parseFloat(formData.weather_temp)
      };
      const response = await fetch(`${API_URL}/model/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error((await response.json()).detail);
      const result = await response.json();
      setPredictionResult(result);
      setRecentPredictions(prev => [result, ...prev.slice(0, 4)]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadHeatmapData = async () => {
    try {
      const response = await fetch(`${API_URL}/map/heatmap`);
      const data = await response.json();
      setHeatmapData(data);
    } catch (error) {
      console.error('Heatmap error:', error);
    }
  };

  const loadOutbreakProbability = async () => {
    try {
      const response = await fetch(`${API_URL}/analytics/outbreak-probability`);
      const data = await response.json();
      setOutbreakProbability(data);
    } catch (error) {
      console.error('Outbreak probability error:', error);
    }
  };

  const checkForAlerts = async () => {
    try {
      const response = await fetch(`${API_URL}/alerts/active`);
      const data = await response.json();
      if (data.alerts && data.alerts.length > 0) {
        const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTGH0fPTgjMGHm7A7+OZWR...');
        audio.play();
        setNotifications(prev => [...data.alerts, ...prev].slice(0, 5));
      }
    } catch (error) {
      console.error('Alert check error:', error);
    }
  };

  const startVoiceControl = () => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.onstart = () => setVoiceEnabled(true);
      recognition.onend = () => setVoiceEnabled(false);
      recognition.onresult = (event) => {
        const command = event.results[0][0].transcript.toLowerCase();
        let responseText = `Executing ${command}`;
        if (command.includes('predict')) { setActiveTab('single'); } 
        else if (command.includes('analytics')) { setActiveTab('analytics'); } 
        else if (command.includes('batch')) { setActiveTab('batch'); }
        else if (command.includes('compare')) { setActiveTab('compare'); }
        else { responseText = "Sorry, I didn't recognize that command."; }
        const utterance = new SpeechSynthesisUtterance(responseText);
        window.speechSynthesis.speak(utterance);
      };
      recognition.start();
    } else {
      alert("Voice control is not supported by your browser.");
    }
  };

  const getRiskColor = (level) => ({
    'low': '#10b981', 'medium': '#f59e0b', 'high': '#ef4444', 'critical': '#7f1d1d'
  }[level?.toLowerCase()] || '#6b7280');

  const styles = {
    app: { minHeight: '100vh', backgroundColor: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui, -apple-system, sans-serif' },
    notificationsPanel: { position: 'fixed', top: '20px', right: '20px', zIndex: 1000, width: '350px' },
    notification: { backgroundColor: '#1e293b', padding: '15px', borderRadius: '8px', marginBottom: '10px', display: 'flex', gap: '12px', border: '1px solid #334155' },
    notificationIcon: { fontSize: '24px' },
    notificationContent: { flex: 1 },
    notificationTime: { fontSize: '12px', color: '#94a3b8' },
    header: { backgroundColor: '#1e293b', padding: '20px 40px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #334155' },
    logo: { display: 'flex', gap: '12px', alignItems: 'center' },
    logoIcon: { fontSize: '32px' },
    appSubtitle: { color: '#94a3b8', fontSize: '14px', margin: 0 },
    voiceBtn: { padding: '10px 20px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '14px' },
    voiceBtnListening: { backgroundColor: '#ef4444', animation: 'pulse 1.5s infinite' },
    statusIndicator: { display: 'flex', gap: '8px', alignItems: 'center' },
    statusDot: { width: '10px', height: '10px', borderRadius: '50%' },
    statusOnline: { backgroundColor: '#10b981' },
    statusOffline: { backgroundColor: '#ef4444' },
    tabsContainer: { display: 'flex', gap: '8px', padding: '20px 40px', backgroundColor: '#1e293b', borderBottom: '1px solid #334155' },
    tab: { padding: '12px 24px', backgroundColor: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', borderRadius: '6px', fontSize: '14px', fontWeight: '500', transition: 'all 0.2s' },
    tabActive: { backgroundColor: '#3b82f6', color: 'white' },
    mainContent: { padding: '40px', maxWidth: '1400px', margin: '0 auto' },
    card: { backgroundColor: '#1e293b', borderRadius: '12px', padding: '24px', marginBottom: '24px', border: '1px solid #334155' },
    cardHeader: { marginBottom: '20px' },
    cardTitle: { fontSize: '20px', fontWeight: '600', margin: 0 },
    cardDescription: { color: '#94a3b8', marginTop: '8px' },
    form: { marginBottom: '24px' },
    formGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '20px' },
    formGroup: { display: 'flex', flexDirection: 'column', gap: '6px' },
    label: { fontSize: '14px', color: '#94a3b8', fontWeight: '500' },
    input: { padding: '10px', backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '6px', color: '#e2e8f0', fontSize: '14px' },
    btnPrimary: { padding: '12px 24px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '14px', fontWeight: '600', width: '100%' },
    btnDisabled: { opacity: 0.5, cursor: 'not-allowed' },
    alert: { padding: '12px', borderRadius: '6px', marginBottom: '16px' },
    alertError: { backgroundColor: '#7f1d1d', color: '#fecaca', border: '1px solid #991b1b' },
    resultsSection: { marginTop: '24px' },
    resultHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' },
    resultTimestamp: { fontSize: '12px', color: '#94a3b8' },
    resultMain: { padding: '24px', backgroundColor: '#0f172a', borderRadius: '8px', borderLeft: '4px solid', marginBottom: '20px' },
    riskBadge: { display: 'inline-block', padding: '8px 16px', borderRadius: '20px', fontWeight: '600', textTransform: 'uppercase', fontSize: '14px', marginBottom: '16px' },
    resultMetrics: { display: 'flex', gap: '32px' },
    metric: { display: 'flex', flexDirection: 'column', gap: '4px' },
    metricLabel: { fontSize: '12px', color: '#94a3b8' },
    metricValue: { fontSize: '24px', fontWeight: '700' },
    recentPredictions: { marginTop: '20px' },
    predictionsList: { display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '12px' },
    predictionItem: { display: 'flex', justifyContent: 'space-between', padding: '12px', backgroundColor: '#0f172a', borderRadius: '6px' },
    uploadZone: { border: '2px dashed #334155', borderRadius: '8px', padding: '40px', textAlign: 'center', cursor: 'pointer' },
    heatmapContainer: { position: 'relative' },
    mapGrid: { position: 'relative', width: '100%', height: '500px', backgroundColor: '#0f172a', borderRadius: '8px', overflow: 'hidden' },
    mapMarker: { position: 'absolute', width: '20px', height: '20px', borderRadius: '50%', cursor: 'pointer', transition: 'transform 0.2s', transform: 'translate(-50%, -50%)' },
    markerTooltip: { position: 'absolute', bottom: '100%', left: '50%', transform: 'translateX(-50%)', backgroundColor: '#1e293b', padding: '8px 12px', borderRadius: '6px', whiteSpace: 'nowrap', marginBottom: '8px', fontSize: '12px', border: '1px solid #334155', display: 'none' },
    mapLegend: { display: 'flex', gap: '20px', marginTop: '16px', justifyContent: 'center' },
    legendItem: { display: 'flex', alignItems: 'center', gap: '8px' },
    legendDot: { width: '12px', height: '12px', borderRadius: '50%' },
    probabilityChart: { display: 'flex', alignItems: 'flex-end', gap: '4px', height: '300px', padding: '20px', backgroundColor: '#0f172a', borderRadius: '8px', marginBottom: '20px' },
    probabilityBar: { flex: 1, backgroundColor: '#3b82f6', borderRadius: '4px 4px 0 0', position: 'relative', minHeight: '5%', transition: 'all 0.3s' },
    dayLabel: { position: 'absolute', bottom: '-20px', fontSize: '10px', color: '#94a3b8' },
    probabilityInfo: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' },
    infoItem: { display: 'flex', flexDirection: 'column', gap: '4px', padding: '16px', backgroundColor: '#0f172a', borderRadius: '6px' },
    statsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' },
    statItem: { display: 'flex', flexDirection: 'column', gap: '8px', padding: '20px', backgroundColor: '#0f172a', borderRadius: '8px' },
    statLabel: { fontSize: '14px', color: '#94a3b8' },
    statValue: { fontSize: '28px', fontWeight: '700', color: '#3b82f6' },
    footer: { textAlign: 'center', padding: '20px', color: '#64748b', borderTop: '1px solid #334155', marginTop: '40px' }
  };

  return (
    <div style={styles.app}>
       {notifications.length > 0 && (
         <div style={styles.notificationsPanel}>
           {notifications.map((notif, idx) => (
             <div key={idx} style={styles.notification}>
               <div style={styles.notificationIcon}>🚨</div>
               <div style={styles.notificationContent}>
                 <strong>{notif.title}</strong>
                 <p>{notif.message}</p>
                 <span style={styles.notificationTime}>{new Date(notif.timestamp).toLocaleTimeString()}</span>
               </div>
               <button onClick={() => setNotifications(prev => prev.filter((_, i) => i !== idx))} style={{...styles.btnPrimary, width: 'auto', padding: '4px 8px'}}>✕</button>
             </div>
           ))}
         </div>
       )}

      <header style={styles.header}>
        <div style={styles.logo}>
          <span style={styles.logoIcon}>🔬</span>
          <div><h1 style={{margin: 0}}>EpiSPY</h1><p style={styles.appSubtitle}>AI-Powered Disease Surveillance</p></div>
        </div>
         <button style={{...styles.voiceBtn, ...(voiceEnabled ? styles.voiceBtnListening : {})}} onClick={startVoiceControl}>
           🎤 {voiceEnabled ? 'Listening...' : 'Voice Control'}
         </button>
        <div style={styles.statusIndicator}>
          <span style={{...styles.statusDot, ...(apiStatus.online ? styles.statusOnline : styles.statusOffline)}}></span>
          <span>{apiStatus.text}</span>
        </div>
      </header>

      <div style={styles.tabsContainer}>
        <button style={{...styles.tab, ...(activeTab === 'single' ? styles.tabActive : {})}} onClick={() => setActiveTab('single')}>🔮 Single Prediction</button>
        <button style={{...styles.tab, ...(activeTab === 'batch' ? styles.tabActive : {})}} onClick={() => setActiveTab('batch')}>📊 Batch Analysis</button>
        <button style={{...styles.tab, ...(activeTab === 'analytics' ? styles.tabActive : {})}} onClick={() => setActiveTab('analytics')}>📈 Analytics</button>
        <button style={{...styles.tab, ...(activeTab === 'compare' ? styles.tabActive : {})}} onClick={() => setActiveTab('compare')}>🔍 Compare</button>
      </div>

      <main style={styles.mainContent}>
        {activeTab === 'single' && (
          <section className="card">
             <div className="card-header"><h2 className="card-title">🔮 Risk Prediction</h2></div>
             <form onSubmit={handleSubmit} className="form">
               <div className="form-grid">
                 <div className="form-group">
                   <label>Location ID</label>
                   <input type="text" name="location_id" value={formData.location_id} onChange={handleInputChange} required />
                 </div>
                 <div className="form-group">
                   <label>Cases</label>
                   <input type="number" name="cases" value={formData.cases} onChange={handleInputChange} required />
                 </div>
                 <div className="form-group">
                   <label>Severity Score</label>
                   <input type="number" step="0.1" name="severity_score" value={formData.severity_score} onChange={handleInputChange} required />
                 </div>
                 <div className="form-group">
                   <label>Population Density</label>
                   <input type="number" name="population_density" value={formData.population_density} onChange={handleInputChange} required />
                 </div>
                 <div className="form-group">
                   <label>Weather Temp (°C)</label>
                   <input type="number" step="0.1" name="weather_temp" value={formData.weather_temp} onChange={handleInputChange} required />
                 </div>
               </div>
               <button type="submit" className="btn-primary" disabled={loading}>
                 {loading ? 'Analyzing...' : '🔮 Generate Prediction'}
               </button>
             </form>
             
             {error && <div className="alert alert-error">⚠️ {error}</div>}
             
             {predictionResult && (
               <div className="results-section">
                 <div className="result-header">
                   <h3>Prediction Results</h3>
                   <span className="result-timestamp">{new Date(predictionResult.timestamp).toLocaleString()}</span>
                 </div>
                 <div className="result-main" style={{ borderColor: getRiskColor(predictionResult.risk_level) }}>
                   <div className="risk-badge" style={{ backgroundColor: getRiskColor(predictionResult.risk_level) }}>
                     {predictionResult.risk_level}
                   </div>
                   <div className="result-metrics">
                     <div className="metric">
                       <span className="metric-label">Risk Score</span>
                       <span className="metric-value">{predictionResult.risk_score.toFixed(2)}</span>
                     </div>
                     <div className="metric">
                       <span className="metric-label">Confidence</span>
                       <span className="metric-value">{(predictionResult.confidence * 100).toFixed(1)}%</span>
                     </div>
                   </div>
                 </div>
                 
                 {recentPredictions.length > 1 && (
                   <div className="recent-predictions">
                     <h4>Recent Predictions</h4>
                     <div className="predictions-list">
                       {recentPredictions.slice(1).map((pred, idx) => (
                         <div key={idx} className="prediction-item">
                           <span className="prediction-location">{pred.location_id}</span>
                           <span className="prediction-risk" style={{ color: getRiskColor(pred.risk_level) }}>
                             {pred.risk_level}
                           </span>
                           <span className="prediction-score">{pred.risk_score.toFixed(2)}</span>
                         </div>
                       ))}
                     </div>
                   </div>
                 )}
               </div>
             )}
          </section>
        )}

        {activeTab === 'batch' && (
           <section className="card">
             <div className="card-header"><h2 className="card-title">📊 Batch Analysis</h2></div>
             <p className="card-description">Upload a CSV file with multiple locations for batch prediction</p>
             <div className="upload-zone">
               <input type="file" accept=".csv" onChange={(e) => {
                 const file = e.target.files[0];
                 if (file) setBatchResults({ filename: file.name, status: 'ready' });
               }} />
               <p>Drop CSV file here or click to browse</p>
             </div>
             {batchResults && (
               <div className="batch-results">
                 <p>File ready: {batchResults.filename}</p>
               </div>
             )}
           </section>
        )}

        {activeTab === 'analytics' && (
          <>
            <section className="card">
              <div className="card-header"><h2 className="card-title">🗺️ Geographic Heat Map</h2></div>
              {heatmapData ? (
                <div className="heatmap-container">
                  <div className="map-grid">
                    {heatmapData.locations.map((loc) => (
                      <div key={loc.id} className={`map-marker risk-${loc.risk}`}
                        style={{
                          left: `${((loc.lng - 68) / 28) * 100}%`,
                          top: `${((36 - loc.lat) / 28) * 100}%`
                        }}
                      >
                        <div className="marker-tooltip">
                          <strong>{loc.name}</strong>
                          <div>Cases: {loc.cases}</div>
                          <div>Risk: {loc.risk}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="map-legend">
                    {['low', 'medium', 'high', 'critical'].map(level => (
                       <div key={level} className="legend-item">
                         <span className={`legend-dot ${level}`}></span>
                         <span>{level.charAt(0).toUpperCase() + level.slice(1)} Risk</span>
                       </div>
                    ))}
                  </div>
                </div>
              ) : <p>Loading map data...</p>}
            </section>

            <section className="card">
              <div className="card-header"><h2 className="card-title">📈 30-Day Outbreak Probability</h2></div>
              {outbreakProbability ? (
                <>
                  <div className="probability-chart">
                    {outbreakProbability.timeline.map((point, idx) => (
                      <div key={idx} className="probability-bar"
                        style={{ height: `${point.probability * 100}%` }}
                        title={`Day ${point.day}: ${(point.probability * 100).toFixed(1)}%`}
                      >
                        {idx % 5 === 0 && <span className="day-label">D{point.day}</span>}
                      </div>
                    ))}
                  </div>
                  <div className="probability-info">
                    <div className="info-item"><span>Peak Day</span><strong>Day {outbreakProbability.peak_day.day}</strong></div>
                    <div className="info-item"><span>Peak Probability</span><strong>{(outbreakProbability.peak_day.probability * 100).toFixed(1)}%</strong></div>
                    <div className="info-item"><span>Trend</span><strong className="trend-increasing">{outbreakProbability.current_trend}</strong></div>
                  </div>
                </>
              ) : <p>Loading probability data...</p>}
            </section>

            {systemStats && (
              <section className="card">
                <div className="card-header"><h2 className="card-title">📊 System Statistics</h2></div>
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-label">Model Accuracy</span>
                    <span className="stat-value">{systemStats.accuracy}%</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Total Predictions</span>
                    <span className="stat-value">{systemStats.predictions}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Avg Response Time</span>
                    <span className="stat-value">{systemStats.responseTime}ms</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">System Status</span>
                    <span className="stat-value">{systemStats.status}</span>
                  </div>
                </div>
              </section>
            )}
          </>
        )}

        {activeTab === 'compare' && (
          <section className="card">
            <div className="card-header"><h2 className="card-title">🔍 Location Comparison</h2></div>
            <p className="card-description">This feature is under development. The backend is ready.</p>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>EpiSPY v2.1.0 | <a href={`${API_URL}/docs`} target="_blank" rel="noopener noreferrer">API Docs</a></p>
      </footer>
    </div>
  );
}

export default App;