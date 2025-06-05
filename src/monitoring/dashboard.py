"""
Monitoring Dashboard and Reporting System

This module provides a simple web-based dashboard for viewing monitoring metrics,
generating reports, and managing the monitoring system.
"""

import json
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
import webbrowser
from dataclasses import asdict

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Dashboard will use basic reporting only.")

from .monitoring_integration import MonitoringManager


class MonitoringDashboard:
    """Web-based monitoring dashboard"""
    
    def __init__(
        self,
        monitoring_manager: MonitoringManager,
        host: str = "localhost",
        port: int = 8080
    ):
        self.monitoring_manager = monitoring_manager
        self.host = host
        self.port = port
        
        # Initialize FastAPI app if available
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title="RAG Monitoring Dashboard", version="1.0.0")
            self._setup_routes()
        else:
            self.app = None
    
    def _setup_routes(self):
        """Setup FastAPI routes for the dashboard"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Main dashboard page"""
            return self._generate_dashboard_html()
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/api/monitoring/status")
        async def monitoring_status():
            """Get monitoring system status"""
            return self.monitoring_manager.get_health_status()
        
        @self.app.get("/api/monitoring/report")
        async def monitoring_report():
            """Get comprehensive monitoring report"""
            return self.monitoring_manager.get_comprehensive_report()
        
        @self.app.get("/api/performance/summary")
        async def performance_summary(hours: int = 24):
            """Get performance summary"""
            if self.monitoring_manager.performance_monitor:
                return self.monitoring_manager.performance_monitor.get_performance_summary(hours)
            return {"error": "Performance monitoring not enabled"}
        
        @self.app.get("/api/cost/summary")
        async def cost_summary(days: int = 30):
            """Get cost summary"""
            if self.monitoring_manager.cost_tracker:
                return self.monitoring_manager.cost_tracker.get_cost_summary()
            return {"error": "Cost tracking not enabled"}
        
        @self.app.get("/api/cost/projection")
        async def cost_projection(days_ahead: int = 30):
            """Get cost projection"""
            if self.monitoring_manager.cost_tracker:
                projection = self.monitoring_manager.cost_tracker.get_cost_projection(days_ahead)
                return asdict(projection)
            return {"error": "Cost tracking not enabled"}
        
        @self.app.get("/api/evaluation/metrics")
        async def evaluation_metrics():
            """Get evaluation metrics"""
            if self.monitoring_manager.evaluation_framework:
                metrics = self.monitoring_manager.evaluation_framework.generate_evaluation_report()
                return asdict(metrics)
            return {"error": "Evaluation framework not enabled"}
        
        @self.app.post("/api/monitoring/export")
        async def export_monitoring_data():
            """Export monitoring data"""
            try:
                filepath = self.monitoring_manager.save_monitoring_report()
                return {"success": True, "filepath": filepath}
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML"""
        
        # Get current metrics
        try:
            health_status = self.monitoring_manager.get_health_status()
            report = self.monitoring_manager.get_comprehensive_report()
        except Exception as e:
            health_status = {"error": str(e)}
            report = {"error": str(e)}
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG Monitoring Dashboard</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .card {{
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px 0;
                    border-bottom: 1px solid #eee;
                }}
                .metric:last-child {{
                    border-bottom: none;
                }}
                .metric-label {{
                    font-weight: 500;
                    color: #555;
                }}
                .metric-value {{
                    font-weight: bold;
                    color: #333;
                }}
                .status-healthy {{
                    color: #10b981;
                }}
                .status-error {{
                    color: #ef4444;
                }}
                .status-disabled {{
                    color: #6b7280;
                }}
                .btn {{
                    background: #3b82f6;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    margin: 5px;
                }}
                .btn:hover {{
                    background: #2563eb;
                }}
                .json-container {{
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 15px;
                    margin-top: 10px;
                    max-height: 300px;
                    overflow-y: auto;
                }}
                pre {{
                    margin: 0;
                    font-size: 12px;
                    white-space: pre-wrap;
                }}
                .timestamp {{
                    color: #6b7280;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 RAG Monitoring Dashboard</h1>
                    <p>Real-time monitoring for Retrieval Augmented Generation system</p>
                    <p class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="grid">
                    <!-- System Health -->
                    <div class="card">
                        <h2>🏥 System Health</h2>
                        {self._generate_health_html(health_status)}
                    </div>
                    
                    <!-- Performance Metrics -->
                    <div class="card">
                        <h2>⚡ Performance Metrics</h2>
                        {self._generate_performance_html(report.get('performance', {}))}
                    </div>
                    
                    <!-- Cost Analytics -->
                    <div class="card">
                        <h2>💰 Cost Analytics</h2>
                        {self._generate_cost_html(report.get('cost', {}))}
                    </div>
                    
                    <!-- Evaluation Metrics -->
                    <div class="card">
                        <h2>📊 Evaluation Metrics</h2>
                        {self._generate_evaluation_html(report.get('evaluation', {}))}
                    </div>
                </div>
                
                <!-- Actions -->
                <div class="card">
                    <h2>🔧 Actions</h2>
                    <button class="btn" onclick="refreshDashboard()">🔄 Refresh Dashboard</button>
                    <button class="btn" onclick="exportData()">📥 Export Data</button>
                    <button class="btn" onclick="toggleRawData()">📋 Toggle Raw Data</button>
                </div>
                
                <!-- Raw Data (hidden by default) -->
                <div class="card" id="raw-data" style="display: none;">
                    <h2>📋 Raw Monitoring Data</h2>
                    <div class="json-container">
                        <pre>{json.dumps(report, indent=2, default=str)}</pre>
                    </div>
                </div>
            </div>
            
            <script>
                function refreshDashboard() {{
                    window.location.reload();
                }}
                
                function toggleRawData() {{
                    const rawData = document.getElementById('raw-data');
                    rawData.style.display = rawData.style.display === 'none' ? 'block' : 'none';
                }}
                
                async function exportData() {{
                    try {{
                        const response = await fetch('/api/monitoring/export', {{
                            method: 'POST'
                        }});
                        const result = await response.json();
                        if (result.success) {{
                            alert('Data exported successfully to: ' + result.filepath);
                        }} else {{
                            alert('Export failed: ' + result.error);
                        }}
                    }} catch (error) {{
                        alert('Export failed: ' + error.message);
                    }}
                }}
                
                // Auto-refresh every 30 seconds
                setTimeout(refreshDashboard, 30000);
            </script>
        </body>
        </html>
        """
        return html
    
    def _generate_health_html(self, health_status: Dict[str, str]) -> str:
        """Generate HTML for system health section"""
        html = ""
        for component, status in health_status.items():
            status_class = "status-healthy" if status == "healthy" else ("status-disabled" if status == "disabled" else "status-error")
            html += f"""
            <div class="metric">
                <span class="metric-label">{component.title()}</span>
                <span class="metric-value {status_class}">{status}</span>
            </div>
            """
        return html
    
    def _generate_performance_html(self, performance_data: Dict[str, Any]) -> str:
        """Generate HTML for performance metrics section"""
        if not performance_data or 'error' in performance_data:
            return '<p class="status-disabled">Performance monitoring not available</p>'
        
        html = ""
        metrics = [
            ('avg_cpu_percent', 'Avg CPU %', '%.1f%%'),
            ('avg_memory_percent', 'Avg Memory %', '%.1f%%'),
            ('avg_response_time', 'Avg Response Time', '%.1f ms'),
            ('avg_requests_per_second', 'Requests/sec', '%.2f'),
            ('success_rate', 'Success Rate', '%.1f%%')
        ]
        
        for key, label, format_str in metrics:
            if key in performance_data:
                value = performance_data[key]
                if key == 'success_rate':
                    value *= 100  # Convert to percentage
                formatted_value = format_str % value
                html += f"""
                <div class="metric">
                    <span class="metric-label">{label}</span>
                    <span class="metric-value">{formatted_value}</span>
                </div>
                """
        
        return html
    
    def _generate_cost_html(self, cost_data: Dict[str, Any]) -> str:
        """Generate HTML for cost analytics section"""
        if not cost_data or 'error' in cost_data:
            return '<p class="status-disabled">Cost tracking not available</p>'
        
        html = ""
        
        # Current summary
        if 'current_summary' in cost_data:
            summary = cost_data['current_summary']
            if 'totals' in summary:
                totals = summary['totals']
                html += f"""
                <div class="metric">
                    <span class="metric-label">Total Cost</span>
                    <span class="metric-value">${totals.get('cost', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cost per Request</span>
                    <span class="metric-value">${totals.get('cost_per_request', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Requests</span>
                    <span class="metric-value">{totals.get('requests', 0)}</span>
                </div>
                """
        
        # 30-day projection
        if 'projection_30_days' in cost_data:
            projection = cost_data['projection_30_days']
            html += f"""
            <div class="metric">
                <span class="metric-label">30-day Projection</span>
                <span class="metric-value">${projection.get('projected_cost', 0):.2f}</span>
            </div>
            """
        
        return html
    
    def _generate_evaluation_html(self, evaluation_data: Dict[str, Any]) -> str:
        """Generate HTML for evaluation metrics section"""
        if not evaluation_data or 'error' in evaluation_data:
            return '<p class="status-disabled">Evaluation metrics not available</p>'
        
        html = ""
        
        # Quality metrics
        if 'quality_metrics' in evaluation_data:
            quality = evaluation_data['quality_metrics']
            html += f"""
            <div class="metric">
                <span class="metric-label">Response Quality</span>
                <span class="metric-value">{quality.get('response_quality_score', 0):.2f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Factual Accuracy</span>
                <span class="metric-value">{quality.get('factual_accuracy', 0):.2f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Completeness</span>
                <span class="metric-value">{quality.get('completeness', 0):.2f}</span>
            </div>
            """
        
        # Retrieval metrics
        if 'retrieval_metrics' in evaluation_data:
            retrieval = evaluation_data['retrieval_metrics']
            if 'precision_at_k' in retrieval:
                precision_5 = retrieval['precision_at_k'].get(5, 0)
                html += f"""
                <div class="metric">
                    <span class="metric-label">Precision@5</span>
                    <span class="metric-value">{precision_5:.2f}</span>
                </div>
                """
            
            if 'mrr' in retrieval:
                html += f"""
                <div class="metric">
                    <span class="metric-label">Mean Reciprocal Rank</span>
                    <span class="metric-value">{retrieval['mrr']:.2f}</span>
                </div>
                """
        
        # Total queries
        if 'total_queries' in evaluation_data:
            html += f"""
            <div class="metric">
                <span class="metric-label">Total Queries</span>
                <span class="metric-value">{evaluation_data['total_queries']}</span>
            </div>
            """
        
        return html
    
    def start_server(self, open_browser: bool = True):
        """Start the dashboard server"""
        if not FASTAPI_AVAILABLE:
            print("FastAPI not available. Cannot start web dashboard.")
            return
        
        print(f"Starting monitoring dashboard at http://{self.host}:{self.port}")
        
        if open_browser:
            # Open browser after a short delay
            import threading
            def open_browser_delayed():
                import time
                time.sleep(2)
                webbrowser.open(f"http://{self.host}:{self.port}")
            
            threading.Thread(target=open_browser_delayed, daemon=True).start()
        
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")


class ReportGenerator:
    """Generate various types of monitoring reports"""
    
    def __init__(self, monitoring_manager: MonitoringManager):
        self.monitoring_manager = monitoring_manager
    
    def generate_text_report(self) -> str:
        """Generate a text-based monitoring report"""
        report = self.monitoring_manager.get_comprehensive_report()
        
        text_report = f"""
RAG SYSTEM MONITORING REPORT
============================
Generated: {report.get('timestamp', 'Unknown')}

SYSTEM HEALTH
-------------
"""
        
        # Health status
        health_status = self.monitoring_manager.get_health_status()
        for component, status in health_status.items():
            text_report += f"{component.title()}: {status}\n"
        
        # Performance metrics
        if 'performance' in report and report['performance']:
            perf = report['performance']
            text_report += f"""
PERFORMANCE METRICS (24 hours)
------------------------------
Average CPU Usage: {perf.get('avg_cpu_percent', 0):.1f}%
Average Memory Usage: {perf.get('avg_memory_percent', 0):.1f}%
Average Response Time: {perf.get('avg_response_time', 0):.1f} ms
Requests per Second: {perf.get('avg_requests_per_second', 0):.2f}
Success Rate: {perf.get('success_rate', 0)*100:.1f}%
Total Successful Requests: {perf.get('total_successful_requests', 0)}
Total Failed Requests: {perf.get('total_failed_requests', 0)}
"""
        
        # Cost analytics
        if 'cost' in report and report['cost']:
            cost = report['cost']
            if 'current_summary' in cost and 'totals' in cost['current_summary']:
                totals = cost['current_summary']['totals']
                text_report += f"""
COST ANALYTICS
--------------
Total Cost: ${totals.get('cost', 0):.4f}
Cost per Request: ${totals.get('cost_per_request', 0):.4f}
Total Requests: {totals.get('requests', 0)}
Total Tokens: {totals.get('tokens', 0):,}
"""
                
                if 'projection_30_days' in cost:
                    proj = cost['projection_30_days']
                    text_report += f"30-day Projection: ${proj.get('projected_cost', 0):.2f}\n"
        
        # Evaluation metrics
        if 'evaluation' in report and report['evaluation']:
            eval_data = report['evaluation']
            text_report += f"""
EVALUATION METRICS
------------------
Total Queries Evaluated: {eval_data.get('total_queries', 0)}
"""
            
            if 'quality_metrics' in eval_data:
                quality = eval_data['quality_metrics']
                text_report += f"""Response Quality Score: {quality.get('response_quality_score', 0):.2f}
Factual Accuracy: {quality.get('factual_accuracy', 0):.2f}
Completeness: {quality.get('completeness', 0):.2f}
Coherence: {quality.get('coherence', 0):.2f}
Source Attribution: {quality.get('source_attribution_accuracy', 0):.2f}
"""
            
            if 'retrieval_metrics' in eval_data:
                retrieval = eval_data['retrieval_metrics']
                text_report += f"""Mean Reciprocal Rank: {retrieval.get('mrr', 0):.2f}
"""
                if 'precision_at_k' in retrieval:
                    text_report += "Precision@K: " + ", ".join([
                        f"@{k}: {v:.2f}" for k, v in retrieval['precision_at_k'].items()
                    ]) + "\n"
        
        text_report += "\n" + "="*50 + "\n"
        return text_report
    
    def generate_csv_report(self) -> str:
        """Generate a CSV-formatted report"""
        # This would generate CSV data for various metrics
        # For now, return a simple implementation
        report = self.monitoring_manager.get_comprehensive_report()
        
        csv_lines = [
            "metric_type,metric_name,value,timestamp"
        ]
        
        timestamp = report.get('timestamp', datetime.now().isoformat())
        
        # Add performance metrics
        if 'current_performance' in report:
            perf = report['current_performance']
            for metric, value in perf.items():
                csv_lines.append(f"performance,{metric},{value},{timestamp}")
        
        # Add cost metrics
        if 'cost' in report and 'current_summary' in report['cost']:
            cost_summary = report['cost']['current_summary']
            if 'totals' in cost_summary:
                totals = cost_summary['totals']
                for metric, value in totals.items():
                    csv_lines.append(f"cost,{metric},{value},{timestamp}")
        
        return "\n".join(csv_lines)
    
    def save_report(self, report_type: str = "text", filename: Optional[str] = None) -> str:
        """Save a monitoring report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "txt" if report_type == "text" else "csv"
            filename = f"monitoring_report_{timestamp}.{extension}"
        
        if report_type == "text":
            content = self.generate_text_report()
        elif report_type == "csv":
            content = self.generate_csv_report()
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
        
        # Save to monitoring data directory
        save_path = Path(self.monitoring_manager.save_path) / filename
        
        with open(save_path, 'w') as f:
            f.write(content)
        
        print(f"Report saved to: {save_path}")
        return str(save_path)


# Example usage and CLI interface
if __name__ == "__main__":
    # Demo dashboard
    from .monitoring_integration import MonitoringManager
    
    # Create monitoring manager
    monitoring_manager = MonitoringManager()
    
    # Generate some sample data for demo
    if monitoring_manager.cost_tracker:
        monitoring_manager.cost_tracker.record_openai_usage("text-embedding-3-large", 500)
        monitoring_manager.cost_tracker.record_anthropic_usage("claude-3-5-sonnet-20241022", 200, 100)
    
    # Generate text report
    report_generator = ReportGenerator(monitoring_manager)
    text_report = report_generator.generate_text_report()
    print("TEXT REPORT:")
    print(text_report)
    
    # Save reports
    text_file = report_generator.save_report("text")
    csv_file = report_generator.save_report("csv")
    
    print(f"Reports saved:")
    print(f"  Text: {text_file}")
    print(f"  CSV: {csv_file}")
    
    # Start dashboard if FastAPI is available
    if FASTAPI_AVAILABLE:
        print("\nStarting web dashboard...")
        dashboard = MonitoringDashboard(monitoring_manager, port=8080)
        try:
            dashboard.start_server(open_browser=False)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
    else:
        print("\nWeb dashboard not available (FastAPI not installed)")
    
    # Cleanup
    monitoring_manager.cleanup()
    print("Monitoring dashboard demo completed!")