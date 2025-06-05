"""
Cost Analytics System for RAG Pipeline

This module provides comprehensive cost tracking, analysis, and optimization
for OpenAI and Anthropic API usage, including token counting, cost projections,
and budget monitoring.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import statistics
from enum import Enum



class APIProvider(str, Enum):
    """API providers for cost tracking"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class APIUsage:
    """Single API usage record"""
    timestamp: datetime
    provider: APIProvider
    model: str
    operation: str  # embedding, generation, etc.
    
    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Cost information
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    
    # Request metadata
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    query_type: Optional[str] = None
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens
        if self.total_cost == 0.0:
            self.total_cost = self.input_cost + self.output_cost


@dataclass
class CostProjection:
    """Cost projection for future usage"""
    period_days: int
    projected_requests: int
    projected_cost: float
    confidence_interval: Tuple[float, float]  # (low, high)
    based_on_days: int
    projection_date: datetime


@dataclass
class CostAlert:
    """Cost-related alert"""
    alert_id: str
    timestamp: datetime
    severity: str  # INFO, WARNING, CRITICAL
    alert_type: str  # daily_limit, monthly_budget, spike_detected, etc.
    current_value: float
    threshold: float
    message: str
    provider: APIProvider


class CostCalculator:
    """Calculates costs for different API providers and models"""
    
    # OpenAI pricing (as of 2024) - cost per 1K tokens
    OPENAI_PRICING = {
        'text-embedding-3-small': {
            'input': 0.00002,  # $0.02 per 1M tokens
            'output': 0.0      # No output tokens for embeddings
        },
        'text-embedding-3-large': {
            'input': 0.00013,  # $0.13 per 1M tokens
            'output': 0.0
        },
        'text-embedding-ada-002': {
            'input': 0.0001,   # $0.10 per 1M tokens
            'output': 0.0
        }
    }
    
    # Anthropic pricing (as of 2024) - cost per 1K tokens
    ANTHROPIC_PRICING = {
        'claude-3-5-sonnet-20241022': {
            'input': 0.003,    # $3 per 1M tokens
            'output': 0.015    # $15 per 1M tokens
        },
        'claude-3-5-haiku-20241022': {
            'input': 0.0008,   # $0.80 per 1M tokens
            'output': 0.004    # $4 per 1M tokens
        },
        'claude-3-opus-20240229': {
            'input': 0.015,    # $15 per 1M tokens
            'output': 0.075    # $75 per 1M tokens
        }
    }
    
    @classmethod
    def calculate_openai_cost(
        cls,
        model: str,
        tokens: int,
        operation: str = "embedding"
    ) -> Dict[str, float]:
        """Calculate OpenAI API cost"""
        if model not in cls.OPENAI_PRICING:
            # Default pricing for unknown models
            pricing = {'input': 0.0001, 'output': 0.0}
        else:
            pricing = cls.OPENAI_PRICING[model]
        
        # For embeddings, all tokens are input tokens
        if operation == "embedding":
            input_cost = (tokens / 1000) * pricing['input']
            return {
                'input_tokens': tokens,
                'output_tokens': 0,
                'input_cost': input_cost,
                'output_cost': 0.0,
                'total_cost': input_cost
            }
        else:
            # For other operations, assume input tokens only
            input_cost = (tokens / 1000) * pricing['input']
            return {
                'input_tokens': tokens,
                'output_tokens': 0,
                'input_cost': input_cost,
                'output_cost': 0.0,
                'total_cost': input_cost
            }
    
    @classmethod
    def calculate_anthropic_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """Calculate Anthropic API cost"""
        if model not in cls.ANTHROPIC_PRICING:
            # Default pricing for unknown models
            pricing = {'input': 0.003, 'output': 0.015}
        else:
            pricing = cls.ANTHROPIC_PRICING[model]
        
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost
        }


class CostTracker:
    """Tracks API usage and costs over time"""
    
    def __init__(self, save_path: str = "cost_data"):
        self.usage_records: List[APIUsage] = []
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        # Budget and limit settings
        self.daily_budget: float = 50.0  # $50 per day
        self.monthly_budget: float = 1000.0  # $1000 per month
        self.spike_threshold: float = 5.0  # Alert if cost spikes by 5x
        
        # Alert system
        self.alerts: List[CostAlert] = []
        self.alert_callbacks: List[callable] = []
    
    def record_openai_usage(
        self,
        model: str,
        tokens: int,
        operation: str = "embedding",
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        query_type: Optional[str] = None
    ) -> APIUsage:
        """Record OpenAI API usage"""
        cost_info = CostCalculator.calculate_openai_cost(model, tokens, operation)
        
        usage = APIUsage(
            timestamp=datetime.now(),
            provider=APIProvider.OPENAI,
            model=model,
            operation=operation,
            input_tokens=cost_info['input_tokens'],
            output_tokens=cost_info['output_tokens'],
            total_tokens=cost_info['input_tokens'] + cost_info['output_tokens'],
            input_cost=cost_info['input_cost'],
            output_cost=cost_info['output_cost'],
            total_cost=cost_info['total_cost'],
            request_id=request_id,
            user_id=user_id,
            channel_id=channel_id,
            query_type=query_type
        )
        
        self.usage_records.append(usage)
        self._check_cost_alerts()
        return usage
    
    def record_anthropic_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        query_type: Optional[str] = None
    ) -> APIUsage:
        """Record Anthropic API usage"""
        cost_info = CostCalculator.calculate_anthropic_cost(model, input_tokens, output_tokens)
        
        usage = APIUsage(
            timestamp=datetime.now(),
            provider=APIProvider.ANTHROPIC,
            model=model,
            operation="generation",
            input_tokens=cost_info['input_tokens'],
            output_tokens=cost_info['output_tokens'],
            total_tokens=cost_info['input_tokens'] + cost_info['output_tokens'],
            input_cost=cost_info['input_cost'],
            output_cost=cost_info['output_cost'],
            total_cost=cost_info['total_cost'],
            request_id=request_id,
            user_id=user_id,
            channel_id=channel_id,
            query_type=query_type
        )
        
        self.usage_records.append(usage)
        self._check_cost_alerts()
        return usage
    
    def get_cost_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: Optional[APIProvider] = None
    ) -> Dict[str, Any]:
        """Get cost summary for specified period and provider"""
        
        # Filter records
        filtered_records = self.usage_records
        
        if start_date:
            filtered_records = [r for r in filtered_records if r.timestamp >= start_date]
        
        if end_date:
            filtered_records = [r for r in filtered_records if r.timestamp <= end_date]
        
        if provider:
            filtered_records = [r for r in filtered_records if r.provider == provider]
        
        if not filtered_records:
            return {}
        
        # Calculate aggregated metrics
        total_cost = sum(r.total_cost for r in filtered_records)
        total_tokens = sum(r.total_tokens for r in filtered_records)
        total_requests = len(filtered_records)
        
        # Group by provider
        by_provider = defaultdict(lambda: {'cost': 0.0, 'tokens': 0, 'requests': 0})
        for record in filtered_records:
            by_provider[record.provider.value]['cost'] += record.total_cost
            by_provider[record.provider.value]['tokens'] += record.total_tokens
            by_provider[record.provider.value]['requests'] += 1
        
        # Group by model
        by_model = defaultdict(lambda: {'cost': 0.0, 'tokens': 0, 'requests': 0})
        for record in filtered_records:
            by_model[record.model]['cost'] += record.total_cost
            by_model[record.model]['tokens'] += record.total_tokens
            by_model[record.model]['requests'] += 1
        
        # Group by day
        daily_costs = defaultdict(float)
        for record in filtered_records:
            day_key = record.timestamp.strftime('%Y-%m-%d')
            daily_costs[day_key] += record.total_cost
        
        return {
            'period': {
                'start': min(r.timestamp for r in filtered_records).isoformat(),
                'end': max(r.timestamp for r in filtered_records).isoformat(),
                'days': (max(r.timestamp for r in filtered_records) - 
                        min(r.timestamp for r in filtered_records)).days + 1
            },
            'totals': {
                'cost': total_cost,
                'tokens': total_tokens,
                'requests': total_requests,
                'cost_per_request': total_cost / max(1, total_requests),
                'cost_per_1k_tokens': (total_cost / max(1, total_tokens)) * 1000
            },
            'by_provider': dict(by_provider),
            'by_model': dict(by_model),
            'daily_costs': dict(daily_costs),
            'average_daily_cost': statistics.mean(daily_costs.values()) if daily_costs else 0.0
        }
    
    def get_cost_projection(self, days_ahead: int = 30) -> CostProjection:
        """Project future costs based on historical usage"""
        if not self.usage_records:
            return CostProjection(
                period_days=days_ahead,
                projected_requests=0,
                projected_cost=0.0,
                confidence_interval=(0.0, 0.0),
                based_on_days=0,
                projection_date=datetime.now()
            )
        
        # Use last 7 days for projection
        cutoff_date = datetime.now() - timedelta(days=7)
        recent_records = [r for r in self.usage_records if r.timestamp >= cutoff_date]
        
        if not recent_records:
            # Fall back to all records
            recent_records = self.usage_records
        
        # Calculate daily averages
        daily_costs = defaultdict(float)
        daily_requests = defaultdict(int)
        
        for record in recent_records:
            day_key = record.timestamp.strftime('%Y-%m-%d')
            daily_costs[day_key] += record.total_cost
            daily_requests[day_key] += 1
        
        avg_daily_cost = statistics.mean(daily_costs.values()) if daily_costs else 0.0
        avg_daily_requests = statistics.mean(daily_requests.values()) if daily_requests else 0.0
        
        # Project for future period
        projected_cost = avg_daily_cost * days_ahead
        projected_requests = int(avg_daily_requests * days_ahead)
        
        # Calculate confidence interval (assuming normal distribution)
        if len(daily_costs) > 1:
            std_daily_cost = statistics.stdev(daily_costs.values())
            # 95% confidence interval
            margin = 1.96 * std_daily_cost * (days_ahead ** 0.5)
            confidence_interval = (
                max(0, projected_cost - margin),
                projected_cost + margin
            )
        else:
            # Wide interval if insufficient data
            confidence_interval = (projected_cost * 0.5, projected_cost * 2.0)
        
        return CostProjection(
            period_days=days_ahead,
            projected_requests=projected_requests,
            projected_cost=projected_cost,
            confidence_interval=confidence_interval,
            based_on_days=len(daily_costs),
            projection_date=datetime.now()
        )
    
    def _check_cost_alerts(self):
        """Check for cost-related alerts"""
        now = datetime.now()
        
        # Check daily budget
        daily_cost = self.get_daily_cost(now.date())
        if daily_cost > self.daily_budget:
            alert = CostAlert(
                alert_id=f"daily_budget_{now.strftime('%Y%m%d')}",
                timestamp=now,
                severity='WARNING',
                alert_type='daily_budget',
                current_value=daily_cost,
                threshold=self.daily_budget,
                message=f"Daily cost ${daily_cost:.2f} exceeds budget of ${self.daily_budget:.2f}",
                provider=APIProvider.OPENAI  # Generic for now
            )
            self._trigger_alert(alert)
        
        # Check monthly budget
        monthly_cost = self.get_monthly_cost(now.year, now.month)
        if monthly_cost > self.monthly_budget:
            alert = CostAlert(
                alert_id=f"monthly_budget_{now.strftime('%Y%m')}",
                timestamp=now,
                severity='CRITICAL',
                alert_type='monthly_budget',
                current_value=monthly_cost,
                threshold=self.monthly_budget,
                message=f"Monthly cost ${monthly_cost:.2f} exceeds budget of ${self.monthly_budget:.2f}",
                provider=APIProvider.OPENAI  # Generic for now
            )
            self._trigger_alert(alert)
        
        # Check for cost spikes
        if len(self.usage_records) >= 10:
            recent_avg = statistics.mean([r.total_cost for r in self.usage_records[-10:]])
            if self.usage_records[-1].total_cost > recent_avg * self.spike_threshold:
                alert = CostAlert(
                    alert_id=f"cost_spike_{int(time.time())}",
                    timestamp=now,
                    severity='WARNING',
                    alert_type='cost_spike',
                    current_value=self.usage_records[-1].total_cost,
                    threshold=recent_avg * self.spike_threshold,
                    message=f"Cost spike detected: ${self.usage_records[-1].total_cost:.4f} vs average ${recent_avg:.4f}",
                    provider=self.usage_records[-1].provider
                )
                self._trigger_alert(alert)
    
    def get_daily_cost(self, date) -> float:
        """Get total cost for a specific date"""
        date_str = date.strftime('%Y-%m-%d')
        daily_records = [
            r for r in self.usage_records 
            if r.timestamp.strftime('%Y-%m-%d') == date_str
        ]
        return sum(r.total_cost for r in daily_records)
    
    def get_monthly_cost(self, year: int, month: int) -> float:
        """Get total cost for a specific month"""
        monthly_records = [
            r for r in self.usage_records 
            if r.timestamp.year == year and r.timestamp.month == month
        ]
        return sum(r.total_cost for r in monthly_records)
    
    def _trigger_alert(self, alert: CostAlert):
        """Trigger a cost alert"""
        # Avoid duplicate alerts
        existing_alert_ids = [a.alert_id for a in self.alerts]
        if alert.alert_id not in existing_alert_ids:
            self.alerts.append(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Error in cost alert callback: {e}")
    
    def add_alert_callback(self, callback):
        """Add callback for cost alerts"""
        self.alert_callbacks.append(callback)
    
    def export_usage_data(self, filename: Optional[str] = None) -> str:
        """Export usage data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cost_usage_{timestamp}.json"
        
        filepath = self.save_path / filename
        
        # Prepare data for export
        data = {
            'usage_records': [
                {**asdict(record), 'timestamp': record.timestamp.isoformat()}
                for record in self.usage_records
            ],
            'alerts': [
                {**asdict(alert), 'timestamp': alert.timestamp.isoformat()}
                for alert in self.alerts
            ],
            'settings': {
                'daily_budget': self.daily_budget,
                'monthly_budget': self.monthly_budget,
                'spike_threshold': self.spike_threshold
            },
            'export_metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(self.usage_records),
                'total_alerts': len(self.alerts)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for cost optimization"""
        recommendations = []
        
        if not self.usage_records:
            return recommendations
        
        # Analyze model usage patterns
        model_costs = defaultdict(float)
        model_usage = defaultdict(int)
        
        for record in self.usage_records:
            model_costs[record.model] += record.total_cost
            model_usage[record.model] += 1
        
        # Check for expensive model usage
        total_cost = sum(model_costs.values())
        for model, cost in model_costs.items():
            if cost > total_cost * 0.5:  # More than 50% of total cost
                if model in ['claude-3-opus-20240229', 'text-embedding-3-large']:
                    cheaper_alternative = {
                        'claude-3-opus-20240229': 'claude-3-5-sonnet-20241022',
                        'text-embedding-3-large': 'text-embedding-3-small'
                    }.get(model)
                    
                    if cheaper_alternative:
                        recommendations.append({
                            'type': 'model_optimization',
                            'current_model': model,
                            'suggested_model': cheaper_alternative,
                            'current_cost': cost,
                            'potential_savings': cost * 0.3,  # Estimate 30% savings
                            'description': f"Consider switching from {model} to {cheaper_alternative} for cost savings"
                        })
        
        # Check for usage patterns
        recent_records = self.usage_records[-100:]  # Last 100 requests
        if recent_records:
            avg_tokens_per_request = statistics.mean([r.total_tokens for r in recent_records])
            if avg_tokens_per_request > 2000:
                recommendations.append({
                    'type': 'token_optimization',
                    'current_avg_tokens': avg_tokens_per_request,
                    'description': 'High token usage detected. Consider optimizing prompts or implementing better chunking strategies',
                    'potential_savings': sum(r.total_cost for r in recent_records) * 0.2
                })
        
        # Check for caching opportunities
        if len(self.usage_records) > 50:
            # Simple duplicate detection
            query_hashes = defaultdict(int)
            for record in self.usage_records[-50:]:
                # Simplified query fingerprinting
                if record.request_id:
                    query_hashes[record.request_id] += 1
            
            duplicates = sum(1 for count in query_hashes.values() if count > 1)
            if duplicates > 5:
                recommendations.append({
                    'type': 'caching_opportunity',
                    'duplicate_queries': duplicates,
                    'description': 'Implement caching for repeated queries to reduce API costs',
                    'potential_savings': sum(r.total_cost for r in self.usage_records[-50:]) * 0.15
                })
        
        return recommendations


class CostOptimizer:
    """Provides cost optimization strategies"""
    
    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker
    
    def suggest_model_alternatives(self, current_model: str) -> Dict[str, Any]:
        """Suggest cheaper model alternatives"""
        alternatives = {
            'claude-3-opus-20240229': {
                'alternative': 'claude-3-5-sonnet-20241022',
                'cost_reduction': 0.8,  # 80% cost reduction
                'trade_offs': 'Slightly lower capability for complex reasoning'
            },
            'text-embedding-3-large': {
                'alternative': 'text-embedding-3-small',
                'cost_reduction': 0.85,  # 85% cost reduction
                'trade_offs': 'Lower dimensional embeddings (1536 vs 3072)'
            },
            'claude-3-5-sonnet-20241022': {
                'alternative': 'claude-3-5-haiku-20241022',
                'cost_reduction': 0.73,  # 73% cost reduction
                'trade_offs': 'Faster but potentially less nuanced responses'
            }
        }
        
        return alternatives.get(current_model, {})
    
    def calculate_potential_savings(
        self,
        model_changes: Dict[str, str],
        days_to_project: int = 30
    ) -> Dict[str, float]:
        """Calculate potential savings from model changes"""
        
        # Get recent usage patterns
        cutoff_date = datetime.now() - timedelta(days=7)
        recent_records = [
            r for r in self.cost_tracker.usage_records 
            if r.timestamp >= cutoff_date
        ]
        
        if not recent_records:
            return {}
        
        # Calculate current daily average cost by model
        daily_costs = defaultdict(float)
        for record in recent_records:
            daily_costs[record.model] += record.total_cost
        
        days_in_period = 7
        current_daily_cost = sum(daily_costs.values()) / days_in_period
        
        # Calculate potential savings
        total_savings = 0.0
        for current_model, new_model in model_changes.items():
            if current_model in daily_costs:
                current_cost = daily_costs[current_model] / days_in_period
                
                # Get cost reduction factor
                alternative_info = self.suggest_model_alternatives(current_model)
                cost_reduction = alternative_info.get('cost_reduction', 0.5)
                
                savings_per_day = current_cost * cost_reduction
                total_savings += savings_per_day
        
        projected_savings = total_savings * days_to_project
        
        return {
            'current_daily_cost': current_daily_cost,
            'savings_per_day': total_savings,
            'projected_savings': projected_savings,
            'savings_percentage': (total_savings / max(current_daily_cost, 0.01)) * 100
        }


# Example usage and testing
if __name__ == "__main__":
    # Demo cost tracking
    tracker = CostTracker()
    
    # Add alert callback
    def log_cost_alert(alert: CostAlert):
        print(f"[COST ALERT] {alert.severity}: {alert.message}")
    
    tracker.add_alert_callback(log_cost_alert)
    
    # Simulate some API usage
    for i in range(10):
        # OpenAI embedding
        tracker.record_openai_usage(
            model="text-embedding-3-large",
            tokens=500,
            operation="embedding",
            request_id=f"req_{i}",
            query_type="document_search"
        )
        
        # Anthropic generation
        tracker.record_anthropic_usage(
            model="claude-3-5-sonnet-20241022",
            input_tokens=200,
            output_tokens=100,
            request_id=f"req_{i}",
            query_type="question_answering"
        )
    
    # Get cost summary
    summary = tracker.get_cost_summary()
    print("Cost Summary:", json.dumps(summary, indent=2, default=str))
    
    # Get projections
    projection = tracker.get_cost_projection(30)
    print(f"30-day projection: ${projection.projected_cost:.2f}")
    
    # Get optimization recommendations
    recommendations = tracker.get_optimization_recommendations()
    print("Recommendations:", json.dumps(recommendations, indent=2))
    
    # Export data
    export_file = tracker.export_usage_data()
    print(f"Data exported to: {export_file}")
    
    print("Cost analytics demo completed!")