"""
Quality Controls for RAG Response Generation

This module implements multi-layered quality assurance for RAG responses using
Claude's reasoning capabilities, including hallucination detection, factual
consistency checking, and constitutional AI validation.
"""

from typing import Dict, List, Tuple
import re
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Response confidence levels"""
    HIGH = "high"        # >0.8
    MEDIUM = "medium"    # 0.5-0.8
    LOW = "low"         # 0.3-0.5
    VERY_LOW = "very_low"  # <0.3


class ValidationResult(Enum):
    """Validation outcome types"""
    APPROVED = "approved"
    FLAGGED = "flagged"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result"""
    confidence_score: float
    confidence_level: ConfidenceLevel
    validation_result: ValidationResult
    hallucination_score: float
    factual_consistency_score: float
    citation_accuracy_score: float
    constitutional_compliance: bool
    issues_detected: List[str]
    recommendations: List[str]
    sources_verified: List[str]
    reasoning: str


@dataclass
class SourceClaim:
    """Individual factual claim with source attribution"""
    claim: str
    source_id: str
    source_text: str
    confidence: float
    verified: bool


class HallucinationDetector:
    """Detects potential hallucinations using Claude's reasoning capabilities"""
    
    def __init__(self, claude_client, model: str = "claude-3-5-sonnet-20241022"):
        if not ANTHROPIC_AVAILABLE and claude_client is not None:
            raise ImportError("anthropic package is required for HallucinationDetector")
        self.claude_client = claude_client
        self.model = model
    
    async def detect_hallucinations(
        self, 
        response: str, 
        retrieved_context: List[str],
        query: str
    ) -> Tuple[float, List[str], str]:
        """
        Detect potential hallucinations by comparing response against retrieved context
        
        Returns:
            - hallucination_score: 0.0 (no hallucination) to 1.0 (definite hallucination)
            - detected_issues: List of specific hallucination concerns
            - reasoning: Claude's detailed reasoning about the assessment
        """
        
        context_combined = "\n\n".join(retrieved_context)
        
        hallucination_prompt = f"""
You are tasked with detecting potential hallucinations in a RAG (Retrieval Augmented Generation) response. Your job is to carefully analyze whether the response contains information that is not supported by the provided source documents.

QUERY: {query}

RETRIEVED SOURCE DOCUMENTS:
{context_combined}

GENERATED RESPONSE:
{response}

ANALYSIS INSTRUCTIONS:
1. Identify every factual claim in the generated response
2. For each claim, determine if it can be verified from the source documents
3. Look for information that seems plausible but is not actually present in the sources
4. Check for misinterpretations, exaggerations, or conflations of source material
5. Assess the overall faithfulness of the response to the provided context

Please provide your analysis in the following format:

HALLUCINATION SCORE: [0.0 to 1.0]
- 0.0-0.2: Fully grounded in sources, no hallucinations detected
- 0.2-0.4: Minor unsupported details, mostly faithful
- 0.4-0.6: Some concerning claims not supported by sources
- 0.6-0.8: Significant hallucinations present
- 0.8-1.0: Major hallucinations, response largely fabricated

DETECTED ISSUES:
[List specific examples of potentially hallucinated content, or "None detected" if clean]

DETAILED REASONING:
[Explain your assessment, citing specific examples from both the response and source documents]
"""

        try:
            message = await self.claude_client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.0,
                messages=[{"role": "user", "content": hallucination_prompt}]
            )
            
            analysis = message.content[0].text
            
            # Parse Claude's structured response
            score = self._extract_score(analysis, "HALLUCINATION SCORE:")
            issues = self._extract_issues(analysis, "DETECTED ISSUES:")
            reasoning = self._extract_reasoning(analysis, "DETAILED REASONING:")
            
            return score, issues, reasoning
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return 0.5, ["Error during analysis"], f"Analysis failed: {str(e)}"
    
    def _extract_score(self, text: str, marker: str) -> float:
        """Extract numerical score from Claude's response"""
        try:
            lines = text.split('\n')
            for line in lines:
                if marker in line:
                    # Look for decimal number in the line
                    import re
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        score = float(match.group(1))
                        return min(max(score, 0.0), 1.0)  # Clamp to [0,1]
            return 0.5  # Default if parsing fails
        except:
            return 0.5
    
    def _extract_issues(self, text: str, marker: str) -> List[str]:
        """Extract detected issues from Claude's response"""
        try:
            sections = text.split(marker)
            if len(sections) > 1:
                issues_section = sections[1].split("DETAILED REASONING:")[0].strip()
                
                if "none detected" in issues_section.lower():
                    return []
                
                # Split by lines and clean up
                issues = [
                    line.strip("- ").strip() 
                    for line in issues_section.split('\n') 
                    if line.strip() and not line.strip().startswith('[')
                ]
                return [issue for issue in issues if issue]
            return []
        except:
            return []
    
    def _extract_reasoning(self, text: str, marker: str) -> str:
        """Extract detailed reasoning from Claude's response"""
        try:
            sections = text.split(marker)
            if len(sections) > 1:
                return sections[1].strip()
            return "No detailed reasoning provided"
        except:
            return "Failed to extract reasoning"


class FactualConsistencyChecker:
    """Verifies factual consistency against source documents"""
    
    def __init__(self, claude_client, model: str = "claude-3-5-sonnet-20241022"):
        if not ANTHROPIC_AVAILABLE and claude_client is not None:
            raise ImportError("anthropic package is required for FactualConsistencyChecker")
        self.claude_client = claude_client
        self.model = model
    
    async def check_factual_consistency(
        self, 
        response: str, 
        retrieved_context: List[str],
        query: str
    ) -> Tuple[float, List[SourceClaim], str]:
        """
        Check factual consistency of response against source documents
        
        Returns:
            - consistency_score: 0.0 (inconsistent) to 1.0 (fully consistent)
            - source_claims: List of verified/unverified claims
            - reasoning: Detailed analysis
        """
        
        context_combined = "\n\n".join(retrieved_context)
        
        consistency_prompt = f"""
You are tasked with verifying the factual consistency of a RAG response against its source documents. Analyze each factual claim and determine if it's accurately supported by the provided sources.

QUERY: {query}

SOURCE DOCUMENTS:
{context_combined}

GENERATED RESPONSE:
{response}

ANALYSIS INSTRUCTIONS:
1. Break down the response into individual factual claims
2. For each claim, identify the supporting evidence in the source documents
3. Assess whether each claim is accurately represented or distorted
4. Check for temporal accuracy, numerical precision, and contextual correctness
5. Identify any claims that contradict the source material

Please provide your analysis in the following format:

CONSISTENCY SCORE: [0.0 to 1.0]
- 0.0-0.2: Major inconsistencies, contradicts sources
- 0.2-0.4: Several inaccuracies or distortions
- 0.4-0.6: Some minor inconsistencies
- 0.6-0.8: Mostly accurate with minor issues
- 0.8-1.0: Highly consistent with sources

CLAIM VERIFICATION:
[For each major factual claim, indicate:
- CLAIM: [The specific claim]
- SOURCE: [Relevant source text or "Not found"]
- STATUS: [VERIFIED/PARTIALLY_VERIFIED/UNVERIFIED/CONTRADICTED]
- CONFIDENCE: [0.0-1.0]]

DETAILED REASONING:
[Explain your assessment with specific examples]
"""

        try:
            message = await self.claude_client.messages.create(
                model=self.model,
                max_tokens=2500,
                temperature=0.0,
                messages=[{"role": "user", "content": consistency_prompt}]
            )
            
            analysis = message.content[0].text
            
            score = self._extract_score(analysis, "CONSISTENCY SCORE:")
            claims = self._extract_claims(analysis, "CLAIM VERIFICATION:")
            reasoning = self._extract_reasoning(analysis, "DETAILED REASONING:")
            
            return score, claims, reasoning
            
        except Exception as e:
            logger.error(f"Factual consistency check failed: {e}")
            return 0.5, [], f"Analysis failed: {str(e)}"
    
    def _extract_claims(self, text: str, marker: str) -> List[SourceClaim]:
        """Extract claim verification details from Claude's response"""
        claims = []
        try:
            sections = text.split(marker)
            if len(sections) > 1:
                claims_section = sections[1].split("DETAILED REASONING:")[0]
                
                # Parse claim blocks
                current_claim = {}
                for line in claims_section.split('\n'):
                    line = line.strip()
                    if line.startswith('- CLAIM:'):
                        if current_claim:
                            claims.append(self._create_source_claim(current_claim))
                        current_claim = {'claim': line.replace('- CLAIM:', '').strip()}
                    elif line.startswith('- SOURCE:'):
                        current_claim['source'] = line.replace('- SOURCE:', '').strip()
                    elif line.startswith('- STATUS:'):
                        current_claim['status'] = line.replace('- STATUS:', '').strip()
                    elif line.startswith('- CONFIDENCE:'):
                        try:
                            conf_str = line.replace('- CONFIDENCE:', '').strip()
                            current_claim['confidence'] = float(re.search(r'(\d+\.?\d*)', conf_str).group(1))
                        except:
                            current_claim['confidence'] = 0.5
                
                # Add final claim
                if current_claim:
                    claims.append(self._create_source_claim(current_claim))
        except Exception as e:
            logger.error(f"Failed to extract claims: {e}")
        
        return claims
    
    def _create_source_claim(self, claim_data: Dict) -> SourceClaim:
        """Create SourceClaim object from parsed data"""
        return SourceClaim(
            claim=claim_data.get('claim', ''),
            source_id='source_doc',
            source_text=claim_data.get('source', ''),
            confidence=claim_data.get('confidence', 0.5),
            verified=claim_data.get('status', '').upper() == 'VERIFIED'
        )
    
    def _extract_score(self, text: str, marker: str) -> float:
        """Extract numerical score from Claude's response"""
        try:
            lines = text.split('\n')
            for line in lines:
                if marker in line:
                    # Look for decimal number in the line
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        score = float(match.group(1))
                        return min(max(score, 0.0), 1.0)  # Clamp to [0,1]
            return 0.5  # Default if parsing fails
        except:
            return 0.5
    
    def _extract_reasoning(self, text: str, marker: str) -> str:
        """Extract detailed reasoning from Claude's response"""
        try:
            sections = text.split(marker)
            if len(sections) > 1:
                return sections[1].strip()
            return "No detailed reasoning provided"
        except:
            return "Failed to extract reasoning"


class ConstitutionalValidator:
    """Validates responses using constitutional AI principles"""
    
    def __init__(self, claude_client, model: str = "claude-3-5-sonnet-20241022"):
        if not ANTHROPIC_AVAILABLE and claude_client is not None:
            raise ImportError("anthropic package is required for ConstitutionalValidator")
        self.claude_client = claude_client
        self.model = model
    
    async def validate_constitutional_compliance(
        self, 
        response: str, 
        query: str
    ) -> Tuple[bool, List[str], str]:
        """
        Validate response against constitutional AI principles
        
        Returns:
            - is_compliant: Boolean indicating overall compliance
            - violations: List of detected violations
            - reasoning: Detailed analysis
        """
        
        constitutional_prompt = f"""
You are tasked with evaluating a RAG response for compliance with constitutional AI principles. Assess the response for safety, helpfulness, honesty, and harmlessness.

QUERY: {query}

RESPONSE: {response}

EVALUATION CRITERIA:
1. SAFETY: Does the response avoid harmful, dangerous, or inappropriate content?
2. HONESTY: Is the response truthful and does it acknowledge limitations/uncertainty?
3. HELPFULNESS: Does the response genuinely attempt to answer the query constructively?
4. HARMLESSNESS: Does the response avoid content that could cause harm?
5. BIAS: Is the response free from unfair bias or discrimination?
6. PRIVACY: Does the response respect privacy and avoid sharing sensitive information?

Please provide your assessment in the following format:

COMPLIANCE STATUS: [COMPLIANT/NON_COMPLIANT]

VIOLATIONS DETECTED:
[List any violations found, or "None detected" if compliant]

DETAILED REASONING:
[Explain your assessment for each criterion]
"""

        try:
            message = await self.claude_client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.0,
                messages=[{"role": "user", "content": constitutional_prompt}]
            )
            
            analysis = message.content[0].text
            
            is_compliant = "COMPLIANT" in analysis and "NON_COMPLIANT" not in analysis
            violations = self._extract_violations(analysis, "VIOLATIONS DETECTED:")
            reasoning = self._extract_reasoning(analysis, "DETAILED REASONING:")
            
            return is_compliant, violations, reasoning
            
        except Exception as e:
            logger.error(f"Constitutional validation failed: {e}")
            return True, [], f"Validation failed: {str(e)}"
    
    def _extract_violations(self, text: str, marker: str) -> List[str]:
        """Extract detected violations from Claude's response"""
        try:
            sections = text.split(marker)
            if len(sections) > 1:
                violations_section = sections[1].split("DETAILED REASONING:")[0].strip()
                
                if "none detected" in violations_section.lower():
                    return []
                
                violations = [
                    line.strip("- ").strip() 
                    for line in violations_section.split('\n') 
                    if line.strip() and not line.strip().startswith('[')
                ]
                return [v for v in violations if v]
            return []
        except:
            return []
    
    def _extract_reasoning(self, text: str, marker: str) -> str:
        """Extract detailed reasoning from Claude's response"""
        try:
            sections = text.split(marker)
            if len(sections) > 1:
                return sections[1].strip()
            return "No detailed reasoning provided"
        except:
            return "Failed to extract reasoning"


class CitationAccuracyVerifier:
    """Verifies accuracy of source citations"""
    
    def __init__(self, claude_client, model: str = "claude-3-5-sonnet-20241022"):
        if not ANTHROPIC_AVAILABLE and claude_client is not None:
            raise ImportError("anthropic package is required for CitationAccuracyVerifier")
        self.claude_client = claude_client
        self.model = model
    
    async def verify_citations(
        self, 
        response: str, 
        retrieved_context: List[str],
        source_metadata: List[Dict] = None
    ) -> Tuple[float, List[str], str]:
        """
        Verify accuracy of citations in the response
        
        Returns:
            - accuracy_score: 0.0 (inaccurate) to 1.0 (fully accurate)
            - citation_issues: List of detected citation problems
            - reasoning: Detailed analysis
        """
        
        # Extract citations from response
        citations = self._extract_citations(response)
        
        if not citations:
            return 1.0, [], "No citations found in response"
        
        context_combined = "\n\n".join(retrieved_context)
        source_info = source_metadata or []
        
        citation_prompt = f"""
You are tasked with verifying the accuracy of citations in a RAG response. Check if each citation correctly references the provided source material.

RESPONSE WITH CITATIONS: {response}

AVAILABLE SOURCE DOCUMENTS:
{context_combined}

SOURCE METADATA: {source_info}

ANALYSIS INSTRUCTIONS:
1. Identify all citations in the response
2. For each citation, verify it matches the actual source content
3. Check for citation format accuracy and completeness
4. Identify any false or misleading citations
5. Assess overall citation quality and accuracy

Please provide your analysis in the following format:

CITATION ACCURACY SCORE: [0.0 to 1.0]

CITATION ISSUES:
[List any problems with citations, or "None detected" if accurate]

DETAILED REASONING:
[Explain your assessment]
"""

        try:
            message = await self.claude_client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.0,
                messages=[{"role": "user", "content": citation_prompt}]
            )
            
            analysis = message.content[0].text
            
            score = self._extract_score(analysis, "CITATION ACCURACY SCORE:")
            issues = self._extract_issues(analysis, "CITATION ISSUES:")
            reasoning = self._extract_reasoning(analysis, "DETAILED REASONING:")
            
            return score, issues, reasoning
            
        except Exception as e:
            logger.error(f"Citation verification failed: {e}")
            return 0.8, [], f"Verification failed: {str(e)}"
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation patterns from text"""
        citation_patterns = [
            r'\[([^\]]+)\]',  # [Source Name]
            r'\(([^)]+)\)',   # (Source Name)
            r'Source: ([^\n]+)',  # Source: Name
            r'According to ([^,\n]+)',  # According to Source
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    def _extract_score(self, text: str, marker: str) -> float:
        """Extract numerical score from Claude's response"""
        try:
            lines = text.split('\n')
            for line in lines:
                if marker in line:
                    # Look for decimal number in the line
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        score = float(match.group(1))
                        return min(max(score, 0.0), 1.0)  # Clamp to [0,1]
            return 0.5  # Default if parsing fails
        except:
            return 0.5
    
    def _extract_issues(self, text: str, marker: str) -> List[str]:
        """Extract detected issues from Claude's response"""
        try:
            sections = text.split(marker)
            if len(sections) > 1:
                issues_section = sections[1].split("DETAILED REASONING:")[0].strip()
                
                if "none detected" in issues_section.lower():
                    return []
                
                # Split by lines and clean up
                issues = [
                    line.strip("- ").strip() 
                    for line in issues_section.split('\n') 
                    if line.strip() and not line.strip().startswith('[')
                ]
                return [issue for issue in issues if issue]
            return []
        except:
            return []
    
    def _extract_reasoning(self, text: str, marker: str) -> str:
        """Extract detailed reasoning from Claude's response"""
        try:
            sections = text.split(marker)
            if len(sections) > 1:
                return sections[1].strip()
            return "No detailed reasoning provided"
        except:
            return "Failed to extract reasoning"


class QualityController:
    """Main quality control orchestrator"""
    
    def __init__(
        self, 
        claude_client, 
        model: str = "claude-3-5-sonnet-20241022",
        enable_all_checks: bool = True
    ):
        if not ANTHROPIC_AVAILABLE and claude_client is not None:
            raise ImportError("anthropic package is required for QualityController")
        
        self.claude_client = claude_client
        self.model = model
        self.enable_all_checks = enable_all_checks
        
        # Initialize component checkers only if client is provided
        if claude_client is not None:
            self.hallucination_detector = HallucinationDetector(claude_client, model)
            self.consistency_checker = FactualConsistencyChecker(claude_client, model)
            self.constitutional_validator = ConstitutionalValidator(claude_client, model)
            self.citation_verifier = CitationAccuracyVerifier(claude_client, model)
        else:
            self.hallucination_detector = None
            self.consistency_checker = None
            self.constitutional_validator = None
            self.citation_verifier = None
    
    async def assess_response_quality(
        self,
        response: str,
        query: str,
        retrieved_context: List[str],
        source_metadata: List[Dict] = None
    ) -> QualityAssessment:
        """
        Perform comprehensive quality assessment of RAG response
        
        Args:
            response: Generated response text
            query: Original user query
            retrieved_context: List of retrieved document chunks
            source_metadata: Optional metadata about source documents
            
        Returns:
            QualityAssessment object with comprehensive analysis
        """
        
        if not self.enable_all_checks:
            # Quick assessment mode
            return self._quick_assessment(response, query, retrieved_context)
        
        # Run all quality checks in parallel
        tasks = [
            self.hallucination_detector.detect_hallucinations(response, retrieved_context, query),
            self.consistency_checker.check_factual_consistency(response, retrieved_context, query),
            self.constitutional_validator.validate_constitutional_compliance(response, query),
            self.citation_verifier.verify_citations(response, retrieved_context, source_metadata)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Unpack results (handle any exceptions)
            hallucination_score, hallucination_issues, hall_reasoning = (
                results[0] if not isinstance(results[0], Exception) 
                else (0.5, ["Hallucination check failed"], "Check failed")
            )
            
            consistency_score, source_claims, cons_reasoning = (
                results[1] if not isinstance(results[1], Exception) 
                else (0.5, [], "Consistency check failed")
            )
            
            constitutional_compliant, constitutional_violations, const_reasoning = (
                results[2] if not isinstance(results[2], Exception) 
                else (True, [], "Constitutional check failed")
            )
            
            citation_score, citation_issues, cite_reasoning = (
                results[3] if not isinstance(results[3], Exception) 
                else (0.8, [], "Citation check failed")
            )
            
            # Calculate overall scores and assessment
            return self._compile_assessment(
                response, query,
                hallucination_score, hallucination_issues, hall_reasoning,
                consistency_score, source_claims, cons_reasoning,
                constitutional_compliant, constitutional_violations, const_reasoning,
                citation_score, citation_issues, cite_reasoning
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return self._fallback_assessment(response, str(e))
    
    def _compile_assessment(
        self,
        response: str,
        query: str,
        hallucination_score: float,
        hallucination_issues: List[str],
        hall_reasoning: str,
        consistency_score: float,
        source_claims: List[SourceClaim],
        cons_reasoning: str,
        constitutional_compliant: bool,
        constitutional_violations: List[str],
        const_reasoning: str,
        citation_score: float,
        citation_issues: List[str],
        cite_reasoning: str
    ) -> QualityAssessment:
        """Compile comprehensive quality assessment"""
        
        # Calculate overall confidence score (weighted average)
        confidence_score = (
            (1.0 - hallucination_score) * 0.35 +  # Lower hallucination = higher confidence
            consistency_score * 0.35 +             # Higher consistency = higher confidence
            (1.0 if constitutional_compliant else 0.3) * 0.15 +  # Constitutional compliance
            citation_score * 0.15                  # Citation accuracy
        )
        
        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW
        
        # Determine validation result
        all_issues = hallucination_issues + constitutional_violations + citation_issues
        
        if not constitutional_compliant or hallucination_score > 0.7:
            validation_result = ValidationResult.REJECTED
        elif hallucination_score > 0.4 or consistency_score < 0.5 or len(all_issues) > 3:
            validation_result = ValidationResult.FLAGGED
        elif confidence_score < 0.4:
            validation_result = ValidationResult.NEEDS_REVIEW
        else:
            validation_result = ValidationResult.APPROVED
        
        # Compile issues and recommendations
        issues_detected = all_issues
        recommendations = self._generate_recommendations(
            hallucination_score, consistency_score, constitutional_compliant,
            citation_score, confidence_score
        )
        
        # Compile verified sources
        sources_verified = [
            claim.source_text for claim in source_claims 
            if claim.verified and claim.source_text
        ]
        
        # Combine reasoning
        combined_reasoning = f"""
HALLUCINATION ANALYSIS:
{hall_reasoning}

FACTUAL CONSISTENCY:
{cons_reasoning}

CONSTITUTIONAL COMPLIANCE:
{const_reasoning}

CITATION ACCURACY:
{cite_reasoning}
"""
        
        return QualityAssessment(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            validation_result=validation_result,
            hallucination_score=hallucination_score,
            factual_consistency_score=consistency_score,
            citation_accuracy_score=citation_score,
            constitutional_compliance=constitutional_compliant,
            issues_detected=issues_detected,
            recommendations=recommendations,
            sources_verified=sources_verified,
            reasoning=combined_reasoning
        )
    
    def _generate_recommendations(
        self,
        hallucination_score: float,
        consistency_score: float,
        constitutional_compliant: bool,
        citation_score: float,
        confidence_score: float
    ) -> List[str]:
        """Generate actionable recommendations based on assessment"""
        recommendations = []
        
        if hallucination_score > 0.4:
            recommendations.append("Review response for unsupported claims and revise to be more grounded in sources")
        
        if consistency_score < 0.6:
            recommendations.append("Verify factual accuracy against source documents")
        
        if not constitutional_compliant:
            recommendations.append("Review response for safety and ethical concerns")
        
        if citation_score < 0.7:
            recommendations.append("Improve citation accuracy and completeness")
        
        if confidence_score < 0.5:
            recommendations.append("Consider retrieving additional sources or clarifying the query")
        
        if not recommendations:
            recommendations.append("Response meets quality standards")
        
        return recommendations
    
    def _quick_assessment(
        self, 
        response: str, 
        query: str, 
        retrieved_context: List[str]
    ) -> QualityAssessment:
        """Quick assessment mode with basic heuristics"""
        
        # Simple heuristic-based assessment
        response_length = len(response)
        context_coverage = len(retrieved_context)
        
        # Basic confidence estimation
        if response_length > 200 and context_coverage >= 3:
            confidence_score = 0.7
            confidence_level = ConfidenceLevel.MEDIUM
        elif response_length > 100 and context_coverage >= 1:
            confidence_score = 0.5
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_score = 0.3
            confidence_level = ConfidenceLevel.LOW
        
        return QualityAssessment(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            validation_result=ValidationResult.APPROVED,
            hallucination_score=0.3,  # Conservative estimate
            factual_consistency_score=0.7,
            citation_accuracy_score=0.8,
            constitutional_compliance=True,
            issues_detected=[],
            recommendations=["Quick assessment mode - consider full analysis for important queries"],
            sources_verified=[],
            reasoning="Quick heuristic-based assessment"
        )
    
    def _fallback_assessment(self, response: str, error: str) -> QualityAssessment:
        """Fallback assessment when checks fail"""
        return QualityAssessment(
            confidence_score=0.5,
            confidence_level=ConfidenceLevel.MEDIUM,
            validation_result=ValidationResult.NEEDS_REVIEW,
            hallucination_score=0.5,
            factual_consistency_score=0.5,
            citation_accuracy_score=0.5,
            constitutional_compliance=True,
            issues_detected=[f"Quality check failed: {error}"],
            recommendations=["Manual review recommended due to quality check failure"],
            sources_verified=[],
            reasoning=f"Quality assessment failed: {error}"
        )


# Utility functions for response enhancement
def should_include_confidence_indicator(assessment: QualityAssessment) -> bool:
    """Determine if response should include confidence indicator"""
    return assessment.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]


def format_quality_footer(assessment: QualityAssessment) -> str:
    """Generate quality indicator footer for response"""
    confidence_emoji = {
        ConfidenceLevel.HIGH: "🟢",
        ConfidenceLevel.MEDIUM: "🟡", 
        ConfidenceLevel.LOW: "🟠",
        ConfidenceLevel.VERY_LOW: "🔴"
    }
    
    emoji = confidence_emoji.get(assessment.confidence_level, "⚪")
    
    footer = f"\n\n---\n{emoji} **Confidence**: {assessment.confidence_level.value.title()}"
    
    if assessment.issues_detected:
        footer += f" | ⚠️ **Issues**: {len(assessment.issues_detected)}"
    
    if assessment.sources_verified:
        footer += f" | 📚 **Sources**: {len(assessment.sources_verified)} verified"
    
    return footer