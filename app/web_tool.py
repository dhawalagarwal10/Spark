import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from enum import Enum

class SearchIntent(Enum):
    NEWS_CURRENT = "news_current"
    TECHNICAL_UPDATE = "technical_update"
    PRICE_CHECK = "price_check"
    STATUS_CHECK = "status_check"
    RESEARCH_DEEP = "research_deep"
    FACT_VERIFY = "fact_verify"
    TREND_ANALYSIS = "trend_analysis"
    NO_SEARCH = "no_search"

@dataclass
class SearchContext:
    intent: SearchIntent
    urgency: float  # 0-1 scale
    specificity: float  # 0-1 scale
    temporal_sensitivity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    suggested_query: str
    search_depth: int  # 1-3 (shallow to deep)
    domain_focus: Optional[str] = None

class SmartWebSearch:
    def __init__(self):
        # Time-sensitive keywords
        self.temporal_keywords = {
            'immediate': ['now', 'current', 'today', 'breaking', 'live', 'real-time', 'latest'],
            'recent': ['recent', 'new', 'updated', 'this week', 'this month', '2024', '2025'],
            'historical': ['history', 'past', 'originally', 'when did', 'founded', 'created']
        }
        
        # Domain-specific indicators
        self.domain_indicators = {
            'news': ['news', 'breaking', 'reported', 'announced', 'statement', 'press release'],
            'tech': ['github', 'api', 'framework', 'library', 'version', 'release', 'update', 'bug'],
            'finance': ['price', 'stock', 'market', 'trading', 'crypto', 'bitcoin', 'portfolio'],
            'academic': ['research', 'study', 'paper', 'journal', 'findings', 'analysis'],
            'social': ['trending', 'viral', 'popular', 'discussion', 'reddit', 'twitter']
        }
        
        # Search necessity indicators
        self.search_triggers = {
            'high_priority': [
                'what happened to', 'latest news about', 'current status of',
                'is [name] still', 'recent developments in', 'breaking news',
                'live updates', 'real-time data', 'current price of'
            ],
            'medium_priority': [
                'recent', 'new version', 'updated', 'latest release',
                'current trends', 'what\'s new', 'recent changes'
            ],
            'low_priority': [
                'more information', 'details about', 'learn more',
                'additional resources', 'examples of'
            ]
        }
        
        # Knowledge cutoff awareness
        self.static_knowledge_indicators = [
            'what is', 'how does', 'explain', 'definition of', 'concept of',
            'theory of', 'principle of', 'basics of', 'fundamentals'
        ]
        
        # Search depth indicators
        self.depth_indicators = {
            'shallow': ['quick', 'brief', 'summary', 'overview', 'tldr'],
            'medium': ['detailed', 'comprehensive', 'thorough', 'in-depth'],
            'deep': ['research', 'analysis', 'extensive', 'complete study', 'all information']
        }

    def analyze_search_necessity(self, query: str, context: str = "") -> SearchContext:
        """Intelligently analyze if web search is needed and how to conduct it"""
        
        query_lower = query.lower()
        full_text = f"{query} {context}".lower()
        
        # Determine temporal sensitivity
        temporal_score = self._calculate_temporal_sensitivity(full_text)
        
        # Determine search intent
        intent = self._classify_search_intent(query_lower, temporal_score)
        
        # Calculate urgency based on keywords and temporal sensitivity
        urgency = self._calculate_urgency(query_lower, temporal_score)
        
        # Determine specificity
        specificity = self._calculate_specificity(query_lower)
        
        # Determine search depth needed
        search_depth = self._determine_search_depth(query_lower)
        
        # Generate optimized search query
        optimized_query = self._optimize_search_query(query, intent, temporal_score)
        
        # Determine domain focus
        domain_focus = self._identify_domain_focus(query_lower)
        
        # Calculate confidence in search necessity
        confidence = self._calculate_search_confidence(
            intent, urgency, temporal_score, specificity
        )
        
        return SearchContext(
            intent=intent,
            urgency=urgency,
            specificity=specificity,
            temporal_sensitivity=temporal_score,
            confidence=confidence,
            suggested_query=optimized_query,
            search_depth=search_depth,
            domain_focus=domain_focus
        )

    def _calculate_temporal_sensitivity(self, text: str) -> float:
        """Calculate how time-sensitive the query is"""
        score = 0.0
        
        # Immediate temporal indicators
        immediate_matches = sum(1 for keyword in self.temporal_keywords['immediate'] 
                              if keyword in text)
        score += immediate_matches * 0.4
        
        # Recent temporal indicators
        recent_matches = sum(1 for keyword in self.temporal_keywords['recent'] 
                           if keyword in text)
        score += recent_matches * 0.2
        
        # Date/time patterns
        if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', text):
            score += 0.1
        
        if re.search(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text):
            score += 0.2
        
        # Question patterns indicating current status
        if any(pattern in text for pattern in ['is still', 'currently', 'right now', 'at the moment']):
            score += 0.3
        
        return min(1.0, score)

    def _classify_search_intent(self, query: str, temporal_score: float) -> SearchIntent:
        """Classify the type of search intent"""
        
        # Check for static knowledge patterns first
        if any(pattern in query for pattern in self.static_knowledge_indicators):
            if temporal_score < 0.3:
                return SearchIntent.NO_SEARCH
        
        # News and current events
        if any(keyword in query for keyword in self.domain_indicators['news']):
            return SearchIntent.NEWS_CURRENT
        
        # Technical updates
        if any(keyword in query for keyword in self.domain_indicators['tech']) and temporal_score > 0.4:
            return SearchIntent.TECHNICAL_UPDATE
        
        # Price checks
        if any(keyword in query for keyword in self.domain_indicators['finance']):
            return SearchIntent.PRICE_CHECK
        
        # Status checks
        if any(pattern in query for pattern in ['status of', 'is still', 'currently', 'what happened to']):
            return SearchIntent.STATUS_CHECK
        
        # Research needs
        if any(keyword in query for keyword in self.domain_indicators['academic']):
            return SearchIntent.RESEARCH_DEEP
        
        # Fact verification
        if any(pattern in query for pattern in ['is it true', 'verify', 'confirm', 'fact check']):
            return SearchIntent.FACT_VERIFY
        
        # Trend analysis
        if any(keyword in query for keyword in self.domain_indicators['social']):
            return SearchIntent.TREND_ANALYSIS
        
        # High temporal sensitivity usually means search needed
        if temporal_score > 0.6:
            return SearchIntent.NEWS_CURRENT
        
        return SearchIntent.NO_SEARCH

    def _calculate_urgency(self, query: str, temporal_score: float) -> float:
        """Calculate urgency of the search need"""
        urgency = temporal_score * 0.5
        
        # High priority triggers
        high_priority_matches = sum(1 for trigger in self.search_triggers['high_priority'] 
                                  if any(word in query for word in trigger.split()))
        urgency += high_priority_matches * 0.3
        
        # Medium priority triggers
        medium_priority_matches = sum(1 for trigger in self.search_triggers['medium_priority'] 
                                    if trigger in query)
        urgency += medium_priority_matches * 0.2
        
        # Emergency indicators
        if any(word in query for word in ['urgent', 'emergency', 'critical', 'immediate']):
            urgency += 0.4
        
        return min(1.0, urgency)

    def _calculate_specificity(self, query: str) -> float:
        """Calculate how specific the query is"""
        specificity = 0.0
        
        # Proper nouns indicate specificity
        proper_noun_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        proper_nouns = len(re.findall(proper_noun_pattern, query))
        specificity += min(0.4, proper_nouns * 0.1)
        
        # Numbers and dates
        if re.search(r'\d+', query):
            specificity += 0.2
        
        # Quotes indicate exact search
        if '"' in query:
            specificity += 0.3
        
        # Technical terms
        if any(keyword in query for domain_keywords in self.domain_indicators.values() 
               for keyword in domain_keywords):
            specificity += 0.2
        
        # Question length (longer = more specific)
        word_count = len(query.split())
        if word_count > 10:
            specificity += 0.2
        elif word_count > 5:
            specificity += 0.1
        
        return min(1.0, specificity)

    def _determine_search_depth(self, query: str) -> int:
        """Determine how deep the search should be (1-3)"""
        if any(keyword in query for keyword in self.depth_indicators['deep']):
            return 3
        elif any(keyword in query for keyword in self.depth_indicators['medium']):
            return 2
        elif any(keyword in query for keyword in self.depth_indicators['shallow']):
            return 1
        else:
            # Default based on query complexity
            word_count = len(query.split())
            if word_count > 15:
                return 3
            elif word_count > 8:
                return 2
            else:
                return 1

    def _optimize_search_query(self, original_query: str, intent: SearchIntent, temporal_score: float) -> str:
        """Generate an optimized search query"""
        query = original_query.strip()
        
        # Remove conversational fluff
        fluff_patterns = [
            r'\b(can you|could you|please|help me|i want to|i need to|tell me)\b',
            r'\b(about|regarding|concerning)\b',
        ]
        
        for pattern in fluff_patterns:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
        
        # Add temporal modifiers based on intent
        if intent in [SearchIntent.NEWS_CURRENT, SearchIntent.STATUS_CHECK] and temporal_score > 0.5:
            if not any(word in query.lower() for word in ['latest', 'current', 'recent', '2024', '2025']):
                query = f"{query} latest 2025"
        
        # Add domain-specific modifiers
        if intent == SearchIntent.TECHNICAL_UPDATE:
            query += " github documentation"
        elif intent == SearchIntent.PRICE_CHECK:
            query += " current price"
        elif intent == SearchIntent.NEWS_CURRENT:
            query += " news today"
        
        # Clean up extra spaces
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query

    def _identify_domain_focus(self, query: str) -> Optional[str]:
        """Identify which domain to focus the search on"""
        domain_scores = {}
        
        for domain, keywords in self.domain_indicators.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return None

    def _calculate_search_confidence(self, intent: SearchIntent, urgency: float, 
                                   temporal_score: float, specificity: float) -> float:
        """Calculate confidence that search is needed"""
        if intent == SearchIntent.NO_SEARCH:
            return 0.1
        
        # Base confidence from intent
        intent_confidence = {
            SearchIntent.NEWS_CURRENT: 0.9,
            SearchIntent.TECHNICAL_UPDATE: 0.8,
            SearchIntent.PRICE_CHECK: 0.9,
            SearchIntent.STATUS_CHECK: 0.8,
            SearchIntent.RESEARCH_DEEP: 0.7,
            SearchIntent.FACT_VERIFY: 0.8,
            SearchIntent.TREND_ANALYSIS: 0.7
        }
        
        base_confidence = intent_confidence.get(intent, 0.5)
        
        # Adjust based on other factors
        confidence = base_confidence * 0.5 + urgency * 0.2 + temporal_score * 0.2 + specificity * 0.1
        
        return min(1.0, confidence)

    def should_search(self, query: str, context: str = "", threshold: float = 0.6) -> Tuple[bool, SearchContext]:
        """Determine if a search should be performed"""
        search_context = self.analyze_search_necessity(query, context)
        should_search = search_context.confidence >= threshold
        
        return should_search, search_context

    def enhanced_search(self, query: str, context: str = "") -> Optional[Dict]:
        """Perform intelligent web search with context awareness"""
        should_search, search_context = self.should_search(query, context)
        
        if not should_search:
            return {
                'search_performed': False,
                'reason': f'Search not needed (confidence: {search_context.confidence:.2f})',
                'context': search_context
            }
        
        # Perform the actual search
        search_results = self._perform_contextual_search(search_context)
        
        return {
            'search_performed': True,
            'context': search_context,
            'results': search_results,
            'metadata': {
                'query_original': query,
                'query_optimized': search_context.suggested_query,
                'search_depth': search_context.search_depth,
                'domain_focus': search_context.domain_focus
            }
        }

    def _perform_contextual_search(self, context: SearchContext) -> List[Dict]:
        """Perform search with context awareness"""
        try:
            # Use the optimized query
            results = self._search_web(context.suggested_query, 
                                     max_results=min(10, context.search_depth * 3))
            
            # Filter and rank results based on context
            filtered_results = self._filter_results_by_context(results, context)
            
            # If deep search is needed, scrape top results
            if context.search_depth >= 2 and filtered_results:
                for i, result in enumerate(filtered_results[:2]):
                    if 'url' in result:
                        scraped_content = self.scrape_url(result['url'])
                        result['full_content'] = scraped_content
            
            return filtered_results
            
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}", "context": context}]

    def _filter_results_by_context(self, results: List[Dict], context: SearchContext) -> List[Dict]:
        """Filter and rank results based on search context"""
        if not results or 'error' in results[0]:
            return results
        
        scored_results = []
        
        for result in results:
            score = 0.0
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            content = f"{title} {snippet}"
            
            # Domain relevance scoring
            if context.domain_focus:
                domain_keywords = self.domain_indicators.get(context.domain_focus, [])
                domain_matches = sum(1 for keyword in domain_keywords if keyword in content)
                score += domain_matches * 0.3
            
            # Temporal relevance for time-sensitive queries
            if context.temporal_sensitivity > 0.5:
                recent_indicators = sum(1 for keyword in self.temporal_keywords['recent'] 
                                      if keyword in content)
                score += recent_indicators * 0.2
            
            # Specificity matching
            if context.specificity > 0.5:
                # Prefer results with exact matches
                if any(word in content for word in context.suggested_query.lower().split()):
                    score += 0.4
            
            result['relevance_score'] = score
            scored_results.append(result)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return scored_results[:context.search_depth * 2]

    def _search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Enhanced web search with better error handling"""
        search_url = "https://lite.duckduckgo.com/lite/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            response = requests.post(search_url, data={"q": query}, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            for a in soup.select("a.result-link"):
                title = a.text.strip()
                url = a.get("href", "")
                
                # Find snippet
                snippet_tag = a.find_next("div", class_="result-snippet")
                snippet = snippet_tag.text.strip() if snippet_tag else ""
                
                # Clean and validate URL
                if url and not url.startswith('http'):
                    url = f"https:{url}" if url.startswith('//') else f"https://{url}"
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "relevance_score": 0.0
                })
                
                if len(results) >= max_results:
                    break

            return results if results else [{"error": "No results found"}]

        except requests.RequestException as e:
            return [{"error": f"Network error: {str(e)}"}]
        except Exception as e:
            return [{"error": f"Search parsing error: {str(e)}"}]

    def scrape_url(self, url: str) -> str:
        """Enhanced web scraping with better content extraction"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "header", "footer", "nav", "aside", 
                           "advertisement", "ads", "popup", "modal"]):
                tag.extract()

            # Try to find main content areas
            main_content = None
            for selector in ["main", "article", ".content", ".post", "#content"]:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up the text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            
            # Remove very short lines (likely navigation/footer items)
            content_lines = [line for line in lines if len(line) > 20]
            
            return "\n".join(content_lines[:100])  # Limit to first 100 meaningful lines

        except requests.RequestException as e:
            return f"[Network error accessing {url}: {str(e)}]"
        except Exception as e:
            return f"[Error scraping {url}: {str(e)}]"

    def debug_search_decision(self, query: str, context: str = "") -> Dict:
        """Debug method to understand search decisions"""
        search_context = self.analyze_search_necessity(query, context)
        should_search, _ = self.should_search(query, context)
        
        return {
            'query': query,
            'context': context,
            'should_search': should_search,
            'search_context': {
                'intent': search_context.intent.value,
                'urgency': search_context.urgency,
                'specificity': search_context.specificity,
                'temporal_sensitivity': search_context.temporal_sensitivity,
                'confidence': search_context.confidence,
                'suggested_query': search_context.suggested_query,
                'search_depth': search_context.search_depth,
                'domain_focus': search_context.domain_focus
            }
        }

# Legacy compatibility functions
def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """Legacy compatibility function"""
    smart_search = SmartWebSearch()
    return smart_search._search_web(query, max_results)

def scrape_url(url: str) -> str:
    """Legacy compatibility function"""
    smart_search = SmartWebSearch()
    return smart_search.scrape_url(url)

# Usage example
if __name__ == "__main__":
    smart_search = SmartWebSearch()
    
    # Test queries
    test_queries = [
        "What is machine learning?",  # Should not search (static knowledge)
        "Latest news about OpenAI",   # Should search (current events)
        "Current Bitcoin price",      # Should search (real-time data)
        "How does photosynthesis work?",  # Should not search (static knowledge)
        "Recent updates to Python 3.12",  # Should search (technical updates)
        "What happened to Twitter today?",  # Should search (current status)
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        
        debug_info = smart_search.debug_search_decision(query)
        print(f"Decision: {'SEARCH' if debug_info['should_search'] else 'NO SEARCH'}")
        print(f"Confidence: {debug_info['search_context']['confidence']:.2f}")
        print(f"Intent: {debug_info['search_context']['intent']}")
        print(f"Optimized Query: {debug_info['search_context']['suggested_query']}")
        
        if debug_info['should_search']:
            result = smart_search.enhanced_search(query)
            if result['search_performed']:
                print(f"Found {len(result['results'])} results")
                if result['results'] and 'error' not in result['results'][0]:
                    print(f"Top result: {result['results'][0]['title']}")