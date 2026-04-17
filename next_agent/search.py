from abc import ABC, abstractmethod
from tavily import TavilyClient
from .vars import TAVILY_API_KEY

class SearchProvider(ABC):
    """Abstract base class for web search providers."""
    
    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> str:
        """Executes a search and returns a formatted string of results."""
        pass


class TavilyProvider(SearchProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or TAVILY_API_KEY
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is not set.")
        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, num_results: int = 10) -> str:
        data = self.client.search(
            query=query, 
            max_results=num_results, 
            include_answer=False
        )
        results = data.get("results", [])
        if not results:
            raise ValueError("No results returned from Tavily.")
            
        return "\n".join([f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}\n" for r in results])


class GoogleProvider(SearchProvider):
    def search(self, query: str, num_results: int = 10) -> str:
        from googlesearch import search
        
        # advanced=True returns objects with title, url, and description
        results = list(search(query, num_results=num_results, advanced=True))
        if not results:
            raise ValueError("No results returned from Google.")
            
        return "\n".join([f"Title: {r.title}\nURL: {r.url}\nContent: {r.description}\n" for r in results])


class DDGSProvider(SearchProvider):
    def search(self, query: str, num_results: int = 10) -> str:
        from ddgs import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            
        if not results:
            raise ValueError("No results returned from DuckDuckGo.")
            
        return "\n".join([f"Title: {r.get('title')}\nURL: {r.get('href')}\nContent: {r.get('body')}\n" for r in results])


class HybridProvider(SearchProvider):
    """Tries Tavily -> Google -> DDGS -> Error."""
    
    def search(self, query: str, num_results: int = 10) -> str:
        print(f"\n🌐 Initiating Hybrid Web Search for: '{query}'")
        try:
            print("   -> Trying Tavily...")
            tavily = TavilyProvider()
            return tavily.search(query, num_results)
        except Exception as e:
            print(f"   [!] Tavily failed or unavailable: {e}")
        try:
            print("   -> Trying Google...")
            google = GoogleProvider()
            return google.search(query, num_results)
        except Exception as e:
            print(f"   [!] Google failed: {e}")
        try:
            print("   -> Trying DuckDuckGo...")
            ddgs_provider = DDGSProvider()
            return ddgs_provider.search(query, num_results)
        except Exception as e:
            print(f"   [!] DuckDuckGo failed: {e}")
        return "Error: All search providers failed to retrieve results. Please try again later."
