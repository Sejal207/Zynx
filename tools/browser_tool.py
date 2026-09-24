import time
from typing import Any, Optional
from playwright.sync_api import sync_playwright
from crewai.tools import BaseTool
from pydantic import Field, ConfigDict

class PlaywrightBrowserTool(BaseTool):
    name: str = "browser_interaction_tool"
    description: str = (
        "Navigate, click, type, and scroll on a website. "
        "Input should be an action (click, type, scroll, navigate) and its parameters. "
        "Actions: 'navigate' (value=URL), 'click' (selector=CSS), 'type' (selector=CSS, value=text), 'scroll' (no params)."
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Internal state
    _playwright: Any = None
    _browser: Any = None
    _page: Any = None

    def _get_page(self):
        if not self._browser:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=False)
            self._page = self._browser.new_page()
        return self._page

    def _run(self, action: str, value: Optional[str] = None, selector: Optional[str] = None) -> str:
        """Execute a browser action synchronously."""
        page = self._get_page()
        
        # PSYCHOLOGY HOOK: Simulating human-like interaction delay
        time.sleep(0.5) 

        try:
            if action == "navigate":
                if not value.startswith("http"):
                    value = f"https://{value}"
                page.goto(value)
                page.wait_for_load_state("networkidle")
                
                # Simplified page structure for the LLM to understand
                # We'll just return the title and some key buttons/links for now
                title = page.title()
                return f"Successfully navigated to {value}. Page Title: '{title}'. Please decide your next action."

            elif action == "click":
                # Ensure element exists before clicking
                page.wait_for_selector(selector, timeout=5000)
                page.click(selector, delay=100)
                page.wait_for_load_state("networkidle")
                return f"Clicked element: {selector}. Page updated."

            elif action == "type":
                page.wait_for_selector(selector, timeout=5000)
                page.fill(selector, value)
                return f"Typed '{value}' into {selector}."

            elif action == "scroll":
                page.mouse.wheel(0, 800)
                time.sleep(0.5) # Time to 'read'
                return "Scrolled down 800px."

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            return f"Error executing {action}: {str(e)}. Try a different selector or action."

    def __del__(self):
        """Cleanup playwright on deletion"""
        if hasattr(self, '_browser') and self._browser:
            self._browser.close()
        if hasattr(self, '_playwright') and self._playwright:
            self._playwright.stop()