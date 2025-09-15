from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from google.adk import Agent, ToolConfig
import logging

logger = logging.getLogger(__name__)

class BaseStockAgent(Agent, ABC):
    """Base class for all stock analysis agents"""

    def __init__(self, name: str, description: str, tools: Optional[List[ToolConfig]] = None):
        super().__init__(
            name=name,
            description=description,
            tools=tools or []
        )
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Core analysis method that each agent must implement"""
        pass

    def validate_input(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate that input contains required fields"""
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        return True

    def log_analysis_start(self, symbol: str = None):
        """Log the start of analysis"""
        symbol_info = f" for {symbol}" if symbol else ""
        self.logger.info(f"Starting {self.name} analysis{symbol_info}")

    def log_analysis_complete(self, symbol: str = None, result_summary: str = None):
        """Log the completion of analysis"""
        symbol_info = f" for {symbol}" if symbol else ""
        summary_info = f": {result_summary}" if result_summary else ""
        self.logger.info(f"Completed {self.name} analysis{symbol_info}{summary_info}")

class DataCollectionAgent(BaseStockAgent):
    """Base class for data collection agents"""

    @abstractmethod
    async def collect_data(self, symbols: List[str], **kwargs) -> Dict[str, Any]:
        """Collect data for given symbols"""
        pass

class AnalysisAgent(BaseStockAgent):
    """Base class for analysis agents"""

    @abstractmethod
    async def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on analysis"""
        pass

class CoordinatorAgent(BaseStockAgent):
    """Base class for coordinator agents that orchestrate other agents"""

    def __init__(self, name: str, description: str, child_agents: List[BaseStockAgent] = None):
        super().__init__(name, description)
        self.child_agents = child_agents or []

    async def coordinate_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Coordinate analysis across multiple child agents"""
        results = {}

        for agent in self.child_agents:
            try:
                agent_result = await agent.analyze({"symbols": symbols})
                results[agent.name] = agent_result
            except Exception as e:
                self.logger.error(f"Error in {agent.name}: {str(e)}")
                results[agent.name] = {"error": str(e)}

        return results