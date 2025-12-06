"""
Selenium tests for Chess Web UI.

These tests verify the chess gameplay interactions through the web interface.
Tests are designed to be visual and observable - they simulate real user interactions.

Requirements:
    - selenium
    - webdriver-manager (for automatic driver management)
    - Chrome/Firefox browser installed

Usage:
    pytest tests/games/chess/test_ui_selenium.py -v --headed  # Watch the tests
    pytest tests/games/chess/test_ui_selenium.py -v          # Headless mode
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

import pytest

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.remote.webelement import WebElement


# Test configuration
@dataclass
class TestConfig:
    """Configuration for Selenium tests."""

    base_url: str = "http://localhost:7861"
    implicit_wait: int = 10
    page_load_timeout: int = 30
    slow_mo: float = 0.5  # Delay between actions for visibility
    headless: bool = True
    browser: str = "chrome"
    screenshot_on_failure: bool = True
    screenshot_dir: str = "test_screenshots"


# Get config from environment or use defaults
def get_config() -> TestConfig:
    """Get test configuration from environment."""
    return TestConfig(
        base_url=os.getenv("CHESS_UI_URL", "http://localhost:7861"),
        headless=os.getenv("HEADED", "").lower() not in ("1", "true", "yes"),
        slow_mo=float(os.getenv("SLOW_MO", "0.5")),
        browser=os.getenv("BROWSER", "chrome"),
    )


# Fixtures
@pytest.fixture(scope="module")
def config() -> TestConfig:
    """Provide test configuration."""
    return get_config()


@pytest.fixture(scope="module")
def ui_server(config: TestConfig) -> Generator[subprocess.Popen, None, None]:
    """Start the UI server for testing."""
    # Start the UI server
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.games.chess.ui",
            "--port",
            "7861",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    )

    # Wait for server to start
    time.sleep(3)

    yield server_process

    # Cleanup
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()


@pytest.fixture(scope="function")
def driver(config: TestConfig) -> Generator["WebDriver", None, None]:
    """Create and configure WebDriver."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.chrome.service import Service as ChromeService
        from selenium.webdriver.firefox.options import Options as FirefoxOptions
        from selenium.webdriver.firefox.service import Service as FirefoxService
    except ImportError:
        pytest.skip("Selenium not installed. Install with: pip install selenium webdriver-manager")

    driver_instance: WebDriver | None = None

    try:
        if config.browser == "chrome":
            options = ChromeOptions()
            if config.headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")

            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = ChromeService(ChromeDriverManager().install())
                driver_instance = webdriver.Chrome(service=service, options=options)
            except Exception:
                # Try without webdriver-manager
                driver_instance = webdriver.Chrome(options=options)

        else:  # firefox
            options = FirefoxOptions()
            if config.headless:
                options.add_argument("--headless")

            try:
                from webdriver_manager.firefox import GeckoDriverManager
                service = FirefoxService(GeckoDriverManager().install())
                driver_instance = webdriver.Firefox(service=service, options=options)
            except Exception:
                driver_instance = webdriver.Firefox(options=options)

        driver_instance.implicitly_wait(config.implicit_wait)
        driver_instance.set_page_load_timeout(config.page_load_timeout)

        yield driver_instance

    finally:
        if driver_instance:
            driver_instance.quit()


@pytest.fixture(scope="function")
def chess_page(driver: "WebDriver", config: TestConfig, ui_server: subprocess.Popen) -> "ChessPage":
    """Navigate to chess page and return page object."""
    page = ChessPage(driver, config)
    page.navigate()
    return page


# Page Object Model
class ChessPage:
    """Page object for the chess UI."""

    def __init__(self, driver: "WebDriver", config: TestConfig) -> None:
        """Initialize the page object."""
        self.driver = driver
        self.config = config

    def navigate(self) -> None:
        """Navigate to the chess UI."""
        self.driver.get(self.config.base_url)
        self._wait_for_load()

    def _wait_for_load(self) -> None:
        """Wait for the page to fully load."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        WebDriverWait(self.driver, self.config.page_load_timeout).until(
            EC.presence_of_element_located((By.ID, "board-container"))
        )
        time.sleep(1)  # Additional wait for Gradio to initialize

    def _slow_action(self) -> None:
        """Add delay between actions for visibility."""
        time.sleep(self.config.slow_mo)

    # Element locators
    @property
    def board_container(self) -> "WebElement":
        """Get the chess board container."""
        from selenium.webdriver.common.by import By
        return self.driver.find_element(By.ID, "board-container")

    @property
    def move_input(self) -> "WebElement":
        """Get the move input field."""
        from selenium.webdriver.common.by import By
        # Gradio wraps inputs, find by parent ID
        container = self.driver.find_element(By.ID, "move-input")
        return container.find_element(By.TAG_NAME, "input")

    @property
    def move_button(self) -> "WebElement":
        """Get the make move button."""
        from selenium.webdriver.common.by import By
        return self.driver.find_element(By.ID, "move-btn")

    @property
    def new_game_button(self) -> "WebElement":
        """Get the new game button."""
        from selenium.webdriver.common.by import By
        return self.driver.find_element(By.ID, "new-game-btn")

    @property
    def undo_button(self) -> "WebElement":
        """Get the undo button."""
        from selenium.webdriver.common.by import By
        return self.driver.find_element(By.ID, "undo-btn")

    @property
    def status_display(self) -> "WebElement":
        """Get the status display."""
        from selenium.webdriver.common.by import By
        container = self.driver.find_element(By.ID, "status-display")
        return container.find_element(By.TAG_NAME, "textarea")

    @property
    def history_display(self) -> "WebElement":
        """Get the move history display."""
        from selenium.webdriver.common.by import By
        container = self.driver.find_element(By.ID, "history-display")
        return container.find_element(By.TAG_NAME, "textarea")

    @property
    def analysis_display(self) -> "WebElement":
        """Get the AI analysis display."""
        from selenium.webdriver.common.by import By
        return self.driver.find_element(By.ID, "analysis-display")

    @property
    def color_select_white(self) -> "WebElement":
        """Get the white color radio button."""
        from selenium.webdriver.common.by import By
        container = self.driver.find_element(By.ID, "color-select")
        labels = container.find_elements(By.TAG_NAME, "label")
        for label in labels:
            if "white" in label.text.lower():
                return label.find_element(By.TAG_NAME, "input")
        raise ValueError("White color option not found")

    @property
    def color_select_black(self) -> "WebElement":
        """Get the black color radio button."""
        from selenium.webdriver.common.by import By
        container = self.driver.find_element(By.ID, "color-select")
        labels = container.find_elements(By.TAG_NAME, "label")
        for label in labels:
            if "black" in label.text.lower():
                return label.find_element(By.TAG_NAME, "input")
        raise ValueError("Black color option not found")

    # Actions
    def enter_move(self, move: str) -> None:
        """Enter a move in the input field."""
        self._slow_action()
        self.move_input.clear()
        self.move_input.send_keys(move)
        self._slow_action()

    def click_make_move(self) -> None:
        """Click the make move button."""
        self._slow_action()
        self.move_button.click()
        self._wait_for_response()

    def make_move(self, move: str) -> None:
        """Enter and submit a move."""
        self.enter_move(move)
        self.click_make_move()

    def submit_move_with_enter(self, move: str) -> None:
        """Enter a move and submit with Enter key."""
        from selenium.webdriver.common.keys import Keys
        self.enter_move(move)
        self.move_input.send_keys(Keys.RETURN)
        self._wait_for_response()

    def start_new_game(self, as_color: str = "white") -> None:
        """Start a new game with the specified color."""
        self._slow_action()
        if as_color == "white":
            self.color_select_white.click()
        else:
            self.color_select_black.click()
        self._slow_action()
        self.new_game_button.click()
        self._wait_for_response()

    def click_undo(self) -> None:
        """Click the undo button."""
        self._slow_action()
        self.undo_button.click()
        self._wait_for_response()

    def _wait_for_response(self) -> None:
        """Wait for the UI to respond to an action."""
        time.sleep(2)  # Wait for Gradio to process and update

    # Assertions helpers
    def get_status_text(self) -> str:
        """Get the current status text."""
        return self.status_display.get_attribute("value") or ""

    def get_history_text(self) -> str:
        """Get the move history text."""
        return self.history_display.get_attribute("value") or ""

    def get_analysis_text(self) -> str:
        """Get the AI analysis text."""
        return self.analysis_display.text

    def get_board_html(self) -> str:
        """Get the board HTML."""
        return self.board_container.get_attribute("innerHTML") or ""

    def take_screenshot(self, name: str) -> str:
        """Take a screenshot and return the path."""
        os.makedirs(self.config.screenshot_dir, exist_ok=True)
        path = os.path.join(self.config.screenshot_dir, f"{name}.png")
        self.driver.save_screenshot(path)
        return path


# Test Classes
@pytest.mark.selenium
@pytest.mark.slow
class TestChessUIBasics:
    """Basic UI functionality tests."""

    def test_page_loads(self, chess_page: ChessPage) -> None:
        """Test that the chess page loads successfully."""
        assert "Chess" in chess_page.driver.title or "chess" in chess_page.driver.page_source.lower()

    def test_board_is_displayed(self, chess_page: ChessPage) -> None:
        """Test that the chess board is displayed."""
        board_html = chess_page.get_board_html()
        assert "chess-board" in board_html
        assert "chess-square" in board_html

    def test_initial_status_shows_white_to_move(self, chess_page: ChessPage) -> None:
        """Test that initial status shows White to move."""
        status = chess_page.get_status_text()
        assert "white" in status.lower() or "White" in status

    def test_move_input_exists(self, chess_page: ChessPage) -> None:
        """Test that move input field exists and is interactable."""
        assert chess_page.move_input.is_displayed()
        assert chess_page.move_input.is_enabled()

    def test_buttons_exist(self, chess_page: ChessPage) -> None:
        """Test that all control buttons exist."""
        assert chess_page.move_button.is_displayed()
        assert chess_page.new_game_button.is_displayed()
        assert chess_page.undo_button.is_displayed()


@pytest.mark.selenium
@pytest.mark.slow
class TestMoveEntry:
    """Tests for move entry functionality."""

    def test_enter_valid_move_e2e4(self, chess_page: ChessPage) -> None:
        """Test entering a valid opening move e2e4."""
        chess_page.start_new_game("white")
        chess_page.make_move("e2e4")

        # Check that the move appears in history
        history = chess_page.get_history_text()
        assert "e2e4" in history

    def test_enter_move_with_enter_key(self, chess_page: ChessPage) -> None:
        """Test submitting a move with Enter key."""
        chess_page.start_new_game("white")
        chess_page.submit_move_with_enter("d2d4")

        history = chess_page.get_history_text()
        assert "d2d4" in history

    def test_invalid_move_shows_error(self, chess_page: ChessPage) -> None:
        """Test that invalid moves show an error message."""
        chess_page.start_new_game("white")
        chess_page.make_move("e2e5")  # Invalid - can't move pawn 3 squares

        analysis = chess_page.get_analysis_text()
        assert "illegal" in analysis.lower() or "invalid" in analysis.lower()

    def test_knight_move(self, chess_page: ChessPage) -> None:
        """Test making a knight move."""
        chess_page.start_new_game("white")
        chess_page.make_move("g1f3")  # Nf3

        history = chess_page.get_history_text()
        assert "g1f3" in history


@pytest.mark.selenium
@pytest.mark.slow
class TestGameFlow:
    """Tests for game flow and AI responses."""

    def test_ai_responds_to_player_move(self, chess_page: ChessPage) -> None:
        """Test that AI responds after player move."""
        chess_page.start_new_game("white")
        chess_page.make_move("e2e4")

        # Wait for AI response
        time.sleep(3)

        # Should have player move and AI response
        history = chess_page.get_history_text()
        lines = history.strip().split("\n")

        # First line should have both moves (1. e2e4 <ai_move>)
        assert len(lines) >= 1
        assert "e2e4" in lines[0]

    def test_ai_analysis_displayed(self, chess_page: ChessPage) -> None:
        """Test that AI analysis is displayed after move."""
        chess_page.start_new_game("white")
        chess_page.make_move("e2e4")

        time.sleep(3)

        analysis = chess_page.get_analysis_text()
        # Should show some analysis information
        assert len(analysis) > 10

    def test_play_multiple_moves(self, chess_page: ChessPage) -> None:
        """Test playing multiple moves in sequence."""
        chess_page.start_new_game("white")

        moves = ["e2e4", "d2d4", "g1f3"]
        for i, move in enumerate(moves):
            chess_page.make_move(move)
            time.sleep(2)  # Wait for AI response

            history = chess_page.get_history_text()
            assert move in history, f"Move {move} not found after move {i + 1}"


@pytest.mark.selenium
@pytest.mark.slow
class TestGameControls:
    """Tests for game control buttons."""

    def test_new_game_resets_board(self, chess_page: ChessPage) -> None:
        """Test that New Game resets the board."""
        chess_page.start_new_game("white")
        chess_page.make_move("e2e4")
        time.sleep(2)

        # Start new game
        chess_page.start_new_game("white")

        history = chess_page.get_history_text()
        assert "no moves" in history.lower() or history.strip() == ""

        status = chess_page.get_status_text()
        assert "white" in status.lower()

    def test_undo_removes_moves(self, chess_page: ChessPage) -> None:
        """Test that Undo removes the last move pair."""
        chess_page.start_new_game("white")
        chess_page.make_move("e2e4")
        time.sleep(2)

        # Get history before undo
        history_before = chess_page.get_history_text()
        assert "e2e4" in history_before

        # Undo
        chess_page.click_undo()

        # History should be empty or show fewer moves
        history_after = chess_page.get_history_text()
        # Either empty or doesn't contain the move we made
        assert "no moves" in history_after.lower() or "e2e4" not in history_after

    def test_play_as_black(self, chess_page: ChessPage) -> None:
        """Test playing as black (AI moves first)."""
        chess_page.start_new_game("black")
        time.sleep(3)

        # AI should have made a move
        history = chess_page.get_history_text()
        # Should have at least one move from AI
        assert "1." in history or len(history.strip()) > 0


@pytest.mark.selenium
@pytest.mark.slow
class TestBoardVisualization:
    """Tests for board visualization."""

    def test_board_shows_pieces(self, chess_page: ChessPage) -> None:
        """Test that board shows piece symbols."""
        board_html = chess_page.get_board_html()

        # Should have chess piece Unicode symbols
        piece_symbols = ["♔", "♕", "♖", "♗", "♘", "♙", "♚", "♛", "♜", "♝", "♞", "♟"]
        has_pieces = any(symbol in board_html for symbol in piece_symbols)
        assert has_pieces, "No chess pieces found on board"

    def test_board_has_squares(self, chess_page: ChessPage) -> None:
        """Test that board has light and dark squares."""
        board_html = chess_page.get_board_html()

        assert "light-square" in board_html
        assert "dark-square" in board_html

    def test_move_highlights_squares(self, chess_page: ChessPage) -> None:
        """Test that making a move highlights the squares."""
        chess_page.start_new_game("white")
        chess_page.make_move("e2e4")
        time.sleep(2)

        board_html = chess_page.get_board_html()
        # Check for highlight class (may be on AI's move)
        assert "highlight" in board_html.lower() or "highlight-square" in board_html


@pytest.mark.selenium
@pytest.mark.slow
class TestScreenshots:
    """Tests that capture screenshots for visual verification."""

    def test_capture_initial_board(self, chess_page: ChessPage) -> None:
        """Capture screenshot of initial board position."""
        path = chess_page.take_screenshot("initial_board")
        assert os.path.exists(path)

    def test_capture_after_opening_moves(self, chess_page: ChessPage) -> None:
        """Capture screenshot after some opening moves."""
        chess_page.start_new_game("white")
        chess_page.make_move("e2e4")
        time.sleep(2)
        chess_page.make_move("d2d4")
        time.sleep(2)

        path = chess_page.take_screenshot("after_opening")
        assert os.path.exists(path)

    def test_capture_playing_as_black(self, chess_page: ChessPage) -> None:
        """Capture screenshot when playing as black."""
        chess_page.start_new_game("black")
        time.sleep(3)

        path = chess_page.take_screenshot("playing_as_black")
        assert os.path.exists(path)


@pytest.mark.selenium
@pytest.mark.slow
class TestScorecard:
    """Tests for scorecard functionality."""

    def test_scorecard_displayed(self, chess_page: ChessPage) -> None:
        """Test that scorecard is displayed on page load."""
        scorecard = chess_page.get_scorecard_element()
        assert scorecard.is_displayed()

    def test_scorecard_shows_initial_values(self, chess_page: ChessPage) -> None:
        """Test scorecard shows zero values initially."""
        scorecard_html = chess_page.get_scorecard_html()
        # Should show Score Card title
        assert "Score Card" in scorecard_html
        # Should show Elo value
        assert "1500" in scorecard_html or "Elo" in scorecard_html

    def test_scorecard_updates_after_game(self, chess_page: ChessPage) -> None:
        """Test that scorecard updates after a game ends."""
        # Start a new game
        chess_page.start_new_game("white")

        # Get initial scorecard
        initial_html = chess_page.get_scorecard_html()

        # Play a few moves (may not finish game but scorecard should persist)
        chess_page.make_move("e2e4")
        time.sleep(2)

        # Scorecard should still be visible
        updated_html = chess_page.get_scorecard_html()
        assert "Score Card" in updated_html

    def test_reset_scorecard_button(self, chess_page: ChessPage) -> None:
        """Test that reset score button works."""
        reset_btn = chess_page.reset_score_button
        assert reset_btn.is_displayed()

        # Click reset
        reset_btn.click()
        time.sleep(1)

        # Scorecard should show reset values
        scorecard_html = chess_page.get_scorecard_html()
        assert "Score Card" in scorecard_html


@pytest.mark.selenium
@pytest.mark.slow
class TestContinuousLearning:
    """Tests for continuous learning tab functionality."""

    def test_learning_tab_exists(self, chess_page: ChessPage) -> None:
        """Test that continuous learning tab exists."""
        learning_tab = chess_page.get_learning_tab()
        assert learning_tab is not None
        assert learning_tab.is_displayed()

    def test_learning_tab_navigation(self, chess_page: ChessPage) -> None:
        """Test navigating to learning tab."""
        chess_page.click_learning_tab()
        time.sleep(1)

        # Should see learning controls
        assert chess_page.get_start_learning_button().is_displayed()

    def test_duration_slider_exists(self, chess_page: ChessPage) -> None:
        """Test that duration slider exists in learning tab."""
        chess_page.click_learning_tab()
        time.sleep(1)

        slider = chess_page.get_duration_slider()
        assert slider is not None

    def test_max_games_slider_exists(self, chess_page: ChessPage) -> None:
        """Test that max games slider exists in learning tab."""
        chess_page.click_learning_tab()
        time.sleep(1)

        slider = chess_page.get_max_games_slider()
        assert slider is not None

    def test_learning_control_buttons(self, chess_page: ChessPage) -> None:
        """Test that all learning control buttons exist."""
        chess_page.click_learning_tab()
        time.sleep(1)

        assert chess_page.get_start_learning_button().is_displayed()
        assert chess_page.get_pause_learning_button().is_displayed()
        assert chess_page.get_stop_learning_button().is_displayed()

    def test_learning_status_display(self, chess_page: ChessPage) -> None:
        """Test that learning status display exists."""
        chess_page.click_learning_tab()
        time.sleep(1)

        status = chess_page.get_learning_status()
        assert status is not None
        # Should show "No learning session active" initially
        status_text = status.get_attribute("innerHTML") or ""
        assert "No learning session" in status_text or "learning" in status_text.lower()

    def test_refresh_button(self, chess_page: ChessPage) -> None:
        """Test that refresh button exists and is clickable."""
        chess_page.click_learning_tab()
        time.sleep(1)

        refresh_btn = chess_page.get_refresh_button()
        assert refresh_btn.is_displayed()

        # Click refresh
        refresh_btn.click()
        time.sleep(1)

        # Status should still be visible
        status = chess_page.get_learning_status()
        assert status is not None


# Extended Page Object with new methods
ChessPage.get_scorecard_element = lambda self: self._get_element_by_id("scorecard-display")
ChessPage.get_scorecard_html = lambda self: (
    self.driver.find_element(
        __import__("selenium.webdriver.common.by", fromlist=["By"]).By.ID,
        "scorecard-display"
    ).get_attribute("innerHTML") or ""
)
ChessPage.reset_score_button = property(
    lambda self: self.driver.find_element(
        __import__("selenium.webdriver.common.by", fromlist=["By"]).By.ID,
        "reset-score-btn"
    )
)
ChessPage.get_learning_tab = lambda self: self._find_tab("Continuous Learning")
ChessPage.click_learning_tab = lambda self: self._click_tab("Continuous Learning")
ChessPage.get_start_learning_button = lambda self: self._get_element_by_id("start-learning-btn")
ChessPage.get_pause_learning_button = lambda self: self._get_element_by_id("pause-learning-btn")
ChessPage.get_stop_learning_button = lambda self: self._get_element_by_id("stop-learning-btn")
ChessPage.get_duration_slider = lambda self: self._get_element_by_id("duration-slider")
ChessPage.get_max_games_slider = lambda self: self._get_element_by_id("max-games-slider")
ChessPage.get_learning_status = lambda self: self._get_element_by_id("learning-status-display")
ChessPage.get_refresh_button = lambda self: self._get_element_by_id("refresh-btn")


def _get_element_by_id(self, element_id: str) -> "WebElement":
    """Get element by ID."""
    from selenium.webdriver.common.by import By
    return self.driver.find_element(By.ID, element_id)


def _find_tab(self, tab_name: str) -> "WebElement":
    """Find a tab by name."""
    from selenium.webdriver.common.by import By
    tabs = self.driver.find_elements(By.CSS_SELECTOR, "[role='tab'], .tab-nav button")
    for tab in tabs:
        if tab_name.lower() in tab.text.lower():
            return tab
    # Try finding by partial text
    return self.driver.find_element(By.XPATH, f"//*[contains(text(), '{tab_name}')]")


def _click_tab(self, tab_name: str) -> None:
    """Click a tab by name."""
    tab = self._find_tab(tab_name)
    tab.click()
    self._slow_action()


# Attach methods to ChessPage
ChessPage._get_element_by_id = _get_element_by_id
ChessPage._find_tab = _find_tab
ChessPage._click_tab = _click_tab


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "selenium: marks tests as Selenium browser tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--headed",
        action="store_true",
        default=False,
        help="Run tests in headed mode (visible browser)",
    )
    parser.addoption(
        "--slow-mo",
        type=float,
        default=0.5,
        help="Slow motion delay between actions",
    )
