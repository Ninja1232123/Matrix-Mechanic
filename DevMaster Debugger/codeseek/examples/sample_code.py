"""
Sample Python code for testing CodeSeek.
"""

import json
import requests
from typing import Optional, Dict, List


def validate_email(email: str) -> bool:
    """
    Validate an email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def fetch_user_data(user_id: int) -> Optional[Dict]:
    """
    Fetch user data from API.

    Args:
        user_id: ID of the user to fetch

    Returns:
        User data dictionary or None if not found
    """
    try:
        response = requests.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching user data: {e}")
        return None


class UserManager:
    """Manages user operations."""

    def __init__(self, api_key: str):
        """Initialize user manager with API key."""
        self.api_key = api_key
        self.base_url = "https://api.example.com"

    def create_user(self, email: str, name: str) -> Optional[Dict]:
        """
        Create a new user.

        Args:
            email: User's email address
            name: User's full name

        Returns:
            Created user data or None on error
        """
        if not validate_email(email):
            raise ValueError(f"Invalid email: {email}")

        payload = {"email": email, "name": name}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.post(
                f"{self.base_url}/users",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error creating user: {e}")
            return None

    def list_users(self, limit: int = 10) -> List[Dict]:
        """
        List all users.

        Args:
            limit: Maximum number of users to return

        Returns:
            List of user dictionaries
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(
                f"{self.base_url}/users",
                params={"limit": limit},
                headers=headers
            )
            response.raise_for_status()
            return response.json().get("users", [])
        except requests.RequestException as e:
            print(f"Error listing users: {e}")
            return []


async def async_fetch_data(url: str) -> Optional[Dict]:
    """
    Asynchronously fetch data from a URL.

    Args:
        url: URL to fetch

    Returns:
        JSON data or None on error
    """
    import aiohttp

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Error fetching data: {e}")
            return None


def parse_json_file(file_path: str) -> Optional[Dict]:
    """
    Parse a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data or None on error
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error parsing JSON file: {e}")
        return None
