"""Port Configuration Validator"""
import socket
from typing import List, Dict
from pathlib import Path

class PortValidator:
    def validate(self) -> List[Dict]:
        """Validate port configuration"""
        issues = []
        # Find app port
        app_port = self._find_app_port()
        if app_port and self._is_port_in_use(app_port):
            issues.append({
                'type': 'port_in_use',
                'severity': 'medium',
                'message': f'Port {app_port} already in use'
            })
        return issues
    
    def _find_app_port(self) -> int:
        """Find the port the app uses"""
        return 8000  # Stub
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is in use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        except:
            return False
