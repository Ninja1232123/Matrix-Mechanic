"""K8sGenerator"""
from pathlib import Path
class K8sGenerator:
    def generate_dockerfile(self): return Path("Dockerfile")
    def generate_docker_compose(self): return Path("docker-compose.yml")
    def generate_env(self): return Path(".env")
    def generate_k8s_manifests(self): return []
