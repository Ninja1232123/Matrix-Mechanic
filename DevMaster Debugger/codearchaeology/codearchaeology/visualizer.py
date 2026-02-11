"""
Beautiful terminal visualizations for code archaeology data.

Creates ASCII art heatmaps, graphs, and charts.
"""

from typing import List, Tuple, Dict
from datetime import datetime, timedelta


class Visualizer:
    """Creates beautiful terminal visualizations."""

    # Color codes for terminal
    COLORS = {
        'red': '\033[91m',
        'yellow': '\033[93m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'gray': '\033[90m',
        'reset': '\033[0m',
    }

    # Heat levels
    HEAT_CHARS = ' ▁▂▃▄▅▆▇█'
    HEAT_COLORS = ['green', 'yellow', 'red']

    @classmethod
    def hotspot_heatmap(cls, hotspots: List[Tuple[str, int]], max_width: int = 50) -> str:
        """
        Create a heatmap visualization of code hotspots.

        Args:
            hotspots: List of (file_path, change_count) tuples
            max_width: Maximum width of the bars

        Returns:
            Formatted heatmap string
        """
        if not hotspots:
            return "No hotspots to visualize"

        max_changes = max(count for _, count in hotspots)
        output = []

        output.append("🔥 Code Hotspot Heatmap\n")

        for file_path, count in hotspots[:20]:  # Top 20
            # Calculate bar width
            bar_width = int((count / max_changes) * max_width)

            # Determine color based on intensity
            if count / max_changes > 0.7:
                color = 'red'
                risk = '🔴'
            elif count / max_changes > 0.4:
                color = 'yellow'
                risk = '🟡'
            else:
                color = 'green'
                risk = '🟢'

            # Create bar
            bar = '█' * bar_width

            # Format output
            file_short = cls._shorten_path(file_path, 40)
            line = f"{risk} {file_short:40} {cls.COLORS[color]}{bar}{cls.COLORS['reset']} {count}"
            output.append(line)

        return '\n'.join(output)

    @classmethod
    def change_timeline(cls, changes_by_date: Dict[str, int], days: int = 30) -> str:
        """
        Create a timeline visualization of changes.

        Args:
            changes_by_date: Dictionary of date -> change count
            days: Number of days to show

        Returns:
            Formatted timeline
        """
        if not changes_by_date:
            return "No changes to visualize"

        output = []
        output.append(f"📈 Change Activity (Last {days} Days)\n")

        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Fill in missing dates
        date_range = {}
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            date_range[date_str] = changes_by_date.get(date_str, 0)
            current += timedelta(days=1)

        max_changes = max(date_range.values()) if date_range.values() else 1

        # Group by week for readability
        weeks = []
        week = []
        for date_str, count in sorted(date_range.items()):
            week.append(count)
            if len(week) == 7:
                weeks.append(week)
                week = []

        if week:  # Add remaining days
            weeks.append(week)

        # Create visualization
        for week_data in weeks:
            week_str = ""
            for count in week_data:
                if count == 0:
                    char = '·'
                    color = 'gray'
                else:
                    intensity = count / max_changes
                    char_idx = min(int(intensity * len(cls.HEAT_CHARS)), len(cls.HEAT_CHARS) - 1)
                    char = cls.HEAT_CHARS[char_idx]

                    if intensity > 0.7:
                        color = 'red'
                    elif intensity > 0.3:
                        color = 'yellow'
                    else:
                        color = 'green'

                week_str += f"{cls.COLORS[color]}{char}{cls.COLORS['reset']}"

            output.append(week_str)

        output.append("\nLegend: " + ' '.join([f"{cls.HEAT_CHARS[i]}" for i in range(0, len(cls.HEAT_CHARS), 2)]))

        return '\n'.join(output)

    @classmethod
    def coupling_graph(cls, coupling_pairs: List[Tuple[str, str, float]], top_n: int = 10) -> str:
        """
        Visualize temporal coupling between files.

        Args:
            coupling_pairs: List of (file1, file2, coupling_score) tuples
            top_n: Number of pairs to show

        Returns:
            Formatted coupling graph
        """
        if not coupling_pairs:
            return "No coupling to visualize"

        output = []
        output.append("🔗 Temporal Coupling Graph\n")

        for file1, file2, score in coupling_pairs[:top_n]:
            # Create connection strength indicator
            strength = int(score * 10)
            if strength >= 8:
                indicator = '═══════'
                color = 'red'
                label = 'STRONG'
            elif strength >= 5:
                indicator = '──────'
                color = 'yellow'
                label = 'MEDIUM'
            else:
                indicator = '······'
                color = 'green'
                label = 'WEAK'

            file1_short = cls._shorten_path(file1, 25)
            file2_short = cls._shorten_path(file2, 25)

            line = f"{file1_short:25} {cls.COLORS[color]}{indicator}{cls.COLORS['reset']} {file2_short:25} ({label} {score:.2f})"
            output.append(line)

        output.append("\n💡 Files that change together often may be tightly coupled")

        return '\n'.join(output)

    @classmethod
    def knowledge_distribution(cls, author_knowledge: Dict[str, int]) -> str:
        """
        Visualize knowledge distribution across authors.

        Args:
            author_knowledge: Dictionary of author -> file count

        Returns:
            Formatted knowledge distribution
        """
        if not author_knowledge:
            return "No knowledge distribution data"

        output = []
        output.append("👥 Knowledge Distribution\n")

        total_files = sum(author_knowledge.values())
        max_files = max(author_knowledge.values())

        for author, file_count in sorted(author_knowledge.items(), key=lambda x: x[1], reverse=True):
            percentage = (file_count / total_files) * 100
            bar_width = int((file_count / max_files) * 30)

            # Color based on concentration
            if percentage > 50:
                color = 'red'
                warning = '⚠️ '
            elif percentage > 30:
                color = 'yellow'
                warning = '⚡'
            else:
                color = 'green'
                warning = '✅'

            bar = '█' * bar_width
            author_short = (author[:20] + '...') if len(author) > 20 else author

            line = f"{warning} {author_short:25} {cls.COLORS[color]}{bar}{cls.COLORS['reset']} {file_count} files ({percentage:.1f}%)"
            output.append(line)

        # Calculate bus factor
        cumulative = 0
        authors_needed = 0
        for _, count in sorted(author_knowledge.values(), reverse=True):
            cumulative += count
            authors_needed += 1
            if cumulative >= total_files * 0.5:
                break

        output.append(f"\n🚌 Bus Factor: {authors_needed}")
        if authors_needed == 1:
            output.append("   ⚠️  CRITICAL: Only 1 person knows 50%+ of the code!")
        elif authors_needed == 2:
            output.append("   ⚠️  RISKY: Only 2 people know 50%+ of the code")
        else:
            output.append(f"   ✅ {authors_needed} people needed to cover 50% of codebase")

        return '\n'.join(output)

    @classmethod
    def complexity_trend(cls, complexity_data: List[Tuple[str, int]], threshold: int = 10) -> str:
        """
        Visualize code complexity distribution.

        Args:
            complexity_data: List of (file_path, complexity_score) tuples
            threshold: Complexity threshold for warnings

        Returns:
            Formatted complexity visualization
        """
        if not complexity_data:
            return "No complexity data"

        output = []
        output.append("📊 Code Complexity Distribution\n")

        # Buckets
        buckets = {
            'Simple (1-5)': 0,
            'Moderate (6-10)': 0,
            'Complex (11-20)': 0,
            'Very Complex (21+)': 0
        }

        high_complexity_files = []

        for file_path, complexity in complexity_data:
            if complexity <= 5:
                buckets['Simple (1-5)'] += 1
            elif complexity <= 10:
                buckets['Moderate (6-10)'] += 1
            elif complexity <= 20:
                buckets['Complex (11-20)'] += 1
                high_complexity_files.append((file_path, complexity))
            else:
                buckets['Very Complex (21+)'] += 1
                high_complexity_files.append((file_path, complexity))

        total = sum(buckets.values())

        for bucket, count in buckets.items():
            percentage = (count / total * 100) if total > 0 else 0
            bar_width = int(percentage / 2)  # Scale to 50 chars max
            bar = '█' * bar_width

            if 'Simple' in bucket:
                color = 'green'
            elif 'Moderate' in bucket:
                color = 'cyan'
            elif 'Complex' in bucket:
                color = 'yellow'
            else:
                color = 'red'

            line = f"{bucket:20} {cls.COLORS[color]}{bar}{cls.COLORS['reset']} {count} ({percentage:.1f}%)"
            output.append(line)

        if high_complexity_files:
            output.append("\n⚠️  High Complexity Files:")
            for file_path, complexity in sorted(high_complexity_files, key=lambda x: x[1], reverse=True)[:5]:
                file_short = cls._shorten_path(file_path, 40)
                output.append(f"   • {file_short}: {complexity}")

        return '\n'.join(output)

    @classmethod
    def churn_vs_complexity(cls, data: List[Tuple[str, int, int]], max_points: int = 20) -> str:
        """
        Create a scatter plot of churn vs complexity.

        Args:
            data: List of (file_path, churn, complexity) tuples
            max_points: Maximum number of points to show

        Returns:
            Formatted scatter plot
        """
        if not data:
            return "No data to visualize"

        output = []
        output.append("📉 Code Churn vs Complexity\n")
        output.append("   (Files in top-right quadrant need attention!)\n")

        # Normalize data
        max_churn = max(churn for _, churn, _ in data)
        max_complexity = max(complexity for _, _, complexity in data)

        # Create 2D grid (20x10)
        width, height = 40, 10
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Plot points
        plotted_files = []
        for file_path, churn, complexity in data[:max_points]:
            if max_churn > 0 and max_complexity > 0:
                x = int((churn / max_churn) * (width - 1))
                y = height - 1 - int((complexity / max_complexity) * (height - 1))

                # Determine risk level
                if churn / max_churn > 0.6 and complexity / max_complexity > 0.6:
                    char = '●'  # High risk
                    color = 'red'
                elif churn / max_churn > 0.3 or complexity / max_complexity > 0.3:
                    char = '●'  # Medium risk
                    color = 'yellow'
                else:
                    char = '●'  # Low risk
                    color = 'green'

                grid[y][x] = f"{cls.COLORS[color]}{char}{cls.COLORS['reset']}"
                plotted_files.append((file_path, churn, complexity, color))

        # Draw grid
        output.append("  Complexity")
        output.append("  ↑")

        for row in grid:
            output.append("  │" + ''.join(row))

        output.append("  └" + "─" * width + "→ Churn")

        # Legend
        output.append("\n  Legend:")
        output.append(f"  {cls.COLORS['red']}●{cls.COLORS['reset']} High Risk (high churn + high complexity)")
        output.append(f"  {cls.COLORS['yellow']}●{cls.COLORS['reset']} Medium Risk")
        output.append(f"  {cls.COLORS['green']}●{cls.COLORS['reset']} Low Risk")

        return '\n'.join(output)

    @staticmethod
    def _shorten_path(path: str, max_length: int) -> str:
        """Shorten a file path to fit in max_length."""
        if len(path) <= max_length:
            return path

        parts = path.split('/')
        if len(parts) <= 2:
            return path[:max_length-3] + '...'

        # Try to keep first and last parts
        result = parts[0] + '/.../' + parts[-1]
        if len(result) <= max_length:
            return result

        # Just truncate
        return path[:max_length-3] + '...'
