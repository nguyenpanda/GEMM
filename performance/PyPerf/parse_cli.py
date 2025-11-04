import argparse

class BM_CliParser:
    
    @classmethod
    def setup(cls, parser: argparse.ArgumentParser):
        parser.add_argument("file", type=str, help="JSON file")
        parser.add_argument("--show_img", action='store_true', help="Display Matplotlib plot")
        parser.add_argument("--legend", action='store_true', help="Add legend to the plot")
        parser.add_argument("--display_console", action='store_true', help="Display data to console")
        return parser
    
    def __init__(self, parser: argparse.ArgumentParser | None = None):
        self.description = 'A sample script demonstrating argparse.'
        self.parser = self.setup(
        	argparse.ArgumentParser(description=self.description) if parser is None else parser
		)
        self.args = self.parser.parse_args()
        