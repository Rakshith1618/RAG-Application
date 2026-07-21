import sys
import os

# Add root project directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Export for Vercel Serverless Function
app = app
